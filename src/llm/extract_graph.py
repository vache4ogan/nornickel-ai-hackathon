import json
import time
import requests
import hashlib
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Ваши доступы от Yandex AI Studio
YANDEX_API_KEY = "AQVN3rool3OmWazMSEn_RLApg5jjrTTizTeAonhc"
FOLDER_ID = "b1ggusvist6c2sia1dno"

class Entity(BaseModel):
    id: str = Field(description="РЕАЛЬНОЕ название из текста (например: 'Медь', 'Печь Ванюкова', 'Температура'). СТРОГО ЗАПРЕЩЕНО использовать заглушки типа '1', 'e1'!")
    type: str = Field(description="Строго один из: Material, Process, Equipment, Property, Experiment, Publication, Expert, Facility")

class Relation(BaseModel):
    source: str = Field(description="Название исходной сущности (id)")
    target: str = Field(description="Название целевой сущности (id)")
    type: str = Field(description="Строго один из: uses_material, operates_at_condition, produces_output, described_in, validated_by, contradicts")
    properties: Dict[str, str] = Field(
        default_factory=dict,
        description="Краткие числовые параметры. Например {'value': '200 мг/л'}."
    )

class GraphExtraction(BaseModel):
    entities: List[Entity] = Field(description="Список найденных сущностей")
    relations: List[Relation] = Field(description="Список связей")

SYSTEM_PROMPT = """Ты — автоматический алгоритм-экстрактор для металлургической отрасли. Твоя ЕДИНСТВЕННАЯ задача — переводить текст в строгий JSON.

РАЗРЕШЕННЫЕ СУЩНОСТИ (Entities): Material, Process, Equipment, Property, Experiment, Publication, Expert, Facility.
РАЗРЕШЕННЫЕ СВЯЗИ (Relations): uses_material, operates_at_condition, produces_output, described_in, validated_by, contradicts.

ПРИМЕР ИДЕАЛЬНОГО ОТВЕТА (ты обязан отвечать именно так, начиная с символа {):
{
  "entities": [
    {"id": "Флотационная машина", "type": "Equipment"},
    {"id": "Медная руда", "type": "Material"},
    {"id": "Извлечение", "type": "Property"}
  ],
  "relations": [
    {"source": "Флотационная машина", "target": "Медная руда", "type": "uses_material", "properties": {}},
    {"source": "Флотационная машина", "target": "Извлечение", "type": "operates_at_condition", "properties": {"value": "89 %"}}
  ]
}

КРИТИЧЕСКОЕ ПРАВИЛО: ЗАПРЕЩЕНО писать слова "ENTITIES:", рисовать маркдаун-списки или использовать заглушки. Если не нашел данных — верни пустые списки. Выдавай ТОЛЬКО валидный JSON."""

class APIError(Exception):
    pass

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((APIError, requests.exceptions.RequestException))
)
def extract_graph_from_yandex(chunk_text: str) -> Optional[GraphExtraction]:
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": FOLDER_ID
    }
    data = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.1, 
            "maxTokens": "2000"
        },
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user", "text": f"Извлеки граф из текста:\n{chunk_text}"}
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code != 200:
        error_msg = f"Ошибка Сервера {response.status_code}: {response.text}"
        print(f"\n[{error_msg}]")
        if response.status_code in [429, 503, 500, 502, 504]:
            raise APIError(error_msg)
        return None
        
    result_text = response.json().get('result', {}).get('alternatives', [{}])[0].get('message', {}).get('text', '')
    
    if not result_text.strip():
        return None
        
    start_idx = result_text.find('{')
    end_idx = result_text.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        return None
        
    clean_json = result_text[start_idx:end_idx+1]
    
    try:
        parsed_json = json.loads(clean_json)
        graph_obj = GraphExtraction(**parsed_json)
        return graph_obj
    except (json.JSONDecodeError, ValidationError):
        return None

def get_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def run_extraction():
    input_file = "data/processed/chunks.jsonl"
    if not os.path.exists(input_file):
        input_file = "data/processed/chunks_sample.jsonl"
        
    output_file = "data/processed/graph_triplets.jsonl"
    
    processed_hashes = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if "chunk_text" in data:
                        processed_hashes.add(get_text_hash(data["chunk_text"]))
                except json.JSONDecodeError:
                    pass
    
    print(f"Уже обработано чанков: {len(processed_hashes)}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Файл {input_file} не найден.")
        return
        
    chunks_to_process = []
    for line in lines:
        if not line.strip(): continue
        try:
            chunk_data = json.loads(line)
            if "text" in chunk_data:
                h = get_text_hash(chunk_data["text"])
                if h not in processed_hashes:
                    chunks_to_process.append(chunk_data)
        except json.JSONDecodeError:
            continue
            
    print(f"Осталось обработать чанков: {len(chunks_to_process)}")
    
    if not chunks_to_process:
        print("Всё уже обработано! 🎉")
        return

    with open(output_file, 'a', encoding='utf-8') as f_out:
        for chunk_data in tqdm(chunks_to_process, desc="Обработка LLM"):
            try:
                graph_obj = extract_graph_from_yandex(chunk_data["text"])
            except Exception as e:
                print(f"\n[Критическая ошибка]: {e}")
                continue
            
            if graph_obj:
                final_dict = graph_obj.model_dump()
                final_dict["source_document"] = chunk_data.get("source", "unknown")
                final_dict["chunk_text"] = chunk_data["text"]
                
                f_out.write(json.dumps(final_dict, ensure_ascii=False) + "\n")
                f_out.flush()
            
            time.sleep(0.5)

if __name__ == "__main__":
    run_extraction()