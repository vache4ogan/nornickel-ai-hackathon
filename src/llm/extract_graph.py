import json
import time
import requests
from typing import List, Dict, Optional
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

# Ваши доступы от Yandex AI Studio
YANDEX_API_KEY = "AQVN3rool3OmWazMSEn_RLApg5jjrTTizTeAonhc"
FOLDER_ID = "b1ggusvist6c2sia1dno"

# 1. Строгая схема Pydantic
class Entity(BaseModel):
    id: str = Field(description="РЕАЛЬНОЕ название из текста в именительном падеже (например: 'Медь', 'Печь Ванюкова', 'Температура'). СТРОГО ЗАПРЕЩЕНО использовать заглушки типа '1', '2', 'e1', 'material_1'!")
    type: str = Field(description="Тип сущности: Material, Process, Equipment или Metric")

class Relation(BaseModel):
    source: str = Field(description="Название исходной сущности (id)")
    target: str = Field(description="Название целевой сущности (id)")
    type: str = Field(description="Тип связи: USES (использует), PRODUCES (производит), HAS_PARAMETER (имеет параметр)")
    properties: Dict[str, str] = Field(
        default_factory=dict,
        description="Краткие числовые параметры. Например {'value': '200 мг/л'} или {'value': '80 °C'}. ЗАПРЕЩЕНО писать сюда длинные предложения!"
    )
class GraphExtraction(BaseModel):
    entities: List[Entity] = Field(description="Список найденных сущностей")
    relations: List[Relation] = Field(description="Список связей")

SYSTEM_PROMPT = """Ты — строгий эксперт-металлург и Data Scientist. Твоя задача — извлекать знания из текста для построения графа.

РАЗРЕШЕННЫЕ СУЩНОСТИ (Entities): Material, Process, Equipment, Metric.
РАЗРЕШЕННЫЕ СВЯЗИ (Relations): USES, PRODUCES, HAS_PARAMETER.

КРИТИЧЕСКИЕ ПРАВИЛА (ШТРАФ ЗА НАРУШЕНИЕ):
1. ИМЕНА УЗЛОВ: Извлекай реальные термины в именительном падеже (например, "Серная кислота", "Автоклав"). 
2. ЗАПРЕТ ЗАГЛУШЕК: Никогда не используй абстрактные ID вроде "1", "2", "e1", "material_1", "metric_2". Если не можешь найти реальное название — лучше вообще не создавай этот узел.
3. ПАРАМЕТРЫ: В свойствах связей (properties) пиши ТОЛЬКО короткие значения и цифры (например, {"value": "89%"}, {"value": "pH 6.5-8.3"}). Запрещено копировать туда длинные предложения из текста.
4. Если текст — это оглавление или мусор, возвращай пустые списки.
"""

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
    
    result_text = "" # Инициализируем переменную для логов
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        # Если закончились деньги на балансе или упал сервер Яндекса
        if response.status_code != 200:
            print(f"\n[Ошибка Сервера {response.status_code}]: {response.text}")
            return None
            
        result_text = response.json()['result']['alternatives'][0]['message']['text']
        
        if not result_text.strip():
            # Яндекс иногда возвращает пустоту из-за внутренних фильтров безопасности
            return None
            
        # --- БРОНЕБОЙНЫЙ ПАРСИНГ JSON ---
        # Ищем начало и конец JSON-объекта, игнорируя любой текст до и после
        start_idx = result_text.find('{')
        end_idx = result_text.rfind('}')
        
        if start_idx == -1 or end_idx == -1:
            print(f"\n[Отклонено] Модель не выдала JSON. Ответ: {result_text[:100]}...")
            return None
            
        # Вырезаем ровно то, что находится между скобками
        clean_json = result_text[start_idx:end_idx+1]
        
        parsed_json = json.loads(clean_json)
        graph_obj = GraphExtraction(**parsed_json)
        return graph_obj
        
    except json.JSONDecodeError:
        print(f"\n[Ошибка JSON]: Не удалось распарсить. Сырой текст: {result_text[:150]}...")
    except ValidationError as ve:
        print(f"\n[Ошибка Pydantic]: Неверный формат данных от модели.")
    except Exception as e:
        print(f"\n[Неизвестная ошибка]: {e}")
    
    return None

def run_extraction():
    input_file = "data/processed/chunks_sample.jsonl"
    output_file = "data/processed/graph_triplets.jsonl"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Файл {input_file} не найден. Сначала создайте выборку.")
        return
        
    print(f"Отправляем {len(lines)} чанков в YandexGPT...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines, desc="Обработка LLM"):
            chunk_data = json.loads(line)
            
            # Обработка через YandexGPT + Pydantic
            graph_obj = extract_graph_from_yandex(chunk_data["text"])
            
            if graph_obj:
                # Превращаем Python-объект обратно в словарь
                final_dict = graph_obj.model_dump()
                final_dict["source_document"] = chunk_data["source"]
                
                # Сохраняем в файл
                f_out.write(json.dumps(final_dict, ensure_ascii=False) + "\n")
            
            # Пауза, чтобы не получить бан по Rate Limit (429 Too Many Requests)
            time.sleep(1.5)

if __name__ == "__main__":
    run_extraction()