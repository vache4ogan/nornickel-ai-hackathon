import json
import requests
from tqdm import tqdm
import time

# Завтра вставите сюда данные от организаторов
YANDEX_API_KEY = "ВАШ_АПИ_КЛЮЧ" 
FOLDER_ID = "ВАШ_FOLDER_ID"

SYSTEM_PROMPT = """Ты — эксперт-металлург. Твоя задача — извлекать знания из текста в виде графа.
Разрешенные сущности (Entities): Material, Process, Equipment, Metric.
Разрешенные связи (Relations): USES, PRODUCES, HAS_PARAMETER.

ОТВЕЧАЙ СТРОГО В ФОРМАТЕ JSON, без маркдауна и лишних слов:
{
  "entities": [{"id": "Название", "type": "Тип"}],
  "relations": [{"source": "Название_1", "target": "Название_2", "type": "Тип_связи", "properties": {"value": "значение"}}]
}"""

def process_chunk_yandex(chunk_text):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "x-folder-id": FOLDER_ID
    }
    data = {
        # Используем последнюю версию модели YandexGPT
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest", 
        "completionOptions": {
            "stream": False,
            "temperature": 0.1, # Минимум фантазии, строгий ответ
            "maxTokens": "2000"
        },
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user", "text": f"Извлеки граф:\n{chunk_text}"}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        # Достаем текст из ответа Яндекса
        result_text = response.json()['result']['alternatives'][0]['message']['text']
        
        # Очищаем от возможных маркдаун-кавычек (```json ... ```)
        result_text = result_text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(result_text)
        
    except Exception as e:
        print(f"Ошибка API Яндекса: {e}")
        return None

def run_extraction():
    input_file = "data/processed/chunks_sample.jsonl" # Берем нашу мини-выборку!
    output_file = "data/processed/graph_triplets.jsonl"
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Отправляем {len(lines)} чанков в YandexGPT...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines):
            chunk_data = json.loads(line)
            graph_data = process_chunk_yandex(chunk_data["text"])
            
            if graph_data:
                graph_data["source_document"] = chunk_data["source"]
                f_out.write(json.dumps(graph_data, ensure_ascii=False) + "\n")
            
            # Небольшая пауза, чтобы не словить ошибку Rate Limit (Too Many Requests)
            time.sleep(1) 

if __name__ == "__main__":
    print("Готов к работе с Yandex API!")
    # run_extraction()