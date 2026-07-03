import random
import json
from pathlib import Path

INPUT_FILE = Path("data/processed/chunks.jsonl")
OUTPUT_FILE = Path("data/processed/chunks_sample.jsonl")

# Сколько чанков взять для MVP (3000 - оптимально для хакатона)
SAMPLE_SIZE = 3000 

def create_random_sample():
    if not INPUT_FILE.exists():
        print(f"Файл {INPUT_FILE} не найден!")
        return

    print(f"Чтение всех данных из {INPUT_FILE} (это займет пару секунд)...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    total_chunks = len(all_lines)
    print(f"Всего доступно чанков: {total_chunks}")

    if total_chunks <= SAMPLE_SIZE:
        print("Чанков меньше или равно размеру выборки. Копируем всё.")
        sampled_lines = all_lines
    else:
        print(f"Выбираем {SAMPLE_SIZE} случайных чанков...")
        sampled_lines = random.sample(all_lines, SAMPLE_SIZE)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in sampled_lines:
            f_out.write(line)

    print(f"✅ Готово! Случайная выборка из {len(sampled_lines)} чанков сохранена в {OUTPUT_FILE}")

if __name__ == "__main__":
    create_random_sample()