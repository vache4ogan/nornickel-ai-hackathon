# Файл: src/graph/import_graph.py
import json
from neo4j import GraphDatabase
from tqdm import tqdm

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "hackathon2024"

class GraphImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    @staticmethod
    def _safe_name(name_str):
        # Очистка имени от лишних кавычек и мусора
        return str(name_str).strip().replace('"', '')

    def process_triplets(self, tx, data):
        source_doc = self._safe_name(data.get("source_document", "Unknown"))
        
        # 1. Создаем узел документа-источника
        tx.run("MERGE (d:Document {name: $doc_name})", doc_name=source_doc)

        # 2. Импортируем Сущности (Nodes)
        entities = data.get("entities", [])
        for ent in entities:
            ent_type = ent.get("type", "Entity").capitalize()
            ent_name = self._safe_name(ent.get("id", ""))
            
            if not ent_name:
                continue

            # Защита динамического Cypher от спецсимволов в названии лейбла
            safe_type = ''.join(e for e in ent_type if e.isalnum())
            if safe_type not in ["Material", "Process", "Equipment", "Metric"]:
                safe_type = "Entity"

            # Создаем узел и связываем его с документом
            tx.run(f"MERGE (n:{safe_type} {{name: $name}})", name=ent_name)
            tx.run(f"""
                MATCH (n:{safe_type} {{name: $name}})
                MATCH (d:Document {{name: $doc_name}})
                MERGE (n)-[:MENTIONED_IN]->(d)
            """, name=ent_name, doc_name=source_doc)

        # 3. Импортируем Связи (Relations)
        relations = data.get("relations", [])
        for rel in relations:
            source = self._safe_name(rel.get("source", ""))
            target = self._safe_name(rel.get("target", ""))
            rel_type = rel.get("type", "RELATED_TO").upper().strip()
            props = rel.get("properties", {})

            if not source or not target:
                continue

            safe_rel_type = ''.join(e for e in rel_type if e.isalnum() or e == '_')
            if not safe_rel_type:
                safe_rel_type = "RELATED_TO"

            # Создаем связь между любыми двумя узлами, совпавшими по имени
            tx.run(f"""
                MATCH (s {{name: $source}})
                MATCH (t {{name: $target}})
                MERGE (s)-[r:{safe_rel_type}]->(t)
                SET r += $props
            """, source=source, target=target, props=props)

    def import_data(self, jsonl_file):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        print(f"Начинаем загрузку {len(lines)} ответов от LLM в Neo4j...")
        
        with self.driver.session() as session:
            for line in tqdm(lines):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    session.execute_write(self.process_triplets, data)
                except Exception as e:
                    print(f"Пропущена строка из-за ошибки парсинга JSON: {e}")

if __name__ == "__main__":
    importer = GraphImporter(URI, USER, PASSWORD)
    # Завтра раскомментируете строку ниже, когда появится файл с результатами работы LLM:
    # importer.import_data("data/processed/graph_triplets.jsonl")
    importer.close()