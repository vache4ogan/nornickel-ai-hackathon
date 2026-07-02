# Файл: src/graph/init_db.py
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "hackathon2024"

def init_database():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    # Запросы на создание ограничений уникальности по именам
    constraints = [
        "CREATE CONSTRAINT material_name IF NOT EXISTS FOR (m:Material) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT process_name IF NOT EXISTS FOR (p:Process) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT equipment_name IF NOT EXISTS FOR (e:Equipment) REQUIRE e.name IS UNIQUE",
        "CREATE CONSTRAINT metric_name IF NOT EXISTS FOR (m:Metric) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT doc_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE"
    ]
    
    with driver.session() as session:
        for query in constraints:
            try:
                session.run(query)
                print(f"Успешно выполнено: {query.split('FOR')[0]}")
            except Exception as e:
                print(f"Ошибка при создании ограничения: {e}")
                
    driver.close()
    print("\nБаза данных Neo4j успешно инициализирована!")

if __name__ == "__main__":
    init_database()