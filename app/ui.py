import streamlit as st
import json
import os
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import difflib
import re
from groq import Groq

st.set_page_config(layout="wide", page_title="Knowledge Graph RAG")

# --- Конфигурация API Groq ---
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

DATA_PATH = "data/processed/cleaned_graph_triplets.jsonl"

COLOR_MAP = {
    "Material": "#2ecc71",
    "Process": "#e67e22",
    "Equipment": "#3498db",
    "Property": "#9b59b6",
    "Experiment": "#e74c3c",
    "Publication": "#34495e",
    "Expert": "#1abc9c",
    "Facility": "#f1c40f"
}
DEFAULT_COLOR = "#95a5a6"

STOP_WORDS = {
    "какие", "какой", "как", "что", "где", "для", "есть", "это", "методы", 
    "метод", "процесс", "оборудование", "покажи", "расскажи", "найти", 
    "подходят", "применяются", "из", "на", "в", "к", "с"
}

@st.cache_data
def load_graph():
    G = nx.DiGraph()
    if not os.path.exists(DATA_PATH):
        return G
        
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                chunk = json.loads(line)
                
                # Достаем источник из чанка
                doc_source = chunk.get("source_document", "База знаний R&D")
                
                for ent in chunk.get("entities", []):
                    G.add_node(ent["id"], type=ent.get("type", "Unknown"))
                    
                for rel in chunk.get("relations", []):
                    G.add_edge(
                        rel["source"], 
                        rel["target"], 
                        label=rel["type"],
                        properties=rel.get("properties", {}),
                        doc=doc_source  # Сохраняем источник в ребро
                    )
            except json.JSONDecodeError:
                pass
    return G

def ask_groq_llm(context_text: str, question: str) -> str:
    system_prompt = (
        "Ты — Senior R&D Аналитик в горно-металлургической компании. Твоя задача — давать развернутые, связные и профессиональные ответы на основе предоставленного графа знаний.\n"
        "ПРАВИЛА:\n"
        "1. Синтезируй информацию! Пиши красивым, читаемым текстом (абзацами), объединяй похожие факты. Не делай слепые списки из сырых узлов.\n"
        "2. Фильтруй мусор: игнорируй непонятные аббревиатуры, обрывки слов (например, 'Co', 'apada') или нерелевантные узлы, если они не несут смысла.\n"
        "3. ЦИТИРОВАНИЕ ОБЯЗАТЕЛЬНО: вставляй [Источник: Название_файла] прямо в текст предложений, подтверждая свои выводы.\n"
        "4. Если пользователь пишет не вопрос, а просто термин (например, 'медная руда'), сделай связную аналитическую сводку по этому термину (где применяется, какими методами обрабатывается)."
    )
    
    user_prompt = f"Контекст из Графа Знаний:\n{context_text}\n\nВопрос пользователя: {question}"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM недоступна из-за ошибки API: {e}. Вот найденный контекст из графа:"

def extract_subgraph(G: nx.DiGraph, query: str, hops: int = 1):
    words = re.findall(r'\b\w+\b', query.lower())
    meaningful_words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    
    found_nodes = set()
    all_graph_nodes = list(G.nodes())
    
    for word in meaningful_words:
        matches = difflib.get_close_matches(word, [n.lower() for n in all_graph_nodes], n=5, cutoff=0.6)
        for match in matches:
            for node in all_graph_nodes:
                if node.lower() == match:
                    found_nodes.add(node)
                    
    for node in all_graph_nodes:
        if len(node) > 2 and node.lower() in query.lower():
            if bool(re.search(r'\b' + re.escape(node.lower()) + r'\b', query.lower())):
                found_nodes.add(node)
            
    found_nodes_list = list(found_nodes)
    if not found_nodes_list:
        return None, []
        
    subgraph_nodes = set(found_nodes_list)
    for _ in range(hops):
        current_layer = list(subgraph_nodes)
        for node in current_layer:
            subgraph_nodes.update(G.predecessors(node))
            subgraph_nodes.update(G.successors(node))
            
    return G.subgraph(subgraph_nodes), found_nodes_list

def render_pyvis_graph(subgraph: nx.DiGraph):
    net = Network(height="400px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=150)
    
    for node_id in subgraph.nodes():
        node_type = subgraph.nodes[node_id].get("type", "Unknown")
        color = COLOR_MAP.get(node_type, DEFAULT_COLOR)
        net.add_node(
            node_id, 
            label=node_id, 
            title=f"Тип: {node_type}",
            color=color,
            size=20
        )
        
    for u, v, data in subgraph.edges(data=True):
        label = data.get("label", "")
        # Добавляем инфу об источнике в тултип ребра
        doc = data.get("doc", "")
        edge_title = f"{label} (Источник: {doc})" if doc else label
        net.add_edge(u, v, title=edge_title, label=label, color="#7f8c8d")
        
    path = '/tmp'
    os.makedirs(path, exist_ok=True)
    file_path = f'{path}/rag_subgraph.html'
    net.save_graph(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    components.html(source_code, height=420)

# --- УИ Приложения ---
st.title("🤖 Graph RAG Ассистент")
st.markdown("Задавайте вопросы по металлургическим процессам. Система найдет ответ в графе знаний.")

with st.spinner("Загрузка графа..."):
    G = load_graph()

if len(G) == 0:
    st.error("Граф пуст! Сначала запустите пайплайн экстракции и очистки.")
    st.stop()

st.sidebar.success(f"Граф загружен: {G.number_of_nodes()} узлов, {G.number_of_edges()} связей.")
st.sidebar.markdown("""
**Доступные типы сущностей:**  
Material, Process, Equipment, Property, Experiment, Publication, Expert, Facility.
""")

prompt = st.text_input("Ваш вопрос:", placeholder="Например: Какие методы подходят для медной руды?")

if prompt:
    st.markdown("---")
    with st.spinner("Поиск по графу знаний и генерация ответа..."):
        subgraph, found_entities = extract_subgraph(G, prompt, hops=1)
        
        if not subgraph:
            st.warning("К сожалению, я не смог распознать известные мне сущности (материалы, процессы, оборудование) в вашем запросе. Попробуйте перефразировать.")
        else:
            st.info(f"Распознаны сущности (Smart Match): {', '.join(found_entities)}")
            
            # Формируем контекст с ИСТОЧНИКАМИ
            context_lines = []
            unique_sources = set()
            
            for idx, (u, v, data) in enumerate(subgraph.edges(data=True)):
                if idx >= 50:
                    break
                rel = data.get("label", "связан с")
                doc = data.get("doc", "База знаний R&D")
                
                if doc != "unknown":
                    unique_sources.add(doc)
                    
                context_lines.append(f"Факт: [{u}] -> [{rel}] -> [{v}] (Источник: {doc})")
                
            context_text = "\n".join(context_lines)
            
            # Запрос к LLM
            response = ask_groq_llm(context_text, prompt)
            
            if not response:
                response = "⚠️ Получен пустой ответ от LLM. Вот найденный контекст из графа:"
                
            # 1. Текстовый ответ с цитатами
            st.markdown("### Ответ Ассистента")
            st.markdown(response)
            
            # 2. Вывод списка уникальных источников (Expander)
            if unique_sources:
                with st.expander("📚 Посмотреть список использованных источников"):
                    for src in sorted(unique_sources):
                        st.markdown(f"- {src}")
            
            # 3. Граф
            st.markdown("---")
            st.markdown("#### Подтверждающий граф (найденный контекст)")
            render_pyvis_graph(subgraph)
