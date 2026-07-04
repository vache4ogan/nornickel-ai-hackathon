import streamlit as st
import json
import os
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import difflib
import re

# Используем официальную библиотеку Groq
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

# Жесткий фильтр стоп-слов для русского языка в контексте металлургии
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
                
                # Добавляем узлы
                for ent in chunk.get("entities", []):
                    G.add_node(ent["id"], type=ent.get("type", "Unknown"))
                    
                # Добавляем связи
                for rel in chunk.get("relations", []):
                    # Используем ключи 'source' и 'target'
                    G.add_edge(
                        rel["source"], 
                        rel["target"], 
                        label=rel["type"],
                        properties=rel.get("properties", {})
                    )
            except json.JSONDecodeError:
                pass
    return G

def ask_groq_llm(context_text: str, question: str) -> str:
    system_prompt = (
        "Ты — AI-ассистент в горно-металлургической отрасли. "
        "Твоя задача — отвечать на вопросы пользователя на русском языке, используя ТОЛЬКО предоставленный контекст из Графа Знаний. "
        "Не придумывай информацию. Если в контексте нет ответа, честно скажи: 'В моем графе знаний нет информации об этом'. "
        "Отвечай четко, структурированно, ссылаясь на связи между сущностями."
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
    # Разделяем вопрос на слова, игнорируем пунктуацию
    words = re.findall(r'\b\w+\b', query.lower())
    
    # 3. Умный фильтр: удаляем стоп-слова и слова короче 3 символов
    meaningful_words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
    
    found_nodes = set()
    all_graph_nodes = list(G.nodes())
    
    # 4. Fuzzy Matching (cutoff=0.6) только для оставшихся смысловых слов
    for word in meaningful_words:
        matches = difflib.get_close_matches(word, [n.lower() for n in all_graph_nodes], n=5, cutoff=0.6)
        
        # Если слово совпало, находим оригинальные узлы (с правильным регистром)
        for match in matches:
            for node in all_graph_nodes:
                if node.lower() == match:
                    found_nodes.add(node)
                    
    # Также проверяем точные совпадения подстроки для надежности
    for node in all_graph_nodes:
        if len(node) > 2 and node.lower() in query.lower():
            # Дополнительная проверка, чтобы не находить предлоги внутри слов
            if bool(re.search(r'\b' + re.escape(node.lower()) + r'\b', query.lower())):
                found_nodes.add(node)
            
    found_nodes_list = list(found_nodes)
    if not found_nodes_list:
        return None, []
        
    # Поиск подграфа (ego-graph)
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
        net.add_edge(u, v, title=label, label=label, color="#7f8c8d")
        
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

# Поле ввода вопроса (обычный text_input, не chat)
prompt = st.text_input("Ваш вопрос:", placeholder="Например: Какие методы подходят для медной руды?")

if prompt:
    st.markdown("---")
    with st.spinner("Поиск по графу знаний и генерация ответа..."):
        subgraph, found_entities = extract_subgraph(G, prompt, hops=1)
        
        if not subgraph:
            st.warning("К сожалению, я не смог распознать известные мне сущности (материалы, процессы, оборудование) в вашем запросе. Попробуйте перефразировать.")
        else:
            st.info(f"Распознаны сущности (Smart Match): {', '.join(found_entities)}")
            
            # Формируем текстовый контекст из подграфа (Ограничение до 50 связей)
            context_lines = []
            for idx, (u, v, data) in enumerate(subgraph.edges(data=True)):
                if idx >= 50: # Лимит топ-50
                    break
                rel = data.get("label", "связан с")
                context_lines.append(f"[{u}] -> [{rel}] -> [{v}]")
            context_text = "\n".join(context_lines)
            
            # Запрос к LLM Groq SDK
            response = ask_groq_llm(context_text, prompt)
            
            # Если ответ пустой (на всякий случай)
            if not response:
                response = "⚠️ Получен пустой ответ от LLM. Вот найденный контекст из графа:"
                
            # 1. Выводим текст крупно
            st.markdown("### Ответ Ассистента")
            st.markdown(response)
            
            # 2. Выводим граф как пруф под текстом
            st.markdown("---")
            st.markdown("#### Подтверждающий граф (найденный контекст)")
            render_pyvis_graph(subgraph)
