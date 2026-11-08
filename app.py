"""Interfaz web mínima: streamlit run app.py"""
import streamlit as st

from src.rag.generate import answer
from src.rag.ingest import load_corpus
from src.rag.retrieve import Retriever

st.set_page_config(page_title="Pregúntale a My Data Diary", page_icon="📚")
st.title("Pregúntale a My Data Diary")
st.caption("RAG sobre 3 informes mensuales (ago–oct 2026) · BM25 + TF-IDF + filtros por fecha")


@st.cache_resource
def retriever():
    return Retriever(load_corpus())


q = st.text_input("Pregunta", placeholder="¿Qué publicó sobre MCP en agosto?")
method = st.sidebar.selectbox("Recuperación", Retriever.METHODS, index=3)
k = st.sidebar.slider("Fragmentos (k)", 1, 10, 5)
if q:
    hits = retriever().search(q, k=k, method=method)
    res = answer(q, [c for c, _ in hits])
    st.markdown(res.text)
    with st.expander("Fragmentos recuperados"):
        for c, s in hits:
            st.markdown(f"**{c['id']}** · score {s:.3f}\n\n{c['text']}")
