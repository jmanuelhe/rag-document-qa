"""CLI: python -m src.rag.ask "¿Qué publicó sobre MCP en agosto?" [--method hybrid+filtro] [--k 5]"""
import argparse

from .generate import answer
from .ingest import load_corpus
from .retrieve import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--method", default="hybrid+filtro", choices=Retriever.METHODS)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()
    hits = Retriever(load_corpus()).search(a.question, k=a.k, method=a.method)
    res = answer(a.question, [c for c, _ in hits])
    print(res.text)
    print(f"\n[modo: {res.mode} · fuentes: {', '.join(res.sources) or '—'}]")


if __name__ == "__main__":
    main()
