"""Recuperación: BM25 (implementado desde cero), TF-IDF y fusión híbrida (RRF).

Por qué tres métodos:
- BM25: estándar léxico de los buscadores. Premia términos raros y penaliza documentos largos.
- TF-IDF con n-gramas de caracteres: tolera variaciones morfológicas (agente/agentes/agents)
  y la mezcla español-inglés del corpus, sin descargar modelos.
- Híbrido con Reciprocal Rank Fusion: combina rankings sin tener que normalizar puntajes
  de escalas distintas. score(d) = Σ 1 / (k + rank_i(d)).
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .text import strip_accents, tokenize

MONTH_WORDS = {"agosto": "agosto", "august": "agosto", "septiembre": "septiembre", "september": "septiembre",
               "octubre": "octubre", "october": "octubre"}


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.avgdl = sum(map(len, self.toks)) / len(self.toks)
        self.tf = [Counter(t) for t in self.toks]
        df = Counter(term for t in self.toks for term in set(t))
        n = len(docs)
        # IDF de BM25 con +1 para que nunca sea negativo
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        out = np.zeros(len(self.toks))
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out


class TfidfRetriever:
    def __init__(self, docs: list[str]):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=lambda s: strip_accents(s.lower()),
                                   sublinear_tf=True, min_df=1)
        self.m = self.vec.fit_transform(docs)

    def scores(self, query: str) -> np.ndarray:
        return linear_kernel(self.vec.transform([query]), self.m).ravel()


def rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
    """rankings: arreglos de índices ordenados de mejor a peor."""
    n = len(rankings[0])
    fused = np.zeros(n)
    for r in rankings:
        for rank, idx in enumerate(r):
            fused[idx] += 1.0 / (k + rank + 1)
    return fused


MONTH_NUM = {"agosto": "08", "septiembre": "09", "octubre": "10"}


def detect_date(query: str) -> str | None:
    """'el 6 de octubre' → '2026-10-06'."""
    m = re.search(r"\b(\d{1,2}) de (agosto|septiembre|octubre)\b", strip_accents(query.lower()))
    return f"2026-{MONTH_NUM[m.group(2)]}-{int(m.group(1)):02d}" if m else None


def detect_month(query: str) -> str | None:
    q = strip_accents(query.lower())
    found = {v for k, v in MONTH_WORDS.items() if re.search(rf"\b{k}\b", q)}
    return found.pop() if len(found) == 1 else None


class Retriever:
    METHODS = ("bm25", "tfidf", "hybrid", "hybrid+filtro")

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["text"] + " " + c.get("title", "") for c in chunks]
        self.bm25, self.tfidf = BM25(texts), TfidfRetriever(texts)

    def search(self, query: str, k: int = 5, method: str = "hybrid+filtro") -> list[tuple[dict, float]]:
        if method == "bm25":
            s = self.bm25.scores(query)
        elif method == "tfidf":
            s = self.tfidf.scores(query)
        else:
            s = rrf([np.argsort(-self.bm25.scores(query)), np.argsort(-self.tfidf.scores(query))])
            if method == "hybrid+filtro":
                # Filtros por metadatos: una fecha exacta pesa más que un mes.
                if date := detect_date(query):
                    mask = np.array([c.get("date") == date for c in self.chunks])
                elif month := detect_month(query):
                    mask = np.array([c["report"] == month or c.get("date", "")[5:7] == MONTH_NUM[month]
                                     for c in self.chunks])
                else:
                    mask = None
                if mask is not None and mask.any():
                    s = np.where(mask, s, -1)
        order = np.argsort(-s)[:k]
        return [(self.chunks[i], float(s[i])) for i in order]
