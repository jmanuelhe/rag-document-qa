from types import SimpleNamespace

import pytest

from src.rag.generate import ABSTAIN, answer, build_context, citations_are_valid
from src.rag.ingest import load_corpus
from src.rag.retrieve import BM25, Retriever, detect_date, detect_month, rrf
from src.rag.text import tokenize


@pytest.fixture(scope="module")
def r():
    return Retriever(load_corpus())


def test_corpus_structure():
    c = load_corpus()
    assert sum(x["type"] == "publicacion" for x in c) == 82
    assert len({x["id"] for x in c}) == len(c)


def test_tokenize_strips_accents_and_stopwords():
    assert tokenize("¿Qué publicó sobre Agentes?") == ["publico", "agentes"]


def test_bm25_ranks_exact_match_first():
    bm = BM25(["gatos y perros", "agentes de ia con mcp", "recetas de cocina"])
    assert bm.scores("mcp").argmax() == 1


def test_rrf_rewards_agreement():
    import numpy as np
    fused = rrf([np.array([0, 1, 2]), np.array([1, 0, 2])])
    assert fused[2] < fused[0] and fused[2] < fused[1]


def test_date_and_month_detection():
    assert detect_date("¿Qué publicó el 6 de octubre?") == "2026-10-06"
    assert detect_month("pasantías de septiembre") == "septiembre"
    assert detect_month("agosto vs octubre") is None  # ambiguo → sin filtro


def test_date_filter_retrieves_that_day(r):
    top = r.search("¿Qué publicó el 6 de octubre?", k=1)[0][0]
    assert top["date"] == "2026-10-06"


def test_extractive_mode_without_key(monkeypatch, r):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    hits = [c for c, _ in r.search("MCP agosto", k=3)]
    res = answer("MCP agosto", hits)
    assert res.mode == "extractivo" and res.sources


class FakeClient:
    """Imita la interfaz de anthropic.Anthropic para probar sin red ni costo."""
    def __init__(self, text):
        self.messages = SimpleNamespace(create=lambda **kw: SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            usage=SimpleNamespace(input_tokens=500, output_tokens=40)))


def test_llm_answer_with_valid_citation(r):
    hits = [c for c, _ in r.search("Add an LLM to Your MCP Server", k=3)]
    res = answer("q", hits, client=FakeClient(f"Publicó una guía [{hits[0]['id']}]."))
    assert res.mode == "llm" and res.sources == [hits[0]["id"]] and citations_are_valid(res, hits)


def test_hallucinated_citation_is_detected(r):
    hits = [c for c, _ in r.search("MCP", k=2)]
    res = answer("q", hits, client=FakeClient("Algo [post-2026-12-01-1]."))
    assert not citations_are_valid(res, hits)


def test_context_contains_ids():
    ctx = build_context([{"id": "x-1", "text": "hola"}])
    assert 'id="x-1"' in ctx and ABSTAIN.startswith("No encuentro")
