"""Evaluación de la GENERACIÓN (requiere ANTHROPIC_API_KEY).

Mide, sobre el conjunto de evaluación:
- citas_validas: % de respuestas cuyas citas apuntan solo a fragmentos entregados (no inventa fuentes).
- cita_correcta: % de respuestas que citan al menos un fragmento correcto.
- abstencion_correcta: % de preguntas SIN respuesta en los informes donde el modelo se abstiene.
- abstencion_indebida: % de preguntas CON respuesta donde el modelo se abstuvo sin necesidad.
"""
import json
import time

from .evaluate import EVAL, is_hit
from .generate import ABSTAIN, answer, citations_are_valid
from .ingest import ROOT, load_corpus
from .retrieve import Retriever


def main():
    corpus = load_corpus(); by_id = {c["id"]: c for c in corpus}
    r = Retriever(corpus)
    qs = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    un = [json.loads(l) for l in (ROOT / "eval/unanswerable.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    valid = correct = over_abstain = 0
    tokens_in = tokens_out = 0
    t0 = time.perf_counter()
    for q in qs:
        hits = [c for c, _ in r.search(q["q"], k=5)]
        a = answer(q["q"], hits)
        tokens_in += a.input_tokens; tokens_out += a.output_tokens
        valid += citations_are_valid(a, hits)
        titles = {by_id[g].get("title", "") for g in q["gold"] if by_id[g].get("title")}
        correct += any(is_hit(by_id[s], set(q["gold"]), titles) for s in a.sources)
        over_abstain += ABSTAIN in a.text
    abstain_ok = 0
    for q in un:
        a = answer(q["q"], [c for c, _ in r.search(q["q"], k=5)])
        tokens_in += a.input_tokens; tokens_out += a.output_tokens
        abstain_ok += ABSTAIN in a.text
    res = {"n": len(qs), "citas_validas": valid / len(qs), "cita_correcta": correct / len(qs),
           "abstencion_indebida": over_abstain / len(qs), "abstencion_correcta": abstain_ok / len(un),
           "tokens_entrada": tokens_in, "tokens_salida": tokens_out,
           "segundos": round(time.perf_counter() - t0, 1)}
    (ROOT / "reports/generation_eval.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
