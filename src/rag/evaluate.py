"""Evaluación de la recuperación con un conjunto de preguntas etiquetado a mano.

Métricas:
- Hit@k: ¿apareció algún chunk correcto entre los k primeros?
- MRR (Mean Reciprocal Rank): 1/posición del primer chunk correcto, promediado.
Un chunk "destacado" que menciona la publicación correcta también cuenta como acierto,
porque contiene la respuesta.
"""
import json
from pathlib import Path

import pandas as pd

from .ingest import ROOT, load_corpus
from .retrieve import Retriever

EVAL = ROOT / "eval" / "questions.jsonl"


def is_hit(chunk: dict, gold: set, gold_titles: set) -> bool:
    if chunk["id"] in gold:
        return True
    return chunk["type"] == "destacado" and any(t in chunk["text"] for t in gold_titles)


def evaluate(k_values=(1, 3, 5)) -> pd.DataFrame:
    corpus = load_corpus()
    by_id = {c["id"]: c for c in corpus}
    r = Retriever(corpus)
    qs = [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, per_q = [], []
    for method in Retriever.METHODS:
        hits = {k: 0 for k in k_values}
        rr = 0.0
        for q in qs:
            gold = set(q["gold"])
            titles = {by_id[g].get("title", "") for g in gold if by_id[g].get("title")}
            res = r.search(q["q"], k=max(k_values), method=method)
            flags = [is_hit(c, gold, titles) for c, _ in res]
            for k in k_values:
                hits[k] += any(flags[:k])
            rr += next((1 / (i + 1) for i, f in enumerate(flags) if f), 0)
            per_q.append({"method": method, "tipo": q.get("tipo", ""), "q": q["q"], "hit@1": flags[0],
                          "hit@5": any(flags[:5]), "top1": res[0][0]["id"]})
        rows.append({"method": method, **{f"hit@{k}": hits[k] / len(qs) for k in k_values}, "mrr": rr / len(qs)})
    return pd.DataFrame(rows), pd.DataFrame(per_q)


if __name__ == "__main__":
    df, per_q = evaluate()
    df.round(3).to_csv(ROOT / "reports" / "retrieval_eval.csv", index=False)
    per_q.to_csv(ROOT / "reports" / "retrieval_eval_per_question.csv", index=False)
    print(df.round(3).to_string(index=False))
    print()
    print(per_q.pivot_table(index="tipo", columns="method", values="hit@1", aggfunc="mean").round(2))
    print()
    print(per_q[(per_q.method == "hybrid+filtro") & ~per_q["hit@1"]][["tipo", "q", "top1"]].to_string(index=False))
