"""Ingesta: informes .docx → chunks con metadatos (data/corpus.jsonl).

Estrategia de chunking ESTRUCTURAL (no por número de caracteres):
- Cada fila de la tabla de cronología = 1 chunk (una publicación es la unidad natural de respuesta).
- Cada ítem de "Lo más interesante" = 1 chunk (opinión/curaduría del informe).
- Encabezado + nota metodológica de cada informe = 1 chunk de contexto.
Cortar por N caracteres partiría publicaciones a la mitad y mezclaría fechas, lo que empeora la recuperación.

Uso: python -m src.rag.ingest carpeta_con_informes
"""
import json
import re
import sys
from pathlib import Path

import docx

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus.jsonl"
MONTHS = {"ago": ("08", "agosto"), "sep": ("09", "septiembre"), "oct": ("10", "octubre")}
REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


def _links(row, document):
    for h in row._tr.xpath(".//w:hyperlink"):
        rid = h.get(REL_ID)
        if rid in document.part.rels:
            return document.part.rels[rid].target_ref
    return ""


def chunks_from_report(path: Path) -> list[dict]:
    d = docx.Document(path)
    report = path.stem.rsplit("_", 1)[-1]
    paras = [p.text.strip() for p in d.paragraphs if p.text.strip()]
    out = []

    i_hl = next(i for i, p in enumerate(paras) if p.lower().startswith("lo más interesante"))
    i_cr = next(i for i, p in enumerate(paras) if p.lower().startswith("cronología"))
    context = "\n".join(paras[:i_cr] + paras[i_cr + 1:i_hl])
    out.append({"id": f"contexto-{report}", "type": "contexto", "report": report, "text": context})

    for j, p in enumerate(paras[i_hl + 1:], 1):
        m = re.search(r"\((\d{1,2}) (ago|sep|oct)\)", p)
        date = f"2026-{MONTHS[m.group(2)][0]}-{int(m.group(1)):02d}" if m else ""
        out.append({"id": f"destacado-{report}-{j}", "type": "destacado", "report": report, "date": date,
                    "text": f"[Lo más destacado de {report}] {p}"})

    for t in d.tables:
        for k, r in enumerate(t.rows[1:], 1):
            fecha, titulo, resumen = (c.text.strip() for c in r.cells[:3])
            day, mon = fecha.split()
            mm, month_name = MONTHS[mon]
            date = f"2026-{mm}-{int(day):02d}"
            out.append({"id": f"post-{date}-{k}", "type": "publicacion", "report": report, "date": date,
                        "title": titulo, "link": _links(r, d),
                        "text": f"{int(day)} de {month_name} de 2026 — {titulo}. {resumen}"})
    return out


def main(folder: Path):
    chunks = [c for p in sorted(folder.glob("informe_my_data_diary_*.docx")) for c in chunks_from_report(p)]
    CORPUS.parent.mkdir(exist_ok=True)
    with CORPUS.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(chunks)} chunks → {CORPUS}")


def load_corpus(path: Path = CORPUS) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main(Path(sys.argv[1]))
