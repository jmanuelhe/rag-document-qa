"""Generación de respuestas con Claude, usando SOLO el contexto recuperado.

Sin ANTHROPIC_API_KEY el sistema funciona en modo extractivo (devuelve los fragmentos con su fuente),
así cualquiera puede probar la recuperación sin pagar una API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

SYSTEM = """Eres un asistente que responde preguntas sobre los informes mensuales "My Data Diary" (ago–oct 2026).
Reglas:
1. Responde SOLO con información de los fragmentos de <contexto>. No uses conocimiento propio.
2. Cita cada afirmación con el id del fragmento entre corchetes, por ejemplo [post-2026-08-31-31].
3. Si el contexto no contiene la respuesta, responde exactamente: "No encuentro esa información en los informes."
4. Responde en español, de forma breve. Los títulos de las publicaciones van en su idioma original."""

ABSTAIN = "No encuentro esa información en los informes."
DEFAULT_MODEL = os.getenv("RAG_MODEL", "claude-sonnet-5-5")


@dataclass
class Answer:
    text: str
    sources: list[str]
    mode: str
    input_tokens: int = 0
    output_tokens: int = 0


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        link = f"\nlink: {c['link']}" if c.get("link") else ""
        parts.append(f'<fragmento id="{c["id"]}">\n{c["text"]}{link}\n</fragmento>')
    return "<contexto>\n" + "\n".join(parts) + "\n</contexto>"


def answer(question: str, chunks: list[dict], client=None, model: str = DEFAULT_MODEL) -> Answer:
    if client is None and not os.getenv("ANTHROPIC_API_KEY"):
        lines = [f"- [{c['id']}] {c['text'][:220]}" for c in chunks[:3]]
        return Answer("Modo extractivo (sin API key). Fragmentos más relevantes:\n" + "\n".join(lines),
                      [c["id"] for c in chunks[:3]], "extractivo")
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model, max_tokens=500, temperature=0, system=SYSTEM,
        messages=[{"role": "user", "content": f"{build_context(chunks)}\n\nPregunta: {question}"}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    cited = [c["id"] for c in chunks if f"[{c['id']}]" in text]
    return Answer(text, cited, "llm", msg.usage.input_tokens, msg.usage.output_tokens)


def citations_are_valid(ans: Answer, chunks: list[dict]) -> bool:
    """Toda cita debe apuntar a un fragmento que de verdad se le entregó al modelo."""
    import re
    ids = {c["id"] for c in chunks}
    cited = set(re.findall(r"\[([a-z]+-[\w-]+)\]", ans.text))
    return cited <= ids
