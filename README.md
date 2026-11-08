# RAG Document Q&A · Pregúntale a tus informes

**Sistema de preguntas y respuestas sobre mis tres informes mensuales “My Data Diary”, con búsqueda híbrida, filtros por fecha, citas verificables y evaluación cuantitativa.**
Proyecto de portafolio · Ingeniero de IA · Python, scikit-learn, Claude API, Streamlit

Inspirado en *“Building a Document Q&A System”* (26 ago) y *“End-to-End RAG Projects for Your AI Portfolio”* (2 oct), de My Data Diary. Los datos son los mismos informes analizados en [my-data-diary-analytics](https://github.com/jmanuelhe/my-data-diary-analytics).

---

## ¿Qué es RAG y por qué?

Un LLM no conoce mis informes, y si le pregunto por ellos **inventa**. RAG (*Retrieval-Augmented Generation*) resuelve esto en dos pasos:

1. **Recuperar** los fragmentos de los documentos que responden la pregunta.
2. **Generar** la respuesta pidiéndole al LLM que use **solo** esos fragmentos y que cite de dónde sacó cada dato.

```
pregunta ─► filtros (fecha/mes) ─► BM25 ─┐
                                 TF-IDF ─┴► fusión RRF ─► top-k fragmentos ─► Claude + reglas ─► respuesta con [citas]
                                                                │
                                                                └─► sin API key: modo extractivo
```

## Decisiones de diseño

| Componente | Elección | Por qué |
|---|---|---|
| **Chunking** | Estructural: 1 publicación = 1 fragmento; 1 destacado = 1 fragmento; encabezado + metodología = 1 fragmento por informe | Cortar cada N caracteres partiría publicaciones y mezclaría fechas. Los 3 informes dan **105 fragmentos**. |
| **BM25** | Implementado desde cero (`retrieve.py`) | Es el estándar de los buscadores. Hacerlo a mano demuestra que entiendo TF, IDF y la normalización por longitud. |
| **TF-IDF de n-gramas de caracteres (3–5)** | scikit-learn | Tolera *agente/agentes/agents* y la mezcla español-inglés del corpus, sin descargar modelos. |
| **Fusión** | Reciprocal Rank Fusion (k=60) | Combina rankings sin normalizar puntajes de escalas distintas. |
| **Filtros por metadatos** | “el 6 de octubre” → solo ese día; “en septiembre” → solo ese mes | Las preguntas por fecha fallaban porque el texto de metodología menciona muchos días. |
| **Generación** | Claude, `temperature=0`, instrucciones de citar y de abstenerse | Respuestas reproducibles y verificables. |
| **Verificación de citas** | `citations_are_valid()` | Detecta si el modelo cita un fragmento que no se le entregó (alucinación de fuente). |
| **Modo extractivo** | Sin API key devuelve los fragmentos | Cualquiera puede probar el sistema gratis. |

## Evaluación de la recuperación

45 preguntas escritas a mano (`eval/questions.jsonl`) en 5 tipos: paráfrasis, semántica (sin palabras en común con el texto), fecha, mes y metodología.

| Método | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25 | 0,844 | 0,911 | 0,933 | 0,875 |
| TF-IDF (caracteres) | 0,889 | 0,933 | 0,933 | 0,904 |
| Híbrido (RRF) | 0,911 | 0,933 | 0,933 | 0,922 |
| **Híbrido + filtros** | **0,933** | **0,956** | **0,956** | **0,944** |

**Hit@1 por tipo de pregunta:**

| Tipo | BM25 | TF-IDF | Híbrido | Híbrido + filtros |
|---|---|---|---|---|
| Paráfrasis | 0,96 | 1,00 | 1,00 | 1,00 |
| Metodología | 0,33 | 0,67 | 1,00 | 1,00 |
| Mes | 1,00 | 1,00 | 1,00 | 1,00 |
| Fecha | 0,67 | 0,33 | 0,67 | **1,00** |
| Semántica | 0,70 | 0,80 | 0,70 | 0,70 |

**Lo que muestran los números:**
- El híbrido supera a cada método por separado: los dos se equivocan en preguntas distintas.
- Los filtros por fecha corrigieron todas las preguntas de tipo fecha.
- **El punto débil es la búsqueda semántica** (“¿hubo contenido sobre *reconocimiento de imágenes*?” cuando el texto dice *vision AI*). La búsqueda léxica no sabe que son sinónimos. El siguiente paso natural es agregar **embeddings**.

### Un experimento que no funcionó (y por qué importa)

Probé decidir “no hay respuesta” con un umbral sobre el puntaje de recuperación. **No sirve**: las preguntas sin respuesta (`eval/unanswerable.jsonl`) sacan puntajes BM25 de 3,2–5,0, y varias preguntas con respuesta sacan menos (mínimo 1,6). Las distribuciones se solapan. Por eso la abstención se delega al LLM con una instrucción explícita, y se mide aparte.

## Evaluación de la generación

`python -m src.rag.eval_generation` (requiere `ANTHROPIC_API_KEY`) mide **citas válidas**, **cita correcta**, **abstención correcta** en preguntas sin respuesta y **abstención indebida**, además de tokens y tiempo. Guarda los resultados en `reports/generation_eval.json`. *No se incluyen resultados porque se ejecuta con la clave de cada quien.*

## Uso

```bash
pip install -r requirements.txt
python -m src.rag.ask "¿Qué publicó sobre MCP en agosto?"   # funciona sin API key (modo extractivo)
export ANTHROPIC_API_KEY=...                                 # para respuestas generadas por Claude
streamlit run app.py                                         # interfaz web
python -m src.rag.evaluate                                   # evaluación de recuperación
python -m pytest -q                                          # 10 tests (el LLM se simula con un cliente falso)
```

Para regenerar el corpus a partir de los `.docx`: `python -m src.rag.ingest carpeta_con_informes`.

## Próximos pasos

- Embeddings multilingües y comparación en la misma tabla de evaluación.
- Re-ranking con un *cross-encoder* sobre el top-20.
- Ingesta automática de cada informe nuevo.
