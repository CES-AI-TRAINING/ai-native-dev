# Corrective RAG as a LangGraph

Implement the Corrective RAG (CRAG) pattern as a cyclic LangGraph. Unlike
a linear chain, this graph can loop — retrying generation if hallucination
is detected and falling back to web search when retrieval is insufficient.

## What you will learn

| Concept | Where |
|---------|-------|
| RAG pipeline as a graph | All nodes |
| Grading retrieved documents | `grade_documents_node` |
| Web search fallback | `web_fallback_node` |
| Cyclic graph (retry loop) | `hallucination_check → generate` |
| Hallucination detection | `hallucination_check_node` |

## Graph shape

```
                    ┌─────────────────────────────────────────────┐
START → retrieve → grade_docs                                      │
              irrelevant ─► web_fallback ─┐                       │
              relevant ──────────────────►┴─► generate → hallucination_check
                                                              │
                                                    grounded ─► END
                                                    hallucinating ─► generate (retry)
```

## Setup

```bash
cd demo-09-rag-graph
cp .env.example .env
uv sync
uv run python main.py
```
