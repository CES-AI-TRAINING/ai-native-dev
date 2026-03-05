# LangGraph Hello World

The simplest possible LangGraph program — a three-node pipeline that preprocesses
user input, calls an LLM, then post-processes the response.

## What you will learn

| Concept | Where |
|---------|-------|
| `StateGraph` and `TypedDict` state | `main.py` lines 40-46 |
| Writing node functions | `preprocess_node`, `llm_node`, `postprocess_node` |
| `add_node` / `add_edge` | Graph builder section |
| `START` and `END` sentinels | Edge wiring |
| `graph.compile()` and `graph.invoke()` | Run section |

## Graph shape

```
START → preprocess → call_llm → postprocess → END
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

## Setup

```bash
cd demo-01-langgraph-hello-world
cp .env.example .env          # then fill in your API key
uv sync
uv run python main.py
```

## Expected output

```
DEMO 01 — LangGraph Hello World
[preprocess_node] ...
[llm_node] ...
[postprocess_node] ...
🤖 Answer:
LangGraph is a library for building stateful, multi-actor applications ...
```
