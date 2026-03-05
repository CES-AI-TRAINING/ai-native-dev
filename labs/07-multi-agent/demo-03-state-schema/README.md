# State Schema in LangGraph

Understand how LangGraph merges node return values into graph state — plain
replacement vs. reducer-based accumulation.

## What you will learn

| Concept                                             | Where  |
| --------------------------------------------------- | ------ |
| Plain `TypedDict` — replacement semantics           | Part 1 |
| `Annotated[list, operator.add]` — list accumulation | Part 2 |
| `Annotated[list, add_messages]` — chat history      | Part 3 |

## Setup

```bash
cd demo-03-state-schema
cp .env.example .env
uv sync
uv run python main.py
```
