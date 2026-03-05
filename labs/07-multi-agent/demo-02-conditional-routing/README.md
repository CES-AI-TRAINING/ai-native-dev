# Conditional Routing in LangGraph

Route graph execution to different nodes at runtime based on state values.

## What you will learn

| Concept | Where |
|---------|-------|
| `add_conditional_edges` | Graph builder section |
| Routing function signature | `route_by_sentiment()` |
| Branch-and-merge pattern | positive/negative/neutral → format |

## Graph shape

```
              ┌─ positive ─┐
START → classify ─┤ negative  ├─ format → END
              └─ neutral  ─┘
```

## Setup

```bash
cd demo-02-conditional-routing
cp .env.example .env
uv sync
uv run python main.py
```
