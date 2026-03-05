# Tool-Calling Agent (ReAct Loop)

A LangGraph agent that autonomously decides when to call tools and loops
until it has a final answer — implementing the classic ReAct pattern.

## What you will learn

| Concept | Where |
|---------|-------|
| `@tool` decorator | `calculator`, `get_weather`, `word_count` |
| `llm.bind_tools()` | LLM setup section |
| `ToolNode` | `tool_node = ToolNode(tools)` |
| `tools_condition` | `add_conditional_edges` call |
| Agent → Tools → Agent loop | Graph wiring |

## Graph shape

```
           ┌──────────────────────────┐
           │                          ▼
START → agent ─(tools_condition)─► tools
           │(END)
           ▼
          END
```

## Tools included

- `calculator` — evaluate math expressions
- `get_weather` — mock weather lookup
- `word_count` — count words and characters

## Setup

```bash
cd demo-05-tool-calling-agent
cp .env.example .env
uv sync
uv run python main.py
```
