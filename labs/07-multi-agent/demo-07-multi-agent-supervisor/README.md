# Multi-Agent Supervisor Pattern

A supervisor LLM orchestrates a team of specialist agents (researcher, coder),
routing tasks to the right expert and synthesising a final answer.

## What you will learn

| Concept | Where |
|---------|-------|
| `create_react_agent` — sub-agents | `researcher_agent`, `coder_agent` |
| Supervisor routing with JSON output | `supervisor_node` |
| Looping: agent → supervisor → agent | Graph edges |
| Termination: "FINISH" → synthesise | `route_supervisor` |

## Graph shape

```
                  ┌──────────────┐
                  │              │
START → supervisor ─► researcher ─┤ (loop back)
                  └──► coder ────┘
                  └──► synthesise → END
```

## Agents

| Agent | Tools |
|-------|-------|
| researcher | web_search, summarise_findings |
| coder | write_code, explain_code |

## Setup

```bash
cd demo-07-multi-agent-supervisor
cp .env.example .env
uv sync
uv run python main.py
```
