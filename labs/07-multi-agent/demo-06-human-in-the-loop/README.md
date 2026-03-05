# Human-in-the-Loop Approval Workflow

Pause a LangGraph graph mid-execution and resume it after a human reviews
and approves, rejects, or modifies the AI-drafted plan.

## What you will learn

| Concept | Where |
|---------|-------|
| `interrupt()` — pause and expose value | `human_review_node` |
| `Command(resume=...)` — inject human input | Resume section |
| `MemorySaver` — required for interrupt | `memory = MemorySaver()` |
| `graph.get_state()` — inspect `.next` | After first invoke |

## Workflow

```
START → draft_plan → [INTERRUPT] → execute_plan → END
                           ↑
                   human: approve / reject / revise
```

## Three scenarios

| Scenario | Human Decision |
|----------|---------------|
| 1 | approved — plan runs as-is |
| 2 | rejected — task is aborted |
| 3 | custom instruction — plan is revised |

## Setup

```bash
cd demo-06-human-in-the-loop
cp .env.example .env
uv sync
uv run python main.py
```
