# Parallel Node Execution — Fan-Out / Fan-In

Run four analysis nodes simultaneously, then merge results into a single
report. Demonstrates how a single source edge can dispatch to multiple
nodes in parallel.

## What you will learn

| Concept | Where |
|---------|-------|
| Fan-out: one source → many nodes | `preprocess → 4 analysts` |
| `Annotated[list, operator.add]` for fan-in | `analyses` field |
| Fan-in: many nodes → one sink | `all analysts → synthesise` |
| Parallel vs. sequential performance | Timing output |

## Graph shape

```
                    ┌─► sentiment_analyst ─┐
                    ├─► topic_extractor    ─┤
START → preprocess ─┤                       ├─► synthesise → END
                    ├─► action_detector    ─┤
                    └─► quality_scorer     ─┘
```

## Setup

```bash
cd demo-08-parallel-execution
cp .env.example .env
uv sync
uv run python main.py
```
