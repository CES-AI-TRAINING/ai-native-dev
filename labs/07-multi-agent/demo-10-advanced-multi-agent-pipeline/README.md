# Advanced Multi-Agent Content Pipeline — Capstone Demo

This demo brings together **every concept** from demos 01–09 into a single
production-quality content creation pipeline.

## Pipeline stages

```
START → planner ──► [research_1 ‖ research_2 ‖ research_3] ──► writer
      → editor ──(quality < 7? loop back to writer)──► human_review [⏸ INTERRUPT]
      → (approve/reject/revise) ──► finalise → END
```

## Concepts demonstrated

| Demo | Concept | Where in pipeline |
|------|---------|------------------|
| 01 | StateGraph, nodes, edges | Entire pipeline |
| 02 | Conditional routing | editor → writer/human_review |
| 03 | State reducers | `research_notes: operator.add` |
| 04 | MemorySaver, thread_id | `graph.compile(checkpointer=memory)` |
| 05 | Tool-calling | `search_knowledge_base`, `check_readability` |
| 06 | Human-in-the-loop | `human_review_node` with `interrupt` |
| 07 | Multi-agent orchestration | planner → workers → writer → editor |
| 08 | Parallel execution (fan-out/fan-in) | 3 parallel research workers |
| 09 | RAG-style grounding | Knowledge base lookup in research |

## Scenarios

| # | Topic | Human Decision |
|---|-------|---------------|
| 1 | Generative AI in Software Dev | approved |
| 2 | Python Best Practices | custom revision |

## Setup

```bash
cd demo-10-advanced-multi-agent-pipeline
cp .env.example .env
uv sync
uv run python main.py
```
