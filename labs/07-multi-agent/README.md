# 07-multi-agent — LangGraph Demos (Basics to Advanced)

A progressive series of 10 hands-on demos covering LangGraph from first
principles to a production-quality multi-agent pipeline. Each demo is a
self-contained Python project managed with [uv](https://docs.astral.sh/uv/).

## Quick Start (any demo)

```bash
cd demo-XX-<name>
cp .env.example .env      # add your OpenAI key
uv sync                   # install dependencies
uv run python main.py     # run the demo
```

## Demo Catalogue

| #   | Demo                                                                    | Concepts                                                         |
| --- | ----------------------------------------------------------------------- | ---------------------------------------------------------------- |
| 01  | [langgraph-hello-world](demo-01-langgraph-hello-world/)                 | `StateGraph`, nodes, edges, `START`/`END`, `compile`, `invoke`   |
| 02  | [conditional-routing](demo-02-conditional-routing/)                     | `add_conditional_edges`, routing functions, branch-and-merge     |
| 03  | [state-schema](demo-03-state-schema/)                                   | `TypedDict`, `Annotated`, `operator.add`, `add_messages`         |
| 04  | [chatbot-with-memory](demo-04-chatbot-with-memory/)                     | `MemorySaver`, `thread_id`, `get_state`, `get_state_history`     |
| 05  | [tool-calling-agent](demo-05-tool-calling-agent/)                       | `@tool`, `bind_tools`, `ToolNode`, `tools_condition`, ReAct loop |
| 06  | [human-in-the-loop](demo-06-human-in-the-loop/)                         | `interrupt`, `Command(resume=...)`, approval workflows           |
| 07  | [multi-agent-supervisor](demo-07-multi-agent-supervisor/)               | `create_react_agent`, supervisor routing, agent orchestration    |
| 08  | [parallel-execution](demo-08-parallel-execution/)                       | Fan-out/fan-in, `operator.add` reducer, parallel nodes           |
| 09  | [rag-graph](demo-09-rag-graph/)                                         | Corrective RAG, cycles, grading, hallucination check             |
| 10  | [advanced-multi-agent-pipeline](demo-10-advanced-multi-agent-pipeline/) | All concepts combined — capstone                                 |

## Learning Path

```
01 → 02 → 03    Graph fundamentals
         ↓
04 → 05 → 06    Agents & interaction
         ↓
07 → 08 → 09    Advanced patterns
         ↓
         10     Complete production system
```

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `pip install uv`
- OpenAI API key (or Google Gemini via OpenAI-compatible endpoint)

## Environment Variables

All demos require at minimum:

```dotenv
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o-mini
```

Demo 09 additionally requires `langchain-community` and `chromadb`
(already declared in its `pyproject.toml`).
