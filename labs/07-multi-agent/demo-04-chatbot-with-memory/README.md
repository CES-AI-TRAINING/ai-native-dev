# Chatbot with Persistent Memory

Multi-turn chatbot that remembers previous messages using LangGraph's
`MemorySaver` checkpointer. Multiple independent sessions (`thread_id`)
are fully isolated.

## What you will learn

| Concept | Where |
|---------|-------|
| `MemorySaver` checkpointer | `memory = MemorySaver()` |
| `thread_id` session isolation | `config = {"configurable": {"thread_id": ...}}` |
| Multi-turn state | `add_messages` reducer |
| `graph.get_state()` | State inspection section |
| `graph.get_state_history()` | Checkpoint history section |

## Setup

```bash
cd demo-04-chatbot-with-memory
cp .env.example .env
uv sync
uv run python main.py
```
