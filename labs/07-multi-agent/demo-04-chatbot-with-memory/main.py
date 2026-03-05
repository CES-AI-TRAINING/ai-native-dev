"""
Demo 04: Chatbot with Persistent Memory (MemorySaver)

Without a checkpointer each graph.invoke() starts with an empty state.
Adding a *checkpointer* lets LangGraph persist and reload the full state
between calls — forming the foundation of multi-turn conversations.

Topics covered:
1. MemorySaver — in-process checkpointer (no external DB needed)
2. thread_id — isolates memory per conversation session
3. MessagesState — built-in state with Annotated add_messages field
4. Inspecting saved state with graph.get_state()
5. Listing all checkpoints with graph.get_state_history()

Graph shape:
    START → chatbot → END   (plus checkpointer wired into compile)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from langchain_core.messages import BaseMessage
from typing import TypedDict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set.")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    temperature=0.7,
)

SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. "
    "Remember what the user has told you in this conversation and refer back to it naturally."
)

# ---------------------------------------------------------------------------
# State — using the built-in MessagesState pattern
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Chatbot node
# ---------------------------------------------------------------------------

def chatbot_node(state: State) -> dict:
    """Call the LLM with the full message history (including system prompt)."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Build graph WITH checkpointer
# ---------------------------------------------------------------------------

memory = MemorySaver()          # in-memory checkpointer — replace with SqliteSaver for persistence

builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Passing checkpointer=memory makes every invoke() save & restore state
graph = builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Helper: send a message in a specific thread
# ---------------------------------------------------------------------------

def chat(thread_id: str, user_message: str) -> str:
    """Send a user message and return the assistant reply."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Demo: two independent sessions
# ---------------------------------------------------------------------------

print("=" * 60)
print("DEMO 04 — Chatbot with Persistent Memory")
print("=" * 60)
print()

# ── Session Alice ────────────────────────────────────────────
print("── Session: alice ──────────────────────────────────────")
reply = chat("alice", "Hi! My name is Alice and I love hiking.")
print(f"[alice] User : Hi! My name is Alice and I love hiking.")
print(f"[alice] Bot  : {reply}")
print()

reply = chat("alice", "What hobby did I just mention?")
print(f"[alice] User : What hobby did I just mention?")
print(f"[alice] Bot  : {reply}")
print()

reply = chat("alice", "Can you suggest a hiking trail in Yosemite for a beginner?")
print(f"[alice] User : Can you suggest a hiking trail in Yosemite for a beginner?")
print(f"[alice] Bot  : {reply}")
print()

# ── Session Bob — completely isolated ───────────────────────
print("── Session: bob ────────────────────────────────────────")
reply = chat("bob", "Hello! What is my name?")
print(f"[bob] User : Hello! What is my name?")
print(f"[bob] Bot  : {reply}")
print()

# ── Inspect saved state for alice ───────────────────────────
print("── Inspect state: alice ────────────────────────────────")
state_snapshot = graph.get_state({"configurable": {"thread_id": "alice"}})
print(f"Messages in alice's thread: {len(state_snapshot.values['messages'])}")
for m in state_snapshot.values["messages"]:
    role = type(m).__name__
    print(f"  [{role}] {m.content[:80]}{'...' if len(m.content) > 80 else ''}")
print()

# ── List history of checkpoints ─────────────────────────────
print("── alice checkpoint history ────────────────────────────")
history = list(graph.get_state_history({"configurable": {"thread_id": "alice"}}))
print(f"  Number of checkpoints: {len(history)}")
for i, snap in enumerate(history):
    n_msgs = len(snap.values.get("messages", []))
    print(f"  Checkpoint {i+1}: {n_msgs} messages — next={snap.next}")
