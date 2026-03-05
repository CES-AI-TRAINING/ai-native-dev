"""
Demo 03: State Schema — TypedDict, Annotated fields & reducers

LangGraph merges every node's return dict into the graph state.
By default it *replaces* a key.  Using Annotated<T, reducer> you can
instead *accumulate* values — the canonical example is message history.

Topics covered:
1. Plain TypedDict state (replacement semantics)
2. Annotated field with a custom reducer (len-tracking counter)
3. Annotated list with operator.add (list accumulation)
4. add_messages reducer for chat history (append-only)
5. Inspecting the full state after each step
"""

import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set.")

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.5)

# ===========================================================================
# PART 1 — Plain replacement state
# ===========================================================================
print("=" * 60)
print("PART 1 — Plain replacement state")
print("=" * 60)

class SimpleState(TypedDict):
    value: int
    message: str

def increment(state: SimpleState) -> dict:
    return {"value": state["value"] + 1, "message": f"incremented to {state['value'] + 1}"}

def double(state: SimpleState) -> dict:
    return {"value": state["value"] * 2, "message": f"doubled to {state['value'] * 2}"}

g1 = StateGraph(SimpleState)
g1.add_node("increment", increment)
g1.add_node("double", double)
g1.add_edge(START, "increment")
g1.add_edge("increment", "double")
g1.add_edge("double", END)
graph1 = g1.compile()

out1 = graph1.invoke({"value": 3, "message": ""})
print(f"  Input value=3 → After increment+double: value={out1['value']}")
print(f"  Final message: '{out1['message']}'")
print()

# ===========================================================================
# PART 2 — Accumulated list with operator.add reducer
# ===========================================================================
print("=" * 60)
print("PART 2 — Annotated list with operator.add reducer")
print("=" * 60)

class LogState(TypedDict):
    steps: Annotated[list[str], operator.add]   # each node *appends*, never replaces
    result: str

def step_a(state: LogState) -> dict:
    return {"steps": ["step_a completed"], "result": "a"}

def step_b(state: LogState) -> dict:
    return {"steps": ["step_b completed"], "result": "b"}

def step_c(state: LogState) -> dict:
    combined = state["result"] + "+c"
    return {"steps": ["step_c completed"], "result": combined}

g2 = StateGraph(LogState)
g2.add_node("a", step_a)
g2.add_node("b", step_b)
g2.add_node("c", step_c)
g2.add_edge(START, "a")
g2.add_edge("a", "b")
g2.add_edge("b", "c")
g2.add_edge("c", END)
graph2 = g2.compile()

out2 = graph2.invoke({"steps": [], "result": ""})
print(f"  Accumulated steps: {out2['steps']}")
print(f"  Final result: '{out2['result']}'")
print()

# ===========================================================================
# PART 3 — add_messages reducer for chat history
# ===========================================================================
print("=" * 60)
print("PART 3 — add_messages reducer (message history)")
print("=" * 60)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn: int

def user_turn(state: ChatState) -> dict:
    """Simulate adding a user message."""
    turn = state["turn"]
    user_msg = HumanMessage(content=f"Turn {turn}: tell me a fun fact about the number {turn}.")
    return {"messages": [user_msg], "turn": turn}

def assistant_turn(state: ChatState) -> dict:
    """Call the LLM and append its response."""
    response: AIMessage = llm.invoke(state["messages"])
    print(f"  [turn {state['turn']}] LLM: {response.content[:100]}...")
    return {"messages": [response]}

g3 = StateGraph(ChatState)
g3.add_node("user",      user_turn)
g3.add_node("assistant", assistant_turn)
g3.add_edge(START,       "user")
g3.add_edge("user",      "assistant")
g3.add_edge("assistant", END)
graph3 = g3.compile()

# Two separate invocations to show messages accumulate across turns
# (without a checkpointer each invoke starts fresh — see demo-04 for persistence)
state: ChatState = {"messages": [], "turn": 1}
state = graph3.invoke(state)  # type: ignore[assignment]
state["turn"] = 2
state = graph3.invoke(state)  # type: ignore[assignment]

print(f"\n  Total messages in history: {len(state['messages'])}")
for m in state["messages"]:
    role = "User" if isinstance(m, HumanMessage) else "AI"
    print(f"  [{role}] {m.content[:80]}{'...' if len(m.content) > 80 else ''}")
print()

print("All parts complete. Key takeaways:")
print("  • Default: node return dict *replaces* state keys")
print("  • Annotated[list, operator.add] → lists are concatenated")
print("  • Annotated[list, add_messages] → messages are appended (dedup by id)")
