"""
Demo 01: LangGraph Hello World

The very first step into LangGraph. This demo builds a minimal StateGraph
to show how nodes, edges, state, and compilation work together.

Topics covered:
1. Defining a TypedDict state schema
2. Writing node functions that read and update state
3. Connecting nodes with directed edges (add_edge)
4. Using START and END sentinels
5. Compiling and invoking the graph
6. Visualising the execution trace from the output state
"""

import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set. Copy .env.example → .env and fill in the key.")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    temperature=0.7,
)

# ---------------------------------------------------------------------------
# Step 1: Define the State
#
# State is a plain TypedDict.  Every node receives a copy of the current
# state and returns a dictionary with the keys it wants to update.
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    input: str          # original user input
    step1_result: str   # output from the first processing node
    step2_result: str   # output from the second processing node
    final_output: str   # assembled final answer


# ---------------------------------------------------------------------------
# Step 2: Define nodes
#
# Each node is an ordinary Python function:
#   - receives the current state (dict)
#   - returns a dict with only the keys that changed
# ---------------------------------------------------------------------------

def preprocess_node(state: GraphState) -> dict:
    """Clean and normalise the raw user input."""
    raw = state["input"]
    cleaned = raw.strip().lower()
    print(f"[preprocess_node] raw='{raw}' → cleaned='{cleaned}'")
    return {"step1_result": cleaned}


def llm_node(state: GraphState) -> dict:
    """Call the LLM with the preprocessed input."""
    question = state["step1_result"]
    print(f"[llm_node] sending to LLM: '{question}'")
    response = llm.invoke([HumanMessage(content=question)])
    answer = response.content
    print(f"[llm_node] LLM responded: '{answer[:80]}…'" if len(answer) > 80 else f"[llm_node] LLM: '{answer}'")
    return {"step2_result": answer}


def postprocess_node(state: GraphState) -> dict:
    """Wrap the LLM answer in a friendly envelope."""
    answer = state["step2_result"]
    final = f"🤖 Answer:\n{answer}"
    print(f"[postprocess_node] assembling final output")
    return {"final_output": final}


# ---------------------------------------------------------------------------
# Step 3: Build the graph
# ---------------------------------------------------------------------------

builder = StateGraph(GraphState)

# Register nodes
builder.add_node("preprocess", preprocess_node)
builder.add_node("call_llm",   llm_node)
builder.add_node("postprocess", postprocess_node)

# Wire edges  START → preprocess → call_llm → postprocess → END
builder.add_edge(START,          "preprocess")
builder.add_edge("preprocess",   "call_llm")
builder.add_edge("call_llm",     "postprocess")
builder.add_edge("postprocess",  END)

# Compile — produces a runnable Pregel graph
graph = builder.compile()

# ---------------------------------------------------------------------------
# Step 4: Run it
# ---------------------------------------------------------------------------

print("=" * 60)
print("DEMO 01 — LangGraph Hello World")
print("=" * 60)
print()

questions = [
    "What is LangGraph?",
    "  Why is stateful AI important?  ",   # extra whitespace on purpose
]

for q in questions:
    print(f"INPUT: '{q}'")
    print("-" * 40)
    final_state = graph.invoke({"input": q})
    print()
    print(final_state["final_output"])
    print()
    print("=" * 60)
    print()

# ---------------------------------------------------------------------------
# Step 5: Inspect the graph structure
# ---------------------------------------------------------------------------
print("Graph nodes :", list(graph.nodes))
print()

# Optional: print a Mermaid diagram of the graph
try:
    print("Mermaid diagram:")
    print(graph.get_graph().draw_mermaid())
except Exception:
    pass
