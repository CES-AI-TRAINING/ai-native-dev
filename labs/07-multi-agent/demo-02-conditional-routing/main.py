"""
Demo 02: Conditional Routing in LangGraph

LangGraph lets you route execution to different nodes at runtime using
*conditional edges*. A routing function inspects the current state and
returns the name of the next node (or END).

Topics covered:
1. add_conditional_edges — route based on state at runtime
2. Writing a routing function
3. Multiple branches in a single graph
4. Converging branches back to a shared node

Graph shape:
              ┌─ positive_node ─┐
START → classify ─┤               ├─ format → END
              └─ negative_node ─┘
              └─ neutral_node  ─┘
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set. Copy .env.example → .env")

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    temperature=0,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(TypedDict):
    text: str                                           # user text to classify
    sentiment: Literal["positive", "negative", "neutral"]  # filled by classify node
    branch_output: str                                  # filled by branch node
    final_response: str                                 # filled by format node


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_node(state: State) -> dict:
    """Use an LLM to classify the sentiment of the input text."""
    text = state["text"]
    prompt = [
        SystemMessage(content=(
            "Classify the sentiment of the following text. "
            "Reply with exactly one word: positive, negative, or neutral."
        )),
        HumanMessage(content=text),
    ]
    result = llm.invoke(prompt).content.strip().lower()
    sentiment: Literal["positive", "negative", "neutral"] = (
        result if result in ("positive", "negative", "neutral") else "neutral"
    )
    print(f"[classify_node] text='{text[:50]}' → sentiment='{sentiment}'")
    return {"sentiment": sentiment}


def positive_node(state: State) -> dict:
    """Handle positive sentiment."""
    output = f"😊 Great to hear something positive! '{state['text']}'"
    print(f"[positive_node] handling positive sentiment")
    return {"branch_output": output}


def negative_node(state: State) -> dict:
    """Handle negative sentiment."""
    output = f"😞 I'm sorry to hear that. Let me help: '{state['text']}'"
    print(f"[negative_node] handling negative sentiment")
    return {"branch_output": output}


def neutral_node(state: State) -> dict:
    """Handle neutral sentiment."""
    output = f"🤔 Interesting observation: '{state['text']}'"
    print(f"[neutral_node] handling neutral sentiment")
    return {"branch_output": output}


def format_node(state: State) -> dict:
    """Format final response (merged convergence point)."""
    final = f"[Sentiment: {state['sentiment'].upper()}]\n{state['branch_output']}"
    print(f"[format_node] assembling final response")
    return {"final_response": final}


# ---------------------------------------------------------------------------
# Routing function
#
# This function is the key to conditional edges. It accepts the current
# state and must return the name of the next node (or END).
# ---------------------------------------------------------------------------

def route_by_sentiment(state: State) -> Literal["positive", "negative", "neutral"]:
    """Return the branch node name based on classified sentiment."""
    return state["sentiment"]   # "positive" | "negative" | "neutral"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

builder = StateGraph(State)

builder.add_node("classify",  classify_node)
builder.add_node("positive",  positive_node)
builder.add_node("negative",  negative_node)
builder.add_node("neutral",   neutral_node)
builder.add_node("format",    format_node)

builder.add_edge(START, "classify")

# --- Conditional edge ---------------------------------------------------
# After "classify" runs, call route_by_sentiment(state).
# The returned string determines which node executes next.
builder.add_conditional_edges(
    "classify",                     # source node
    route_by_sentiment,             # routing function
    {                               # mapping: return value → node name
        "positive": "positive",
        "negative": "negative",
        "neutral":  "neutral",
    },
)

# All branches converge back at "format"
builder.add_edge("positive", "format")
builder.add_edge("negative", "format")
builder.add_edge("neutral",  "format")
builder.add_edge("format",   END)

graph = builder.compile()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("=" * 60)
print("DEMO 02 — Conditional Routing")
print("=" * 60)
print()

samples = [
    "I absolutely love this product! It changed my life.",
    "This is the worst experience I've ever had. Terrible.",
    "The package arrived on Tuesday. It contains three items.",
]

for text in samples:
    print(f"INPUT: '{text}'")
    print("-" * 50)
    result = graph.invoke({"text": text})
    print()
    print(result["final_response"])
    print()
    print("=" * 60)
    print()

# Show routing diagram
print("Mermaid diagram:")
try:
    print(graph.get_graph().draw_mermaid())
except Exception:
    pass
