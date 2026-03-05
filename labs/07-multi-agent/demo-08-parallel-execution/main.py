"""
Demo 08: Parallel Node Execution — Fan-Out & Fan-In

LangGraph can run multiple nodes in **parallel** by giving them all the
same source edge. Their outputs are merged back into the state before the
next node runs. This is the *fan-out / fan-in* pattern.

Topics covered:
1. Fan-out: one source node, multiple parallel destination nodes
2. Annotated state with operator.add to accumulate parallel outputs
3. Fan-in: a single aggregation node that combines parallel results
4. Measuring performance: sequential vs. parallel execution time
5. send() API (alternative approach) for dynamic parallelism

Workflow: analyse a product review from 4 angles simultaneously, then
synthesise into a final report.

Graph shape:
                    ┌─► sentiment_analyst ─┐
                    ├─► topic_extractor    ─┤
START → preprocess ─┤                       ├─► synthesise → END
                    ├─► action_detector    ─┤
                    └─► quality_scorer     ─┘
"""

import os
import time
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set.")

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.3)

# ---------------------------------------------------------------------------
# State — parallel branches write to a list via operator.add
# ---------------------------------------------------------------------------

class State(TypedDict):
    review_text: str
    cleaned_text: str
    analyses:      Annotated[list[str], operator.add]   # fan-in accumulator
    final_report:  str


# ---------------------------------------------------------------------------
# Preprocessing node
# ---------------------------------------------------------------------------

def preprocess_node(state: State) -> dict:
    text = state["review_text"].strip()
    print(f"[preprocess] input: '{text[:60]}...'")
    return {"cleaned_text": text}


# ---------------------------------------------------------------------------
# Parallel analysis nodes — each appends one item to `analyses`
# ---------------------------------------------------------------------------

def _call_llm(system: str, text: str) -> str:
    """Helper: call LLM with a system prompt and return response text."""
    return llm.invoke([
        HumanMessage(content=f"{system}\n\nReview text:\n{text}")
    ]).content


def sentiment_analyst_node(state: State) -> dict:
    print("  [sentiment_analyst] running...")
    t0 = time.time()
    result = _call_llm(
        "Analyse the sentiment of this product review. "
        "Rate it as Positive/Neutral/Negative and give a confidence score 0-100. "
        "Keep your response to 2-3 sentences.",
        state["cleaned_text"],
    )
    elapsed = time.time() - t0
    print(f"  [sentiment_analyst] done in {elapsed:.1f}s")
    return {"analyses": [f"## Sentiment Analysis\n{result}"]}


def topic_extractor_node(state: State) -> dict:
    print("  [topic_extractor] running...")
    t0 = time.time()
    result = _call_llm(
        "Extract the key topics and product aspects mentioned in this review. "
        "List them as bullet points (max 5).",
        state["cleaned_text"],
    )
    elapsed = time.time() - t0
    print(f"  [topic_extractor] done in {elapsed:.1f}s")
    return {"analyses": [f"## Key Topics & Aspects\n{result}"]}


def action_detector_node(state: State) -> dict:
    print("  [action_detector] running...")
    t0 = time.time()
    result = _call_llm(
        "Identify any actionable feedback or improvement requests in this review. "
        "What should the product team do? Keep it to 2-3 bullet points.",
        state["cleaned_text"],
    )
    elapsed = time.time() - t0
    print(f"  [action_detector] done in {elapsed:.1f}s")
    return {"analyses": [f"## Actionable Feedback\n{result}"]}


def quality_scorer_node(state: State) -> dict:
    print("  [quality_scorer] running...")
    t0 = time.time()
    result = _call_llm(
        "Score this product review on quality and helpfulness (1-10 scale). "
        "Is it detailed, specific, and useful? Give a score and brief justification.",
        state["cleaned_text"],
    )
    elapsed = time.time() - t0
    print(f"  [quality_scorer] done in {elapsed:.1f}s")
    return {"analyses": [f"## Review Quality Score\n{result}"]}


# ---------------------------------------------------------------------------
# Fan-in: synthesise all parallel results
# ---------------------------------------------------------------------------

def synthesise_node(state: State) -> dict:
    print("[synthesise] combining parallel analyses...")
    analyses = state["analyses"]
    combined = "\n\n".join(analyses)
    
    summary = llm.invoke([
        HumanMessage(content=(
            f"You have received the following multi-angle analysis of a product review:\n\n"
            f"{combined}\n\n"
            f"Write a concise executive summary (3-5 sentences) suitable for a product manager."
        ))
    ]).content
    
    report = f"# Product Review Analysis Report\n\n{combined}\n\n---\n## Executive Summary\n{summary}"
    print("[synthesise] report complete.")
    return {"final_report": report}


# ---------------------------------------------------------------------------
# Build graph with fan-out
# ---------------------------------------------------------------------------

builder = StateGraph(State)

builder.add_node("preprocess",        preprocess_node)
builder.add_node("sentiment_analyst", sentiment_analyst_node)
builder.add_node("topic_extractor",   topic_extractor_node)
builder.add_node("action_detector",   action_detector_node)
builder.add_node("quality_scorer",    quality_scorer_node)
builder.add_node("synthesise",        synthesise_node)

builder.add_edge(START, "preprocess")

# Fan-out: preprocess → all four parallel nodes simultaneously
builder.add_edge("preprocess", "sentiment_analyst")
builder.add_edge("preprocess", "topic_extractor")
builder.add_edge("preprocess", "action_detector")
builder.add_edge("preprocess", "quality_scorer")

# Fan-in: all four → synthesise
builder.add_edge("sentiment_analyst", "synthesise")
builder.add_edge("topic_extractor",   "synthesise")
builder.add_edge("action_detector",   "synthesise")
builder.add_edge("quality_scorer",    "synthesise")

builder.add_edge("synthesise", END)

graph = builder.compile()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

REVIEW = """
I've been using this wireless noise-cancelling headphone for three months now. 
The sound quality is exceptional — the bass is deep without being overwhelming, 
and the highs are crisp and clear. The noise cancellation feature is a game-changer 
for my daily commute. Battery life is good, lasting about 20 hours per charge.

However, there are a few downsides. The ear cushions start feeling uncomfortable 
after about two hours of use. The companion app is buggy and crashes frequently. 
Also, the carrying case feels flimsy for a £300 product.

Overall, I'd recommend these to audiophiles who prioritise sound quality, but 
the app really needs to be fixed. 8/10 — would buy again if the software issues 
are resolved.
"""

print("=" * 60)
print("DEMO 08 — Parallel Node Execution (Fan-Out / Fan-In)")
print("=" * 60)
print()
print(f"Review ({len(REVIEW.split())} words):")
print(REVIEW.strip())
print()
print("-" * 60)
print("Running 4 analyses in parallel...")
print()

start = time.time()

result = graph.invoke({
    "review_text":  REVIEW,
    "cleaned_text": "",
    "analyses":     [],
    "final_report": "",
})

total = time.time() - start
print()
print(f"Total wall-clock time: {total:.1f}s")
print()
print(result["final_report"][:1200])
print()
print("Mermaid diagram:")
try:
    print(graph.get_graph().draw_mermaid())
except Exception:
    pass
