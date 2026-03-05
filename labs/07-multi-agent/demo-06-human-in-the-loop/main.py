"""
Demo 06: Human-in-the-Loop with LangGraph

Sometimes an AI agent needs explicit human approval before proceeding —
for example before sending an email, deleting data, or spending money.
LangGraph supports this via *interrupt* and *Command(resume=...)*.

Topics covered:
1. interrupt() — pause graph execution and wait for human input
2. MemorySaver — required so the graph can be resumed
3. Command(resume=...) — inject human input to resume
4. graph.get_state() — inspect what the graph is waiting for
5. interrupt_before — alternative approach to pause before a node

Workflow being modelled:
  User requests an action → AI drafts a plan → **Human approves / rejects / edits**
  → AI executes (if approved) → Done

Graph shape:
  START → draft_plan → [INTERRUPT] → execute_plan → END
                              ↑
                     human reviews & decides
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing import Annotated

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set.")

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.3)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str           # the user's requested task
    plan: str           # AI's drafted execution plan
    human_decision: str # "approved" | "rejected" | custom instruction
    result: str         # final execution result


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def draft_plan_node(state: State) -> dict:
    """AI drafts a step-by-step plan for the requested task."""
    task = state["task"]
    print(f"\n[draft_plan] Generating plan for: '{task}'")
    
    response = llm.invoke([
        HumanMessage(content=(
            f"You are a careful AI assistant. Draft a concise, numbered step-by-step plan "
            f"to accomplish this task:\n\n{task}\n\n"
            f"Keep it to 3-5 steps. Be specific."
        ))
    ])
    plan = response.content
    print(f"[draft_plan] Plan drafted:\n{plan}\n")
    return {"plan": plan, "messages": [AIMessage(content=f"I've drafted a plan:\n{plan}")]}


def human_review_node(state: State) -> dict:
    """Pause here and wait for human review/approval."""
    plan = state["plan"]
    
    print("\n" + "="*50)
    print("⏸  PAUSED — Waiting for human review")
    print("="*50)
    print(f"\nPlan to review:\n{plan}\n")
    
    # interrupt() suspends the graph and surfaces a value to the caller.
    # Execution resumes when graph.invoke(Command(resume=...)) is called.
    human_input = interrupt({
        "message": "Please review the plan above.",
        "options": ["approved", "rejected", "<type custom instruction>"],
        "plan": plan,
    })
    
    print(f"\n[human_review] Human responded: '{human_input}'")
    return {
        "human_decision": human_input,
        "messages": [HumanMessage(content=f"Human review: {human_input}")],
    }


def execute_plan_node(state: State) -> dict:
    """Execute, modify, or abandon based on human decision."""
    decision = state.get("human_decision", "")
    plan = state["plan"]
    task = state["task"]
    
    if decision.lower() == "rejected":
        result = "❌ Plan was rejected by the human reviewer. Task aborted."
        print(f"[execute_plan] {result}")
    elif decision.lower() == "approved":
        print(f"[execute_plan] Plan approved! Simulating execution...")
        response = llm.invoke([
            HumanMessage(content=(
                f"Task: {task}\n\nApproved plan:\n{plan}\n\n"
                f"Simulate executing this plan. Provide a brief execution summary."
            ))
        ])
        result = f"✅ Execution complete:\n{response.content}"
        print(f"[execute_plan] {result[:100]}...")
    else:
        # Human provided custom instruction
        print(f"[execute_plan] Executing with custom instruction: '{decision}'")
        response = llm.invoke([
            HumanMessage(content=(
                f"Task: {task}\n\nOriginal plan:\n{plan}\n\n"
                f"Human's revision instruction: {decision}\n\n"
                f"Execute with this revision. Provide a brief execution summary."
            ))
        ])
        result = f"✅ Executed with revision:\n{response.content}"
        print(f"[execute_plan] {result[:100]}...")
    
    return {
        "result": result,
        "messages": [AIMessage(content=result)],
    }


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

memory = MemorySaver()   # checkpointer is REQUIRED for interrupt to work

builder = StateGraph(State)
builder.add_node("draft_plan",    draft_plan_node)
builder.add_node("human_review",  human_review_node)
builder.add_node("execute_plan",  execute_plan_node)

builder.add_edge(START,           "draft_plan")
builder.add_edge("draft_plan",    "human_review")
builder.add_edge("human_review",  "execute_plan")
builder.add_edge("execute_plan",  END)

graph = builder.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# Demo run
# ---------------------------------------------------------------------------

print("=" * 60)
print("DEMO 06 — Human-in-the-Loop")
print("=" * 60)

scenarios = [
    {
        "thread_id": "scenario-1",
        "task": "Send a bulk promotional email to all 10,000 customers announcing a 30% discount",
        "human_decision": "approved",
        "description": "Scenario 1: Human approves the plan",
    },
    {
        "thread_id": "scenario-2",
        "task": "Delete all test records from the production database older than 90 days",
        "human_decision": "rejected",
        "description": "Scenario 2: Human rejects the plan",
    },
    {
        "thread_id": "scenario-3",
        "task": "Post a tweet announcing our new AI product launch",
        "human_decision": "Change the tone to be more casual and add relevant hashtags",
        "description": "Scenario 3: Human provides custom revision",
    },
]

for scenario in scenarios:
    print(f"\n{'#' * 60}")
    print(f"  {scenario['description']}")
    print(f"{'#' * 60}")
    
    config = {"configurable": {"thread_id": scenario["thread_id"]}}
    
    # ── Step 1: Run until the graph pauses at interrupt ──────────
    initial_state: State = {
        "messages": [],
        "task": scenario["task"],
        "plan": "",
        "human_decision": "",
        "result": "",
    }
    
    try:
        result = graph.invoke(initial_state, config=config)
        # If we reach here, graph completed without interrupting (unlikely)
        print("Graph completed without interrupt.")
    except Exception:
        # Graph paused — this is expected with interrupt()
        pass
    
    # Check what the graph is waiting for
    state_snapshot = graph.get_state(config)
    print(f"\n  Graph is waiting at: {state_snapshot.next}")
    
    # ── Step 2: Resume with human decision ───────────────────────
    print(f"\n  Human's decision: '{scenario['human_decision']}'")
    print(f"  Resuming graph...")
    
    final = graph.invoke(
        Command(resume=scenario["human_decision"]),
        config=config,
    )
    
    print(f"\n  FINAL RESULT:")
    print(f"  {final.get('result', 'N/A')[:200]}")

print("\n" + "=" * 60)
print("All scenarios complete.")
print("Key takeaway: interrupt() pauses execution; Command(resume=...) continues it.")
