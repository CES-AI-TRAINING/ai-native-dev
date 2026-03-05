# Multi-Agent Systems with LangGraph

## Table of Contents

1. [When Multi-Agent Systems Are Needed](#1-when-multi-agent-systems-are-needed)
2. [LangGraph Concepts: Nodes, Edges, and State](#2-langgraph-concepts-nodes-edges-and-state)
3. [Planner–Executor–Reviewer Pattern](#3-plannerexecutorreviewer-pattern)
4. [Deterministic vs LLM-Driven Orchestration](#4-deterministic-vs-llm-driven-orchestration)
5. [Human Approval Loops](#5-human-approval-loops)
6. [Putting It All Together: End-to-End Example](#6-putting-it-all-together-end-to-end-example)

---

## 1. When Multi-Agent Systems Are Needed

A single LLM call handles a narrow, well-scoped task well. As complexity grows, a single agent becomes a bottleneck: context windows fill up, tool calls become entangled, and error recovery is fragile. Multi-agent systems decompose a problem across several specialized agents that cooperate.

### Signs You Need Multiple Agents

| Signal                                                   | Why a Single Agent Falls Short                                    |
| -------------------------------------------------------- | ----------------------------------------------------------------- |
| Task requires distinct roles (research, coding, review)  | One prompt cannot be optimized for all roles simultaneously       |
| Long workflows with many steps                           | Context window overflow; hard to track partial progress           |
| Parallel sub-tasks                                       | A single agent is sequential by nature                            |
| Different tools per sub-task                             | Tool pollution — all tools visible to one agent creates confusion |
| You need a human checkpoint before spending more compute | Single agents cannot pause themselves reliably                    |
| Different reliability/cost profiles needed per step      | You want GPT-4o for reasoning but a cheaper model for formatting  |

### Common Use Cases

- **Research pipelines**: Planner decides what to research; Search agent fetches; Summarizer distills; Critic evaluates quality.
- **Software engineering**: Architect designs; Coder implements; Tester writes tests; Reviewer approves.
- **Document processing**: Extractor pulls data; Transformer normalises; Validator checks; Writer produces output.
- **Customer support**: Classifier routes; Specialist agents handle billing, technical, or escalation paths.

---

## 2. LangGraph Concepts: Nodes, Edges, and State

LangGraph models an agentic workflow as a **directed graph** where data flows through nodes connected by edges. Every step in the workflow is a node; routing decisions are encoded in edges.

### 2.1 State

State is a typed Python `TypedDict` (or Pydantic model) that is passed between every node. Each node **reads** the current state and **returns a partial update** — LangGraph merges the returned dict back into the global state automatically.

```python
from typing import TypedDict, Annotated, List
import operator

class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]  # append-only list
    plan: str
    result: str
    approved: bool
    iteration: int
```

Key design rules:

- Keep state **flat** where possible — deeply nested dicts are harder to debug.
- Use `Annotated[List[T], operator.add]` for fields that should accumulate (e.g., message history).
- Use plain assignment for fields that should be overwritten (e.g., `plan`, `result`).

### 2.2 Nodes

A node is a regular Python function (or async function) that:

1. Accepts the current `AgentState`.
2. Performs work (LLM call, tool call, computation, etc.).
3. Returns a **dict** containing only the fields that changed.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def planner_node(state: AgentState) -> dict:
    """Breaks the user request into an ordered plan."""
    response = llm.invoke([
        SystemMessage(content="You are a planning assistant. Break the task into clear steps."),
        HumanMessage(content=state["messages"][-1]),
    ])
    return {"plan": response.content}

def executor_node(state: AgentState) -> dict:
    """Carries out each step in the plan."""
    response = llm.invoke([
        SystemMessage(content="You are an executor. Follow the plan exactly."),
        HumanMessage(content=f"Plan:\n{state['plan']}\n\nExecute this plan now."),
    ])
    return {
        "result": response.content,
        "messages": [response.content],
        "iteration": state.get("iteration", 0) + 1,
    }

def reviewer_node(state: AgentState) -> dict:
    """Evaluates the result and decides if it is acceptable."""
    response = llm.invoke([
        SystemMessage(content=(
            "You are a quality reviewer. Reply with JSON: "
            '{"approved": true/false, "feedback": "..."}'
        )),
        HumanMessage(content=f"Result to review:\n{state['result']}"),
    ])
    import json
    data = json.loads(response.content)
    return {
        "approved": data["approved"],
        "messages": [f"Reviewer: {data['feedback']}"],
    }
```

### 2.3 Edges

Edges define how control flows between nodes. There are two types:

#### Unconditional edges

Always move from one node to another.

```python
graph.add_edge("planner", "executor")
```

#### Conditional edges

A function inspects the state and returns the **name** of the next node (or `END`).

```python
from langgraph.graph import END

def route_after_review(state: AgentState) -> str:
    if state["approved"]:
        return END
    if state.get("iteration", 0) >= 3:
        return END          # give up after 3 retries
    return "executor"       # loop back and retry

graph.add_conditional_edges(
    "reviewer",
    route_after_review,
    {END: END, "executor": "executor"},
)
```

### 2.4 Building and Compiling a Graph

```python
from langgraph.graph import StateGraph, END

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("planner",  planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("reviewer", reviewer_node)

    # Set the entry point
    graph.set_entry_point("planner")

    # Add edges
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {END: END, "executor": "executor"},
    )

    return graph.compile()

app = build_graph()
```

### 2.5 Running the Graph

```python
initial_state: AgentState = {
    "messages": ["Write a Python function to sort a list of dicts by a given key."],
    "plan": "",
    "result": "",
    "approved": False,
    "iteration": 0,
}

final_state = app.invoke(initial_state)
print(final_state["result"])
```

### 2.6 Visual Summary

```
[START]
   │
   ▼
[planner]  ─────────────►  [executor]  ─────────────►  [reviewer]
                               ▲                            │
                               │      approved=False        │
                               └────────────────────────────┘
                                        approved=True
                                              │
                                            [END]
```

---

## 3. Planner–Executor–Reviewer Pattern

This three-role pattern is the most widely used structure for reliable agentic workflows. Each role has a clear responsibility, which makes the system easier to debug and extend.

### Role Responsibilities

```
┌─────────────────────────────────────────────────────────────────┐
│  PLANNER                                                        │
│  • Understands the user's goal                                  │
│  • Breaks it into ordered, atomic steps                         │
│  • Does NOT execute anything                                    │
│  • Output: structured plan (numbered steps or JSON)             │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXECUTOR                                                       │
│  • Receives the plan                                            │
│  • Calls tools, APIs, code runners                              │
│  • Does NOT judge quality                                       │
│  • Output: raw result of execution                              │
└─────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  REVIEWER                                                       │
│  • Evaluates result against original goal                       │
│  • Routes: approve → END, reject → Executor (with feedback)     │
│  • Optionally escalates to human                                │
└─────────────────────────────────────────────────────────────────┘
```

### Full Implementation

```python
import json
import operator
from typing import TypedDict, Annotated, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── State ──────────────────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    task: str
    plan: str
    result: str
    feedback: str
    approved: bool
    iteration: int
    history: Annotated[List[str], operator.add]

# ── Nodes ──────────────────────────────────────────────────────────────────────

PLANNER_PROMPT = """You are a meticulous planner.
Given a task, output a numbered list of concrete steps to accomplish it.
Be specific and ordered. Do not include any preamble."""

EXECUTOR_PROMPT = """You are a focused executor.
You will receive a task and a numbered plan.
Carry out each step and return the final deliverable.
If a previous review round provided feedback, incorporate it."""

REVIEWER_PROMPT = """You are a strict quality reviewer.
Given the original task and the execution result, decide whether the result
fully satisfies the task requirements.
Respond ONLY with valid JSON in this exact format:
{"approved": true|false, "feedback": "<concise feedback if not approved>"}"""

def planner(state: WorkflowState) -> dict:
    response = llm.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Task: {state['task']}"),
    ])
    return {
        "plan": response.content,
        "history": [f"[PLAN]\n{response.content}"],
    }

def executor(state: WorkflowState) -> dict:
    feedback_section = (
        f"\n\nPrevious reviewer feedback (must be addressed):\n{state['feedback']}"
        if state.get("feedback") else ""
    )
    response = llm.invoke([
        SystemMessage(content=EXECUTOR_PROMPT),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Plan:\n{state['plan']}"
            f"{feedback_section}"
        )),
    ])
    return {
        "result": response.content,
        "iteration": state.get("iteration", 0) + 1,
        "history": [f"[EXECUTE iteration={state.get('iteration',0)+1}]\n{response.content}"],
    }

def reviewer(state: WorkflowState) -> dict:
    response = llm.invoke([
        SystemMessage(content=REVIEWER_PROMPT),
        HumanMessage(content=(
            f"Original task: {state['task']}\n\n"
            f"Result to review:\n{state['result']}"
        )),
    ])
    try:
        data = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: treat as approved to avoid infinite loops
        data = {"approved": True, "feedback": ""}

    return {
        "approved": data["approved"],
        "feedback": data.get("feedback", ""),
        "history": [f"[REVIEW approved={data['approved']}] {data.get('feedback','')}"],
    }

# ── Routing ────────────────────────────────────────────────────────────────────

MAX_ITERATIONS = 3

def should_retry(state: WorkflowState) -> str:
    if state["approved"]:
        return END
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return END
    return "executor"

# ── Graph ──────────────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    g = StateGraph(WorkflowState)

    g.add_node("planner",  planner)
    g.add_node("executor", executor)
    g.add_node("reviewer", reviewer)

    g.set_entry_point("planner")
    g.add_edge("planner",  "executor")
    g.add_edge("executor", "reviewer")
    g.add_conditional_edges("reviewer", should_retry, {END: END, "executor": "executor"})

    return g.compile()

# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    workflow = build_workflow()

    state = workflow.invoke({
        "task": "Write a Python function that merges two sorted lists into one sorted list.",
        "plan": "",
        "result": "",
        "feedback": "",
        "approved": False,
        "iteration": 0,
        "history": [],
    })

    print("=== FINAL RESULT ===")
    print(state["result"])
    print("\n=== HISTORY ===")
    for entry in state["history"]:
        print(entry)
        print("─" * 60)
```

### Why This Pattern Works

- **Separation of concerns**: Prompts can be tuned independently — a planner prompt optimised for structure doesn't interfere with an executor prompt optimised for tool use.
- **Natural retry boundary**: The reviewer controls loop-back, so bad results don't silently propagate.
- **Observability**: The `history` list in state gives a full audit trail.
- **Extensible**: A `critic` node, `human_review` node, or `formatter` node can be inserted without touching existing nodes.

---

## 4. Deterministic vs LLM-Driven Orchestration

Orchestration describes _how the next step is chosen_. There are two fundamentally different strategies.

### 4.1 Deterministic (Rule-Based) Orchestration

The routing function is pure Python — it inspects state fields and returns a fixed next node. **No LLM call is made during routing.**

```python
# Deterministic router — branching on a typed state field
def route_on_task_type(state: WorkflowState) -> str:
    task_type = state["task_type"]           # set during initial classification
    routing_table = {
        "code":     "code_agent",
        "research": "research_agent",
        "math":     "math_agent",
    }
    return routing_table.get(task_type, "fallback_agent")
```

**Characteristics**

| Property       | Value                                         |
| -------------- | --------------------------------------------- |
| Latency        | Near-zero (no LLM call)                       |
| Cost           | Zero extra tokens                             |
| Predictability | Fully deterministic given the same state      |
| Flexibility    | Fixed; adding a new path requires code change |
| Best for       | Known, enumerable routing scenarios           |

**Example: multi-domain customer support**

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class SupportState(TypedDict):
    query: str
    category: str      # set by classifier node
    response: str

def classifier(state: SupportState) -> dict:
    # LLM classifies once; router uses the result deterministically
    response = llm.invoke([
        HumanMessage(content=(
            f"Classify this query into exactly one of: billing, technical, general.\n"
            f"Query: {state['query']}\nReply with the category word only."
        ))
    ])
    return {"category": response.content.strip().lower()}

def billing_agent(state: SupportState) -> dict:
    # Billing-specific prompt and tools
    return {"response": "Handled by billing agent."}

def technical_agent(state: SupportState) -> dict:
    return {"response": "Handled by technical agent."}

def general_agent(state: SupportState) -> dict:
    return {"response": "Handled by general agent."}

def route_support(state: SupportState) -> str:
    return {
        "billing":   "billing_agent",
        "technical": "technical_agent",
    }.get(state["category"], "general_agent")

g = StateGraph(SupportState)
g.add_node("classifier",     classifier)
g.add_node("billing_agent",  billing_agent)
g.add_node("technical_agent",technical_agent)
g.add_node("general_agent",  general_agent)

g.set_entry_point("classifier")
g.add_conditional_edges(
    "classifier", route_support,
    {"billing_agent": "billing_agent",
     "technical_agent": "technical_agent",
     "general_agent": "general_agent"},
)
g.add_edge("billing_agent",   END)
g.add_edge("technical_agent", END)
g.add_edge("general_agent",   END)

app = g.compile()
```

---

### 4.2 LLM-Driven (Autonomous) Orchestration

The LLM itself decides the next step by reasoning about available options and outputting structured JSON. This is also called **ReAct** (Reason + Act) or **agentic** orchestration.

```python
import json
from langchain_core.messages import HumanMessage, SystemMessage

ORCHESTRATOR_PROMPT = """You are an orchestrator managing a pipeline.
Based on the current state, decide the next action.
Available actions: ["run_research", "run_code", "run_review", "finish"]
Respond ONLY with JSON: {"action": "<action>", "reason": "<one sentence>"}"""

def llm_orchestrator(state: dict) -> str:
    response = llm.invoke([
        SystemMessage(content=ORCHESTRATOR_PROMPT),
        HumanMessage(content=f"Current state summary:\n{json.dumps(state, indent=2)}"),
    ])
    data = json.loads(response.content)
    action_map = {
        "run_research": "research_agent",
        "run_code":     "code_agent",
        "run_review":   "reviewer_agent",
        "finish":       END,
    }
    return action_map.get(data["action"], END)
```

**Characteristics**

| Property       | Value                                                     |
| -------------- | --------------------------------------------------------- |
| Latency        | +1 LLM call per routing decision                          |
| Cost           | Extra tokens per step                                     |
| Predictability | Non-deterministic (same input can yield different routes) |
| Flexibility    | High; LLM can reason about novel situations               |
| Best for       | Open-ended tasks where the path is unknown in advance     |

---

### 4.3 Hybrid Approach (Recommended)

Most production systems combine both:

- **Deterministic** for known, high-frequency, low-risk branches (e.g., "if error → error handler").
- **LLM-driven** for open-ended decision points that cannot be enumerated upfront.

```python
def smart_router(state: WorkflowState) -> str:
    # Fast deterministic checks first
    if state.get("critical_error"):
        return "error_handler"
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "summarizer"

    # Fall through to LLM decision only when needed
    return llm_orchestrator(state)
```

---

## 5. Human Approval Loops

Not every decision should be fully automated. Human-in-the-loop (HITL) patterns let you pause a workflow, surface it to a person, and resume based on their decision.

### 5.1 Why Human Approval Matters

- **High-stakes actions**: sending emails, deploying code, making purchases.
- **Low-confidence outputs**: when the LLM is uncertain, escalate instead of guessing.
- **Regulatory requirements**: some domains require a human sign-off before action.
- **Trust building**: early deployments should have more checkpoints; relax them over time as confidence grows.

### 5.2 Implementation with `interrupt_before`

LangGraph's `compile(interrupt_before=[...])` pauses execution **before** the named node and allows external input to be injected before resuming.

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Optional

class ApprovalState(TypedDict):
    task: str
    plan: str
    result: str
    human_decision: Optional[str]   # "approve" | "reject" | "modify"
    human_feedback: Optional[str]

# ── Nodes ──────────────────────────────────────────────────────────────────────

def planner(state: ApprovalState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Create a brief action plan for the task."),
        HumanMessage(content=state["task"]),
    ])
    return {"plan": response.content}

def human_approval_node(state: ApprovalState) -> dict:
    """
    This node is the pause point.
    In production, it would notify a human (Slack, email, UI) and wait.
    LangGraph interrupts BEFORE this node when interrupt_before is set,
    so this function body only runs after the human has injected their decision.
    """
    decision = state.get("human_decision", "approve")
    if decision == "reject":
        return {"plan": "", "human_feedback": state.get("human_feedback", "")}
    return {}

def executor(state: ApprovalState) -> dict:
    response = llm.invoke([
        SystemMessage(content="Execute the following plan precisely."),
        HumanMessage(content=f"Plan:\n{state['plan']}"),
    ])
    return {"result": response.content}

def rejected_handler(state: ApprovalState) -> dict:
    print(f"Workflow rejected by human. Feedback: {state.get('human_feedback')}")
    return {}

# ── Routing ────────────────────────────────────────────────────────────────────

def route_after_human(state: ApprovalState) -> str:
    decision = state.get("human_decision", "approve")
    if decision == "reject":
        return "rejected_handler"
    return "executor"

# ── Graph with checkpointing ───────────────────────────────────────────────────

checkpointer = MemorySaver()   # use SqliteSaver or RedisSaver in production

def build_approval_workflow():
    g = StateGraph(ApprovalState)

    g.add_node("planner",          planner)
    g.add_node("human_approval",   human_approval_node)
    g.add_node("executor",         executor)
    g.add_node("rejected_handler", rejected_handler)

    g.set_entry_point("planner")
    g.add_edge("planner", "human_approval")
    g.add_conditional_edges(
        "human_approval", route_after_human,
        {"executor": "executor", "rejected_handler": "rejected_handler"},
    )
    g.add_edge("executor",         END)
    g.add_edge("rejected_handler", END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"],  # pause here and wait for human input
    )

app = build_approval_workflow()
```

### 5.3 Running a Workflow with Human-in-the-Loop

```python
import uuid

thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# ── Step 1: Start the workflow (runs until interrupt) ──────────────────────────
print("Starting workflow...")
for event in app.stream(
    {
        "task": "Draft and send a summary email to the team about Q1 results.",
        "plan": "",
        "result": "",
        "human_decision": None,
        "human_feedback": None,
    },
    config=config,
):
    print(event)
# Workflow pauses before "human_approval"

# ── Step 2: Human reviews the plan ────────────────────────────────────────────
snapshot = app.get_state(config)
print("\n=== PLAN FOR HUMAN REVIEW ===")
print(snapshot.values["plan"])

# Simulate a human decision (in production, this comes from a UI / webhook)
human_response = {
    "human_decision": "approve",   # or "reject"
    "human_feedback": None,
}
# Inject the decision into the paused state
app.update_state(config, human_response)

# ── Step 3: Resume the workflow ────────────────────────────────────────────────
print("\nResuming workflow after human approval...")
for event in app.stream(None, config=config):
    print(event)

final = app.get_state(config)
print("\n=== FINAL RESULT ===")
print(final.values["result"])
```

### 5.4 Advanced: Conditional Interrupts

You don't always need to interrupt. Pause only when confidence is low or when the action is high-risk:

```python
class SmartApprovalState(TypedDict):
    task: str
    plan: str
    confidence_score: float   # 0.0 – 1.0, set by planner
    result: str
    human_decision: Optional[str]

CONFIDENCE_THRESHOLD = 0.75

def needs_human_review(state: SmartApprovalState) -> str:
    """Interrupt only when confidence is below threshold."""
    if state.get("confidence_score", 0.0) < CONFIDENCE_THRESHOLD:
        return "human_approval"
    return "executor"    # skip human review for high-confidence plans

g.add_conditional_edges(
    "planner", needs_human_review,
    {"human_approval": "human_approval", "executor": "executor"},
)
```

### 5.5 Production Checkpointers

| Backend    | Import                                          | Use Case                   |
| ---------- | ----------------------------------------------- | -------------------------- |
| In-memory  | `MemorySaver`                                   | Development / testing      |
| SQLite     | `SqliteSaver`                                   | Single-process production  |
| PostgreSQL | `PostgresSaver` (langgraph-checkpoint-postgres) | Multi-process, cloud       |
| Redis      | `RedisSaver` (langgraph-checkpoint-redis)       | High-throughput, real-time |

```python
# PostgreSQL example
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://user:pass@host:5432/db") as checkpointer:
    app = build_approval_workflow_with(checkpointer)
```

---

## 6. Putting It All Together: End-to-End Example

The following combines every concept: Planner–Executor–Reviewer pattern, hybrid deterministic+LLM orchestration, and an optional human approval gate.

```python
"""
End-to-end multi-agent research-and-write workflow.

Agents:
  1. Classifier  – deterministically routes by task type
  2. Planner     – creates a structured plan
  3. [Optional Human Gate] – pauses for approval on high-risk tasks
  4. Researcher  – gathers information (tool-equipped)
  5. Writer      – drafts the deliverable
  6. Reviewer    – approves or requests revision (up to n retries)
"""

import json
import operator
import uuid
from typing import TypedDict, Annotated, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ── State ──────────────────────────────────────────────────────────────────────

class FullState(TypedDict):
    task: str
    task_type: str                          # "research" | "code" | "analysis"
    is_high_risk: bool
    plan: str
    research_notes: str
    draft: str
    approved: bool
    feedback: str
    iteration: int
    human_decision: Optional[str]
    log: Annotated[List[str], operator.add]

MAX_RETRIES = 3

# ── Helper ─────────────────────────────────────────────────────────────────────

def invoke(system: str, user: str) -> str:
    return llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]).content

# ── Nodes ──────────────────────────────────────────────────────────────────────

def classifier(state: FullState) -> dict:
    raw = invoke(
        "Classify the task. Reply with JSON only: "
        '{"task_type": "research|code|analysis", "is_high_risk": true|false}',
        state["task"],
    )
    data = json.loads(raw)
    return {
        "task_type":   data["task_type"],
        "is_high_risk": data["is_high_risk"],
        "log": [f"Classified as '{data['task_type']}', high_risk={data['is_high_risk']}"],
    }

def planner(state: FullState) -> dict:
    plan = invoke(
        "Create a numbered action plan. Be concise and specific.",
        f"Task: {state['task']}\nTask type: {state['task_type']}",
    )
    return {"plan": plan, "log": [f"[PLAN]\n{plan}"]}

def human_gate(state: FullState) -> dict:
    # Body runs only after human has injected their decision via update_state
    return {}

def researcher(state: FullState) -> dict:
    notes = invoke(
        "You are a researcher. Gather relevant facts and context.",
        f"Task: {state['task']}\nPlan:\n{state['plan']}",
    )
    return {"research_notes": notes, "log": [f"[RESEARCH]\n{notes}"]}

def writer(state: FullState) -> dict:
    fb = f"\n\nAddress this feedback: {state['feedback']}" if state.get("feedback") else ""
    draft = invoke(
        "You are a writer. Produce the final deliverable based on research.",
        f"Task: {state['task']}\nResearch:\n{state['research_notes']}{fb}",
    )
    return {
        "draft":     draft,
        "iteration": state.get("iteration", 0) + 1,
        "log":       [f"[DRAFT {state.get('iteration',0)+1}]\n{draft}"],
    }

def reviewer(state: FullState) -> dict:
    raw = invoke(
        'Review the draft. Reply with JSON only: {"approved": true|false, "feedback": "..."}',
        f"Task: {state['task']}\nDraft:\n{state['draft']}",
    )
    data = json.loads(raw)
    return {
        "approved": data["approved"],
        "feedback": data.get("feedback", ""),
        "log":      [f"[REVIEW approved={data['approved']}] {data.get('feedback','')}"],
    }

# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_classify(state: FullState) -> str:
    """Deterministic: always go to planner (type used downstream)."""
    return "planner"

def route_after_plan(state: FullState) -> str:
    """Deterministic: pause for human if high-risk, else proceed."""
    if state.get("is_high_risk"):
        return "human_gate"
    return "researcher"

def route_after_human(state: FullState) -> str:
    decision = state.get("human_decision", "approve")
    return "researcher" if decision == "approve" else END

def route_after_review(state: FullState) -> str:
    if state["approved"]:
        return END
    if state.get("iteration", 0) >= MAX_RETRIES:
        return END
    return "writer"

# ── Graph assembly ─────────────────────────────────────────────────────────────

checkpointer = MemorySaver()

def build_full_workflow():
    g = StateGraph(FullState)

    for name, fn in [
        ("classifier",  classifier),
        ("planner",     planner),
        ("human_gate",  human_gate),
        ("researcher",  researcher),
        ("writer",      writer),
        ("reviewer",    reviewer),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("classifier")
    g.add_conditional_edges("classifier", route_after_classify,
                            {"planner": "planner"})
    g.add_conditional_edges("planner",    route_after_plan,
                            {"human_gate": "human_gate", "researcher": "researcher"})
    g.add_conditional_edges("human_gate", route_after_human,
                            {"researcher": "researcher", END: END})
    g.add_edge("researcher", "writer")
    g.add_conditional_edges("reviewer", route_after_review,
                            {END: END, "writer": "writer"})
    g.add_edge("writer", "reviewer")

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_gate"],
    )

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    workflow       = build_full_workflow()
    thread_config  = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state: FullState = {
        "task":          "Research and write a short report on the impact of LLMs on software testing.",
        "task_type":     "",
        "is_high_risk":  False,
        "plan":          "",
        "research_notes":"",
        "draft":         "",
        "approved":      False,
        "feedback":      "",
        "iteration":     0,
        "human_decision": None,
        "log":           [],
    }

    # Run until potential interrupt
    for chunk in workflow.stream(initial_state, thread_config):
        node_name = list(chunk.keys())[0]
        print(f"✓ {node_name}")

    snapshot = workflow.get_state(thread_config)
    if snapshot.next:
        print(f"\n⏸  Paused before: {snapshot.next}")
        print(f"Plan to approve:\n{snapshot.values['plan']}")
        # Inject human approval
        workflow.update_state(thread_config, {"human_decision": "approve"})
        for chunk in workflow.stream(None, thread_config):
            node_name = list(chunk.keys())[0]
            print(f"✓ {node_name}")

    final = workflow.get_state(thread_config)
    print("\n=== FINAL DRAFT ===")
    print(final.values["draft"])
```

---

## Quick Reference

### LangGraph API Cheat Sheet

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Build
g = StateGraph(MyState)
g.add_node("name", function)
g.set_entry_point("first_node")

# Unconditional edge
g.add_edge("node_a", "node_b")

# Conditional edge
g.add_conditional_edges("node_a", routing_fn, {"next1": "node_b", END: END})

# Compile (with optional HITL checkpoint)
app = g.compile(checkpointer=MemorySaver(), interrupt_before=["approval_node"])

# Run (full)
result = app.invoke(initial_state)

# Run (streaming)
for chunk in app.stream(initial_state, config):
    print(chunk)

# HITL: inspect paused state
snapshot = app.get_state(config)

# HITL: inject decision and resume
app.update_state(config, {"human_decision": "approve"})
app.stream(None, config)
```

### Pattern Decision Matrix

| Requirement                           | Recommended Pattern                       |
| ------------------------------------- | ----------------------------------------- |
| Fixed, enumerable routes              | Deterministic routing                     |
| Dynamic, open-ended routing           | LLM-driven orchestration                  |
| Iterative quality improvement         | Planner–Executor–Reviewer loop            |
| High-stakes or low-confidence actions | Human approval with `interrupt_before`    |
| Parallel independent sub-tasks        | Fan-out graph with join node              |
| Different LLMs per role               | Assign different `llm` instances per node |

### Dependencies

```bash
pip install langgraph langchain-openai langchain-core
# Optional checkpointers:
pip install langgraph-checkpoint-postgres
pip install langgraph-checkpoint-redis
```
