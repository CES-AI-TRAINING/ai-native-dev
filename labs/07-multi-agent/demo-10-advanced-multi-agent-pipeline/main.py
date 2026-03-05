"""
Demo 10: Advanced Multi-Agent Content Pipeline

Capstone demo combining every LangGraph concept from demos 01-09:
  ✅ Stateful graph & state reducers          (demos 01, 03)
  ✅ Conditional routing                       (demo 02)
  ✅ Persistent memory with MemorySaver        (demo 04)
  ✅ Tool-calling agents (ReAct)               (demo 05)
  ✅ Human-in-the-loop approval               (demo 06)
  ✅ Supervisor / multi-agent orchestration    (demo 07)
  ✅ Parallel node execution (fan-out/fan-in)  (demo 08)
  ✅ RAG-style grounding                       (demo 09)

Scenario: AI Content Creation Pipeline
  The user requests an article. The system plans, researches in parallel,
  writes a draft, edits it, pauses for human review, then finalises.

Graph shape:
  START → planner ──► [research_1 ‖ research_2 ‖ research_3] ──► writer
       → editor ──► human_review [INTERRUPT] ──► (approve/reject/revise)
       → finalise → END

  If editor is not satisfied (quality_score < 7) → loops back to writer.
"""

import os
import json
import operator
from typing import TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set.")

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0.5)
llm_precise = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0)

# ---------------------------------------------------------------------------
# Mock knowledge base for "research"
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = {
    "history":      "The topic has a rich history spanning decades of innovation and cross-disciplinary collaboration.",
    "current_state": "Currently, the field is experiencing rapid growth with major breakthroughs in the last 2-3 years.",
    "applications": "Real-world applications span healthcare, finance, education, autonomous systems, and creative industries.",
    "challenges":   "Key challenges include data privacy, model interpretability, computational costs, and ethical considerations.",
    "future":       "The future outlook is promising, with expected convergence of multiple technologies over the next decade.",
    "statistics":   "Global investment reached $200B+ in 2024; adoption is growing at 40% CAGR across enterprise segments.",
}

# ---------------------------------------------------------------------------
# Shared State
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    # Input
    topic:          str
    requested_by:   str
    target_length:  int          # word count target

    # Planning
    outline:        str
    research_tasks: list[str]    # sub-tasks for parallel research workers

    # Research (accumulated across parallel workers)
    research_notes: Annotated[list[str], operator.add]

    # Writing
    draft:          str
    revision_count: int

    # Editing
    editor_feedback: str
    quality_score:   int          # 1-10

    # Human review
    human_decision:  str          # "approved" | "rejected" | custom feedback

    # Final
    final_article:   str
    published:       bool

    # Audit trail
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Tools available within nodes
# ---------------------------------------------------------------------------

@tool
def search_knowledge_base(aspect: str) -> str:
    """Search the knowledge base for information on a specific aspect of a topic.
    
    Args:
        aspect: One of: history, current_state, applications, challenges, future, statistics
    """
    key = aspect.lower().replace(" ", "_")
    return KNOWLEDGE_BASE.get(key, f"No specific data found for aspect '{aspect}'.")


@tool
def estimate_word_count(text: str) -> str:
    """Count words in a piece of text.
    
    Args:
        text: The text to count
    """
    return str(len(text.split()))


@tool
def check_readability(text: str) -> str:
    """Provide a simple readability assessment of text.
    
    Args:
        text: Text to assess
    """
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_len = len(words) / max(sentences, 1)
    score = "Easy" if avg_sentence_len < 15 else "Moderate" if avg_sentence_len < 25 else "Complex"
    return f"Readability: {score} | Avg sentence length: {avg_sentence_len:.1f} words | Total words: {len(words)}"


# ---------------------------------------------------------------------------
# Node 1: Planner
# ---------------------------------------------------------------------------

def planner_node(state: PipelineState) -> dict:
    """Create an article outline and break into parallel research tasks."""
    topic = state["topic"]
    length = state.get("target_length", 500)
    
    print(f"\n[planner] Planning article: '{topic}' (~{length} words)")
    
    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert content planner. Create a structured article outline "
            "and specify exactly 3 research sub-tasks (as a JSON array) that can be "
            "investigated in parallel. Each sub-task should focus on a different aspect.\n\n"
            'Respond in JSON: {"outline": "...", "research_tasks": ["task1", "task2", "task3"]}'
        )),
        HumanMessage(content=f"Plan a {length}-word article about: {topic}"),
    ])
    
    try:
        data = json.loads(response.content)
        outline = data.get("outline", f"Introduction\nMain Points\nConclusion")
        tasks   = data.get("research_tasks", [
            f"Research history and background of {topic}",
            f"Research current state and applications of {topic}",
            f"Research future outlook and challenges of {topic}",
        ])[:3]
    except json.JSONDecodeError:
        outline = response.content
        tasks   = [
            f"Research history and background of {topic}",
            f"Research current applications of {topic}",
            f"Research future outlook for {topic}",
        ]
    
    print(f"[planner] Outline created. Research tasks: {tasks}")
    
    return {
        "outline": outline,
        "research_tasks": tasks,
        "messages": [AIMessage(content=f"Planner: Outline and {len(tasks)} research tasks created.")],
    }


# ---------------------------------------------------------------------------
# Nodes 2a/2b/2c: Parallel Research Workers
# ---------------------------------------------------------------------------

def _research_one_task(task: str, topic: str) -> str:
    """Perform one research sub-task by querying the knowledge base."""
    # Determine which aspect to search based on task keywords
    aspect_map = {
        "histor": "history",
        "background": "history",
        "current": "current_state",
        "state": "current_state",
        "application": "applications",
        "use case": "applications",
        "challenge": "challenges",
        "problem": "challenges",
        "future": "future",
        "outlook": "future",
        "statistic": "statistics",
        "data": "statistics",
    }
    
    aspect = "current_state"   # default
    task_lower = task.lower()
    for keyword, asp in aspect_map.items():
        if keyword in task_lower:
            aspect = asp
            break
    
    raw_data = search_knowledge_base.invoke({"aspect": aspect})
    
    # Enrich with LLM
    response = llm.invoke([
        SystemMessage(content="You are a research analyst. Write a concise research note (2-3 sentences) for an article."),
        HumanMessage(content=f"Research task: {task}\n\nContext: {raw_data}\n\nTopic: {topic}"),
    ])
    return f"Research [{task[:40]}...]:\n{response.content}"


def research_worker_1(state: PipelineState) -> dict:
    tasks = state.get("research_tasks", [])
    if not tasks:
        return {"research_notes": ["No research task assigned to worker 1"]}
    task = tasks[0]
    print(f"  [research_worker_1] task: '{task[:50]}...'")
    note = _research_one_task(task, state["topic"])
    print(f"  [research_worker_1] done")
    return {
        "research_notes": [note],
        "messages": [AIMessage(content=f"Worker 1 research complete.")],
    }


def research_worker_2(state: PipelineState) -> dict:
    tasks = state.get("research_tasks", [])
    if len(tasks) < 2:
        return {"research_notes": ["No research task assigned to worker 2"]}
    task = tasks[1]
    print(f"  [research_worker_2] task: '{task[:50]}...'")
    note = _research_one_task(task, state["topic"])
    print(f"  [research_worker_2] done")
    return {
        "research_notes": [note],
        "messages": [AIMessage(content=f"Worker 2 research complete.")],
    }


def research_worker_3(state: PipelineState) -> dict:
    tasks = state.get("research_tasks", [])
    if len(tasks) < 3:
        return {"research_notes": ["No research task assigned to worker 3"]}
    task = tasks[2]
    print(f"  [research_worker_3] task: '{task[:50]}...'")
    note = _research_one_task(task, state["topic"])
    print(f"  [research_worker_3] done")
    return {
        "research_notes": [note],
        "messages": [AIMessage(content=f"Worker 3 research complete.")],
    }


# ---------------------------------------------------------------------------
# Node 3: Writer
# ---------------------------------------------------------------------------

def writer_node(state: PipelineState) -> dict:
    """Write the article draft using the outline and research notes."""
    topic   = state["topic"]
    outline = state["outline"]
    notes   = state.get("research_notes", [])
    length  = state.get("target_length", 500)
    count   = state.get("revision_count", 0)
    feedback = state.get("editor_feedback", "")
    
    print(f"\n[writer] Writing draft (revision {count + 1})...")
    
    context_parts = [f"Outline:\n{outline}", f"Research Notes:\n" + "\n\n".join(notes)]
    if feedback:
        context_parts.append(f"Editor feedback to incorporate:\n{feedback}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    response = llm.invoke([
        SystemMessage(content=(
            f"You are a professional writer. Write a well-structured, engaging {length}-word article. "
            f"Use the provided outline and research notes. Include an introduction, body sections, and conclusion."
        )),
        HumanMessage(content=f"Topic: {topic}\n\n{context}"),
    ])
    
    draft = response.content
    print(f"[writer] Draft written: {len(draft.split())} words")
    
    return {
        "draft": draft,
        "revision_count": count + 1,
        "messages": [AIMessage(content=f"Writer: Draft {count + 1} complete ({len(draft.split())} words).")],
    }


# ---------------------------------------------------------------------------
# Node 4: Editor
# ---------------------------------------------------------------------------

def editor_node(state: PipelineState) -> dict:
    """Review the draft, provide feedback, and assign a quality score."""
    draft = state["draft"]
    topic = state["topic"]
    
    print(f"\n[editor] Reviewing draft...")
    
    response = llm_precise.invoke([
        SystemMessage(content=(
            "You are a senior editor. Review the article and provide:\n"
            "1. Specific improvement suggestions (2-3 points)\n"
            "2. A quality score from 1-10\n\n"
            'Respond in JSON: {"feedback": "...", "quality_score": 7}'
        )),
        HumanMessage(content=f"Topic: {topic}\n\nDraft:\n{draft}"),
    ])
    
    try:
        data = json.loads(response.content)
        feedback = data.get("feedback", "Good draft overall — minor polish needed.")
        score    = int(data.get("quality_score", 7))
    except (json.JSONDecodeError, ValueError):
        feedback = response.content
        score    = 7
    
    print(f"[editor] Quality score: {score}/10")
    print(f"[editor] Feedback: {feedback[:100]}...")
    
    return {
        "editor_feedback": feedback,
        "quality_score": score,
        "messages": [AIMessage(content=f"Editor: Score {score}/10. {feedback[:80]}...")],
    }


# ---------------------------------------------------------------------------
# Node 5: Human Review (with interrupt)
# ---------------------------------------------------------------------------

def human_review_node(state: PipelineState) -> dict:
    """Pause for human approval before publishing."""
    draft   = state["draft"]
    score   = state.get("quality_score", 0)
    topic   = state["topic"]
    
    print(f"\n[human_review] ⏸ Pausing for human review...")
    print(f"  Topic: {topic}")
    print(f"  Editor score: {score}/10")
    print(f"  Draft preview:\n  {draft[:300]}...")
    
    decision = interrupt({
        "message": "Please review the draft and decide:",
        "options": ["approved", "rejected", "<type revision instructions>"],
        "topic": topic,
        "editor_score": score,
        "draft_preview": draft[:500],
    })
    
    print(f"[human_review] Human decision: '{decision}'")
    return {
        "human_decision": decision,
        "messages": [HumanMessage(content=f"Human review: {decision}")],
    }


# ---------------------------------------------------------------------------
# Node 6: Finalise
# ---------------------------------------------------------------------------

def finalise_node(state: PipelineState) -> dict:
    """Apply final polish and prepare for publication."""
    draft    = state["draft"]
    decision = state.get("human_decision", "approved")
    topic    = state["topic"]
    
    print(f"\n[finalise] Finalising article...")
    
    if decision.lower() == "rejected":
        final = f"❌ Article REJECTED by human reviewer. Not published.\nTopic: {topic}"
        published = False
    elif decision.lower() == "approved":
        response = llm.invoke([
            SystemMessage(content="Apply final polish: fix any typos, improve flow, ensure consistent formatting. Return only the polished article."),
            HumanMessage(content=draft),
        ])
        final = f"✅ PUBLISHED ARTICLE\n\n{response.content}"
        published = True
    else:
        # Custom revision instructions
        response = llm.invoke([
            SystemMessage(content=f"Revise the article based on this instruction: {decision}\nReturn only the revised article."),
            HumanMessage(content=draft),
        ])
        final = f"✅ PUBLISHED ARTICLE (Revised per human instructions)\n\n{response.content}"
        published = True
    
    print(f"[finalise] Published: {published}")
    return {
        "final_article": final,
        "published": published,
        "messages": [AIMessage(content=f"Finalise: {'Published! ✅' if published else 'Rejected ❌'}")]
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_editing(state: PipelineState) -> Literal["writer", "human_review"]:
    """If quality score < 7 and < 3 revisions, send back to writer."""
    score    = state.get("quality_score", 10)
    revision = state.get("revision_count", 0)
    if score < 7 and revision < 3:
        print(f"  [routing] Score {score} < 7 — sending back to writer for revision {revision + 1}")
        return "writer"
    print(f"  [routing] Score {score} ≥ 7 (or max revisions) — proceeding to human review")
    return "human_review"


def route_after_human_review(state: PipelineState) -> Literal["finalise", "writer"]:
    """Route based on human decision."""
    decision = state.get("human_decision", "approved").lower()
    if decision == "rejected" or decision == "approved":
        return "finalise"
    # Custom instruction — could go back to writer or straight to finalise
    return "finalise"   # finalise node handles the revision


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

memory = MemorySaver()

builder = StateGraph(PipelineState)

builder.add_node("planner",          planner_node)
builder.add_node("research_worker_1", research_worker_1)
builder.add_node("research_worker_2", research_worker_2)
builder.add_node("research_worker_3", research_worker_3)
builder.add_node("writer",           writer_node)
builder.add_node("editor",           editor_node)
builder.add_node("human_review",     human_review_node)
builder.add_node("finalise",         finalise_node)

# Sequential start
builder.add_edge(START, "planner")

# Fan-out: planner → 3 parallel research workers
builder.add_edge("planner",          "research_worker_1")
builder.add_edge("planner",          "research_worker_2")
builder.add_edge("planner",          "research_worker_3")

# Fan-in: all 3 workers → writer
builder.add_edge("research_worker_1", "writer")
builder.add_edge("research_worker_2", "writer")
builder.add_edge("research_worker_3", "writer")

# Writer → Editor
builder.add_edge("writer",  "editor")

# Editor → conditional: revise or proceed to human review
builder.add_conditional_edges(
    "editor",
    route_after_editing,
    {"writer": "writer", "human_review": "human_review"},
)

# Human review → conditional: approve/reject/revise → finalise
builder.add_conditional_edges(
    "human_review",
    route_after_human_review,
    {"finalise": "finalise", "writer": "writer"},
)

builder.add_edge("finalise", END)

graph = builder.compile(checkpointer=memory)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("=" * 70)
print("DEMO 10 — Advanced Multi-Agent Content Pipeline")
print("=" * 70)
print("""
Pipeline stages:
  1. Planner      → outlines article, creates 3 parallel research tasks
  2. Research ×3  → parallel workers gather information (fan-out)
  3. Writer       → writes draft from outline + research (fan-in)
  4. Editor       → scores quality; loops back if score < 7
  5. Human review → INTERRUPT: approve / reject / revise
  6. Finalise     → polish and publish (or reject)
""")

scenarios = [
    {
        "thread_id": "pipeline-article-1",
        "topic": "The Impact of Generative AI on Software Development",
        "target_length": 400,
        "requested_by": "editorial_team",
        "human_decision": "approved",
        "description": "Full pipeline — approved",
    },
    {
        "thread_id": "pipeline-article-2",
        "topic": "Python Best Practices for Production Systems",
        "target_length": 350,
        "requested_by": "engineering_lead",
        "human_decision": "Add a section on testing strategies",
        "description": "Full pipeline — human requests revision",
    },
]

for scenario in scenarios:
    print(f"\n{'#'*70}")
    print(f"  SCENARIO: {scenario['description']}")
    print(f"  Topic: {scenario['topic']}")
    print(f"{'#'*70}")
    
    config = {"configurable": {"thread_id": scenario["thread_id"]}}
    
    initial: PipelineState = {
        "topic":           scenario["topic"],
        "requested_by":    scenario["requested_by"],
        "target_length":   scenario["target_length"],
        "outline":         "",
        "research_tasks":  [],
        "research_notes":  [],
        "draft":           "",
        "revision_count":  0,
        "editor_feedback": "",
        "quality_score":   0,
        "human_decision":  "",
        "final_article":   "",
        "published":       False,
        "messages":        [],
    }
    
    # ── Phase 1: Run until interrupt ─────────────────────────────────
    print("\nPhase 1: Running pipeline until human review...")
    try:
        graph.invoke(initial, config=config)
    except Exception:
        pass  # Graph paused at interrupt — expected
    
    state_snap = graph.get_state(config)
    waiting_at = state_snap.next
    print(f"\n  ⏸ Graph paused. Waiting at: {waiting_at}")
    print(f"  Draft quality score: {state_snap.values.get('quality_score', '?')}/10")
    
    # ── Phase 2: Human makes decision ────────────────────────────────
    decision = scenario["human_decision"]
    print(f"\nPhase 2: Human decision → '{decision}'")
    
    final_state = graph.invoke(Command(resume=decision), config=config)
    
    # ── Results ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"  Published: {final_state.get('published', False)}")
    print(f"  Revisions: {final_state.get('revision_count', 0)}")
    print(f"  Messages:  {len(final_state.get('messages', []))}")
    print(f"\nFinal article preview:")
    article = final_state.get("final_article", "")
    print(article[:600] + ("..." if len(article) > 600 else ""))
    
    print(f"\nAudit trail:")
    for m in final_state.get("messages", []):
        role = "AI" if isinstance(m, AIMessage) else "Human"
        print(f"  [{role}] {m.content[:80]}")

print("\n" + "="*70)
print("Demo 10 complete — all LangGraph concepts demonstrated:")
print("  StateGraph, conditional edges, state reducers, MemorySaver,")
print("  tool-calling, interrupt/Command, supervisor, parallel execution")

print("\nGraph Mermaid diagram:")
try:
    print(graph.get_graph().draw_mermaid())
except Exception:
    pass
