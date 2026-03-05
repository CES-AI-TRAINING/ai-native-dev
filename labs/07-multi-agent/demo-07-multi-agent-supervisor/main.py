"""
Demo 07: Multi-Agent Supervisor Pattern

The supervisor pattern is the most common multi-agent architecture:
  - A **supervisor** LLM receives a task and decides which specialist agent to call.
  - After each agent finishes, control returns to the supervisor.
  - The supervisor either delegates to another agent or declares the task complete.

Agents in this demo:
  - researcher  — performs web-style research (mock), summarises findings
  - coder       — writes and explains code
  - supervisor  — orchestrates and synthesises final answer

Topics covered:
1. create_react_agent — quick way to build a tool-equipped sub-agent
2. Supervisor LLM with structured routing output
3. Adding sub-graphs / calling agents as nodes
4. Termination condition: supervisor returns "FINISH"

Graph shape:
                  ┌──────────────┐
                  │              │
START → supervisor ─► researcher ─┤
                  └──► coder ────┘
                  └──► FINISH → END
"""

import os
import json
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set.")

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0)

# ---------------------------------------------------------------------------
# Specialist tools
# ---------------------------------------------------------------------------

@tool
def web_search(query: str) -> str:
    """Search the web for information on a topic (mock).
    
    Args:
        query: The search query
    """
    mock_results = {
        "python": "Python is a high-level, general-purpose programming language known for its readability. Created by Guido van Rossum in 1991.",
        "langchain": "LangChain is a framework for developing applications powered by LLMs. It provides abstractions for chains, agents, and memory.",
        "langgraph": "LangGraph is a library built on top of LangChain for building stateful, multi-actor applications using a graph-based approach.",
        "openai": "OpenAI is an AI safety company that developed GPT-4, DALL-E, and Whisper. It offers APIs for language, vision, and speech models.",
        "rag": "Retrieval-Augmented Generation (RAG) combines retrieval of relevant documents with LLM generation for more accurate, grounded responses.",
    }
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value
    return f"Search results for '{query}': No specific mock data available. This is a demo simulation."


@tool
def summarise_findings(text: str) -> str:
    """Summarise a block of research text into bullet points.
    
    Args:
        text: The research text to summarise
    """
    lines = text.split(". ")
    bullets = "\n".join(f"• {line.strip()}" for line in lines if line.strip())
    return f"Summary:\n{bullets}"


@tool
def write_code(task_description: str, language: str = "python") -> str:
    """Write a code snippet for a given task description.
    
    Args:
        task_description: What the code should do
        language: Programming language (default: python)
    """
    # Mock code generation for common tasks
    templates = {
        "hello world": f'# Hello World in {language}\nprint("Hello, World!")',
        "fibonacci": (
            "def fibonacci(n: int) -> list[int]:\n"
            "    a, b = 0, 1\n"
            "    result = []\n"
            "    for _ in range(n):\n"
            "        result.append(a)\n"
            "        a, b = b, a + b\n"
            "    return result\n\n"
            "print(fibonacci(10))"
        ),
        "api": (
            "from fastapi import FastAPI\n\n"
            "app = FastAPI()\n\n"
            "@app.get('/hello')\n"
            "def hello():\n"
            "    return {'message': 'Hello, World!'}"
        ),
    }
    for key, code in templates.items():
        if key.lower() in task_description.lower():
            return f"```{language}\n{code}\n```"
    return f"```{language}\n# Code for: {task_description}\n# [Implementation would go here]\nprint('Task complete')\n```"


@tool
def explain_code(code: str) -> str:
    """Provide a plain-English explanation of a code snippet.
    
    Args:
        code: The code snippet to explain
    """
    return (
        f"Code explanation:\n"
        f"The provided code snippet demonstrates {code[:50]}... "
        f"It uses standard constructs to implement the required functionality. "
        f"Key points: proper error handling should be added for production use."
    )


# ---------------------------------------------------------------------------
# Build specialist agents using create_react_agent
# ---------------------------------------------------------------------------

researcher_agent = create_react_agent(
    model=llm,
    tools=[web_search, summarise_findings],
    state_modifier=(
        "You are a Research Specialist. Your job is to search for information "
        "and provide well-organised, factual summaries. Always cite your search results."
    ),
)

coder_agent = create_react_agent(
    model=llm,
    tools=[write_code, explain_code],
    state_modifier=(
        "You are a Code Specialist. Your job is to write clean, well-commented code "
        "and explain technical concepts clearly. Always provide working examples."
    ),
)

# ---------------------------------------------------------------------------
# Supervisor state & routing
# ---------------------------------------------------------------------------

AGENTS = ["researcher", "coder"]

class SupervisorState(TypedDict):
    messages:    Annotated[list[BaseMessage], add_messages]
    task:        str
    next_agent:  str   # "researcher" | "coder" | "FINISH"
    agent_results: Annotated[list[str], lambda a, b: a + b]  # accumulated results


def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor decides which agent to dispatch next (or finishes)."""
    task = state["task"]
    results_so_far = state.get("agent_results", [])
    
    system_prompt = (
        "You are a task supervisor managing a team of specialist agents.\n"
        "Available agents:\n"
        "- researcher: for searching information, facts, definitions, explanations\n"
        "- coder: for writing code, debugging, or explaining code\n"
        "- FINISH: when you have enough information to answer the task completely\n\n"
        "Respond ONLY with a JSON object: {\"next\": \"researcher\" | \"coder\" | \"FINISH\", "
        "\"reasoning\": \"brief reason\"}\n\n"
        f"Task: {task}\n"
        f"Results collected so far ({len(results_so_far)}):\n" +
        ("\n".join(f"  [{i+1}] {r[:150]}..." if len(r) > 150 else f"  [{i+1}] {r}" 
                   for i, r in enumerate(results_so_far)) if results_so_far else "  (none yet)")
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="What is your next routing decision?"),
    ])
    
    try:
        decision = json.loads(response.content)
        next_agent = decision.get("next", "FINISH")
        reasoning  = decision.get("reasoning", "")
    except json.JSONDecodeError:
        # Fallback: parse text response
        content = response.content.lower()
        if "researcher" in content:
            next_agent, reasoning = "researcher", "Contains 'researcher'"
        elif "coder" in content:
            next_agent, reasoning = "coder", "Contains 'coder'"
        else:
            next_agent, reasoning = "FINISH", "Defaulting to finish"
    
    print(f"  [supervisor] → {next_agent}  ({reasoning})")
    return {
        "next_agent": next_agent,
        "messages": [AIMessage(content=f"Supervisor routing to: {next_agent}. Reason: {reasoning}")],
    }


def researcher_node(state: SupervisorState) -> dict:
    """Invoke the researcher agent."""
    task = state["task"]
    print(f"  [researcher] working on task...")
    result = researcher_agent.invoke({"messages": [HumanMessage(content=task)]})
    last_msg = result["messages"][-1].content
    print(f"  [researcher] done. Result snippet: {last_msg[:100]}...")
    return {
        "agent_results": [f"Researcher finding: {last_msg}"],
        "messages": [AIMessage(content=f"[Researcher]: {last_msg}")],
    }


def coder_node(state: SupervisorState) -> dict:
    """Invoke the coder agent."""
    task = state["task"]
    print(f"  [coder] working on task...")
    result = coder_agent.invoke({"messages": [HumanMessage(content=task)]})
    last_msg = result["messages"][-1].content
    print(f"  [coder] done. Result snippet: {last_msg[:100]}...")
    return {
        "agent_results": [f"Code output: {last_msg}"],
        "messages": [AIMessage(content=f"[Coder]: {last_msg}")],
    }


def synthesise_node(state: SupervisorState) -> dict:
    """Produce the final synthesised answer from all agent results."""
    task = state["task"]
    results = state.get("agent_results", [])
    
    combined = "\n\n".join(results) if results else "No results collected."
    
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant. Synthesise the research findings into a clear, comprehensive final answer."),
        HumanMessage(content=f"Task: {task}\n\nAgent findings:\n{combined}\n\nProvide the final answer."),
    ])
    
    final = response.content
    print(f"  [synthesise] Final answer: {final[:120]}...")
    return {"messages": [AIMessage(content=f"Final Answer:\n{final}")]}


def route_supervisor(state: SupervisorState) -> str:
    """Routing function for supervisor's conditional edge."""
    return state.get("next_agent", "FINISH")


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

builder = StateGraph(SupervisorState)

builder.add_node("supervisor",  supervisor_node)
builder.add_node("researcher",  researcher_node)
builder.add_node("coder",       coder_node)
builder.add_node("synthesise",  synthesise_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "researcher": "researcher",
        "coder":      "coder",
        "FINISH":     "synthesise",
    },
)

# After each agent, return to supervisor
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder",      "supervisor")
builder.add_edge("synthesise", END)

graph = builder.compile()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("=" * 60)
print("DEMO 07 — Multi-Agent Supervisor")
print("=" * 60)
print()

tasks = [
    "Explain what LangGraph is and show me a simple Python hello-world example using it.",
    "Research what RAG is, then write a simple Python function that demonstrates the concept.",
]

for i, task in enumerate(tasks, 1):
    print(f"Task {i}: {task}")
    print("-" * 60)
    
    initial: SupervisorState = {
        "messages": [HumanMessage(content=task)],
        "task": task,
        "next_agent": "",
        "agent_results": [],
    }
    result = graph.invoke(initial)
    
    print(f"\nFinal message:")
    print(result["messages"][-1].content[:500])
    print()
    print("=" * 60)
    print()
