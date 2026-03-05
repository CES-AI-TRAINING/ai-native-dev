"""
Demo 05: Tool-Calling Agent with LangGraph

LangGraph's built-in ReAct agent loop: the LLM decides whether to call a
tool or produce a final answer; a ToolNode executes the call; the result
is fed back to the LLM; repeat until done.

Topics covered:
1. @tool decorator — define custom tools
2. bind_tools — attach tools to an LLM
3. ToolNode — automatic tool execution node
4. tools_condition — built-in routing function (tools vs END)
5. The classic ReAct loop: agent → tools → agent → … → END

Graph shape:
           ┌───────────────────────────┐
           │                           ▼
START → agent ──(tools_condition)──► tool_node
           │                             │
           ▼(END)                        │
          END ◄──────────────────────────┘
"""

import os
import math
import json
from typing import Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set.")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.
    
    Args:
        expression: A Python math expression string, e.g. '2 ** 10' or 'math.sqrt(144)'
    """
    try:
        # Provide math module in the evaluation context
        result = eval(expression, {"__builtins__": {}, "math": math})
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city (mock data for demo).
    
    Args:
        city: Name of the city
    """
    mock_weather = {
        "london": "12°C, cloudy with light rain",
        "new york": "18°C, partly sunny",
        "tokyo": "25°C, clear skies",
        "paris": "15°C, overcast",
        "sydney": "22°C, sunny",
    }
    weather = mock_weather.get(city.lower(), f"Weather data unavailable for '{city}'")
    return f"Weather in {city}: {weather}"


@tool
def word_count(text: str) -> str:
    """Count the number of words and characters in a text.
    
    Args:
        text: Input text to analyse
    """
    words = len(text.split())
    chars = len(text)
    return json.dumps({"words": words, "characters": chars})


tools = [calculator, get_weather, word_count]

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

llm_with_tools = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
    temperature=0,
).bind_tools(tools)


def agent_node(state: State) -> dict:
    """LLM decides: call a tool or produce final answer."""
    print(f"  [agent] thinking... (messages so far: {len(state['messages'])})")
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        print(f"  [agent] wants to call: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print(f"  [agent] final answer ready")
    return {"messages": [response]}


# ToolNode automatically calls the right tool and appends ToolMessages
tool_node = ToolNode(tools)

# ---------------------------------------------------------------------------
# Build ReAct graph
# ---------------------------------------------------------------------------

builder = StateGraph(State)

builder.add_node("agent",  agent_node)
builder.add_node("tools",  tool_node)

builder.add_edge(START, "agent")

# tools_condition: if last message has tool_calls → "tools", else END
builder.add_conditional_edges("agent", tools_condition)

# After tools run, always go back to agent
builder.add_edge("tools", "agent")

graph = builder.compile()

# ---------------------------------------------------------------------------
# Run demos
# ---------------------------------------------------------------------------

print("=" * 60)
print("DEMO 05 — Tool-Calling Agent")
print("=" * 60)
print()

queries = [
    "What is 2 raised to the power of 16? Also, what is the square root of 1764?",
    "How is the weather in Tokyo and London today?",
    "Count the words and characters in: 'The quick brown fox jumps over the lazy dog'",
    "If I start with 1000 and double it 8 times, what do I end up with? Show each step.",
]

for query in queries:
    print(f"Query: {query}")
    print("-" * 50)
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    final = result["messages"][-1].content
    print(f"Answer: {final}")
    print()
    print("=" * 60)
    print()

print(f"Total tools available: {[t.name for t in tools]}")
print()
try:
    print("Graph structure:")
    print(graph.get_graph().draw_mermaid())
except Exception:
    pass
