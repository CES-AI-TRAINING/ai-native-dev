"""
Demo 04: LangChain Auto-Tracing with LangSmith

When LANGSMITH_TRACING=true is set in the environment, LangChain
automatically sends traces for every component in your chain — no
decorators or wrappers needed. This demo shows you exactly what
gets traced and how to read the resulting run trees.

Topics covered:
1. LLM invocation tracing (ChatOpenAI)
2. Prompt template + LLM chain tracing (LCEL pipe syntax)
3. Multi-step sequential chain with automatic nesting
4. Tool-calling agent with step-level traces
5. Streaming responses with trace capture
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langsmith import traceable

# ============================================================================
# SETUP
# ============================================================================

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set. See .env.example.")

# LANGSMITH_TRACING=true in .env enables automatic tracing for all LangChain
# components — no additional code changes required.
tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
if tracing_enabled:
    print("✅ LangSmith tracing is ENABLED")
else:
    print("⚠️  LangSmith tracing is DISABLED — set LANGSMITH_TRACING=true in .env")
print()

MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model=MODEL, temperature=0.3)

print("=" * 70)
print("DEMO 04: LANGCHAIN AUTO-TRACING WITH LANGSMITH")
print("=" * 70)
print()

# ============================================================================
# DEMO 1: Direct LLM invocation
# ============================================================================

print("── DEMO 1: Direct LLM invocation ────────────────────────────────────")
print()
print("   A simple ChatOpenAI.invoke() call is automatically traced.")
print()

messages = [
    SystemMessage(content="You are a concise assistant. Answer in 1-2 sentences."),
    HumanMessage(content="What is LangSmith used for?"),
]
response = llm.invoke(messages)
print(f"   Response: {response.content}")
print()

# ============================================================================
# DEMO 2: Prompt template + LLM chain (LCEL)
# ============================================================================

print("── DEMO 2: LCEL chain ───────────────────────────────────────────────")
print()
print("   LangSmith captures each step in the pipe: Prompt → LLM → Parser")
print()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer in 1-2 sentences."),
        ("human", "{question}"),
    ]
)

# LCEL pipe: prompt | llm | parser
chain = prompt | llm | StrOutputParser()

# Each | symbol creates a child run in LangSmith:
#   RunnableSequence        ← parent
#     ├── ChatPromptTemplate
#     ├── ChatOpenAI
#     └── StrOutputParser

questions = [
    "What is a span in distributed tracing?",
    "How does LangSmith differ from Prometheus?",
]

for q in questions:
    answer = chain.invoke({"question": q})
    print(f"   Q: {q}")
    print(f"   A: {answer}")
    print()

# ============================================================================
# DEMO 3: Multi-step chain with intermediate outputs
# ============================================================================

print("── DEMO 3: Multi-step sequential chain ─────────────────────────────")
print()
print("   Step 1: extract keywords  →  Step 2: write description")
print()

# Step 1: extract keywords from a topic
keyword_prompt = ChatPromptTemplate.from_template(
    "List 3 key technical terms related to: {topic}. "
    "Format: term1, term2, term3. No explanations."
)
keyword_chain = keyword_prompt | llm | StrOutputParser()

# Step 2: write a sentence using the keywords
description_prompt = ChatPromptTemplate.from_template(
    "Write one sentence that naturally includes these terms: {keywords}"
)
description_chain = description_prompt | llm | StrOutputParser()

# Compose the two steps with a lambda to pass the output of step 1 into step 2
full_chain = (
    keyword_chain
    | (lambda keywords: {"keywords": keywords})
    | description_chain
)

topic = "observability in distributed AI systems"
result = full_chain.invoke({"topic": topic})
print(f"   Topic: {topic}")
print(f"   Description: {result}")
print()

# ============================================================================
# DEMO 4: Agent with tools — step-level traces
# ============================================================================

print("── DEMO 4: Agent with tools ─────────────────────────────────────────")
print()
print("   Each agent thought, tool call, and observation gets its own run.")
print()


@tool
def get_word_count(text: str) -> int:
    """Return the number of words in the given text."""
    count = len(text.split())
    print(f"   [Tool] get_word_count({repr(text)[:40]}…) = {count}")
    return count


@tool
def reverse_text(text: str) -> str:
    """Return the text reversed word by word."""
    reversed_text = " ".join(text.split()[::-1])
    print(f"   [Tool] reverse_text → {reversed_text[:50]}")
    return reversed_text


tools = [get_word_count, reverse_text]
agent_llm = ChatOpenAI(model=MODEL, temperature=0)

agent = create_agent(
    model=agent_llm,
    tools=tools,
    system_prompt="You are a helpful text processing assistant.",
)

task = (
    "Count the words in 'LangSmith makes AI observability effortless', "
    "then reverse those words."
)
agent_result = agent.invoke({"messages": [("human", task)]})
output = agent_result["messages"][-1].content
print(f"   Task: {task}")
print(f"   Result: {output}")
print()

# ============================================================================
# DEMO 5: @traceable wrapping a LangChain chain
# ============================================================================

print("── DEMO 5: @traceable + LangChain chain ─────────────────────────────")
print()
print("   Wrap a chain in @traceable to add a top-level business-logic run.")
print()

translate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following text to {language}. Reply with only the translation."),
        ("human", "{text}"),
    ]
)
translate_chain = translate_prompt | llm | StrOutputParser()


@traceable(
    name="full_translation_pipeline",
    tags=["translation", "demo-04"],
    metadata={"pipeline_version": "1.0"},
)
def translate_pipeline(text: str, language: str) -> dict:
    """
    Full translation pipeline with pre/post processing.

    The @traceable parent run records the inputs + final dict output.
    Inside it, the LCEL chain appears as nested child runs.
    """
    translated = translate_chain.invoke({"text": text, "language": language})
    return {
        "original": text,
        "language": language,
        "translation": translated,
        "word_count": len(translated.split()),
    }


result = translate_pipeline("Observability is the foundation of production AI.", "Spanish")
print(f"   Original: {result['original']}")
print(f"   Spanish: {result['translation']}")
print()

print("=" * 70)
print("DONE — in LangSmith you will see:")
print("  • Each chain component as a separate run")
print("  • Full input/output at every step")
print("  • Token usage and latency per LLM call")
print("  URL: https://smith.langchain.com")
print("=" * 70)
