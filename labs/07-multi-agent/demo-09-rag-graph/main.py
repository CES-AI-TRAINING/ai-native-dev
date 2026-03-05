"""
Demo 09: Corrective RAG Pipeline as a LangGraph

Retrieval-Augmented Generation (RAG) implemented as a graph enables
us to add *loops* and *decision points* that plain chain-based RAG cannot.
This demo implements Corrective RAG (CRAG): if retrieved documents are not
relevant enough, the pipeline routes to a fallback web search before generating.

Topics covered:
1. RAG pipeline as a directed graph (retrieve → grade → generate)
2. Cyclic graphs — looping back when documents are insufficient
3. Grade node as a conditional routing decision
4. Hallucination check before returning the final answer
5. In-memory Chroma vector store for demo simplicity

Graph shape:
                    ┌──────────────────────────────────────────────┐
                    │                                              │
START → retrieve → grade_docs ──(irrelevant)──► web_fallback ───► generate → hallucination_check
                        │                                              │
                        └──(relevant)──────────────────────────────► │
                                                                      ▼
                                                                 (grounded) → END
                                                                 (hallucinating) → generate (retry)
"""

import os
import operator
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, START, END

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY is not set.")

llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ---------------------------------------------------------------------------
# In-memory vector store (seeded with demo documents)
# ---------------------------------------------------------------------------

DEMO_DOCS = [
    Document(
        page_content=(
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
            "It extends LangChain with the ability to create cyclic graphs — essential for agent loops. "
            "Key primitives: StateGraph, nodes, edges, conditional edges, and checkpointers."
        ),
        metadata={"source": "langgraph-docs", "topic": "langgraph"},
    ),
    Document(
        page_content=(
            "Retrieval-Augmented Generation (RAG) improves LLM accuracy by retrieving relevant documents "
            "before generation. The standard pipeline: embed query → search vector store → inject context → generate. "
            "RAG reduces hallucinations by grounding answers in retrieved evidence."
        ),
        metadata={"source": "rag-guide", "topic": "rag"},
    ),
    Document(
        page_content=(
            "LangChain is an open-source framework for building LLM-powered applications. "
            "It provides abstractions for prompts, chains, agents, memory, and tools. "
            "LangChain supports many LLM providers including OpenAI, Anthropic, and Google."
        ),
        metadata={"source": "langchain-docs", "topic": "langchain"},
    ),
    Document(
        page_content=(
            "Corrective RAG (CRAG) adds a grading step after retrieval. "
            "If retrieved documents are not relevant to the query, CRAG triggers a web search fallback. "
            "After generation, a hallucination checker verifies the answer is grounded in context."
        ),
        metadata={"source": "crag-paper", "topic": "rag"},
    ),
    Document(
        page_content=(
            "Vector databases store document embeddings for semantic search. "
            "Popular choices: Chroma (local), Pinecone (cloud), Qdrant, Weaviate, and pgvector. "
            "Similarity search returns documents with embeddings closest to the query embedding."
        ),
        metadata={"source": "vectordb-guide", "topic": "vectordb"},
    ),
]

print("Initialising in-memory vector store...")
vectorstore = Chroma.from_documents(
    documents=DEMO_DOCS,
    embedding=embeddings,
    collection_name="demo09-rag",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
print(f"Vector store ready with {len(DEMO_DOCS)} documents.")
print()

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question:       str
    documents:      list[Document]          # retrieved or fallback documents
    relevance:      str                     # "relevant" | "irrelevant"
    generation:     str                     # LLM generated answer
    hallucination:  str                     # "grounded" | "hallucinating"
    retry_count:    int                     # guard against infinite loops


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve_node(state: RAGState) -> dict:
    """Retrieve documents from the vector store."""
    question = state["question"]
    print(f"[retrieve] question: '{question}'")
    docs = retriever.invoke(question)
    print(f"[retrieve] found {len(docs)} documents")
    for i, d in enumerate(docs):
        print(f"  [{i+1}] ({d.metadata.get('source', '?')}) {d.page_content[:80]}...")
    return {"documents": docs}


def grade_documents_node(state: RAGState) -> dict:
    """Grade each retrieved document for relevance to the question."""
    question  = state["question"]
    documents = state["documents"]
    
    print(f"[grade_docs] grading {len(documents)} documents...")
    
    relevant_docs = []
    for doc in documents:
        response = llm.invoke([
            SystemMessage(content=(
                "You are a relevance grader. Given a question and a document, "
                "decide if the document is relevant to answer the question. "
                "Respond with exactly one word: 'yes' or 'no'."
            )),
            HumanMessage(content=f"Question: {question}\n\nDocument: {doc.page_content}"),
        ])
        grade = response.content.strip().lower()
        print(f"  → Grade: {grade} | {doc.page_content[:60]}...")
        if grade == "yes":
            relevant_docs.append(doc)
    
    relevance = "relevant" if relevant_docs else "irrelevant"
    print(f"[grade_docs] relevance decision: {relevance} ({len(relevant_docs)}/{len(documents)} relevant)")
    return {"documents": relevant_docs, "relevance": relevance}


def web_fallback_node(state: RAGState) -> dict:
    """Simulate a web search when retrieval is insufficient."""
    question = state["question"]
    print(f"[web_fallback] retrieval was insufficient — simulating web search for: '{question}'")
    
    # In production, integrate with a real search API (Tavily, Serper, etc.)
    fallback_content = (
        f"Web search results for '{question}':\n"
        f"Based on recent online sources, here is relevant information. "
        f"This is a simulated web search result providing additional context "
        f"about {question}. In a real deployment, this would contain actual "
        f"web search results from Tavily or a similar API."
    )
    fallback_doc = Document(
        page_content=fallback_content,
        metadata={"source": "web_search", "query": question},
    )
    print(f"[web_fallback] retrieved 1 fallback document from web")
    return {"documents": [fallback_doc], "relevance": "relevant"}


def generate_node(state: RAGState) -> dict:
    """Generate an answer using the (graded) documents as context."""
    question  = state["question"]
    documents = state["documents"]
    retry     = state.get("retry_count", 0)
    
    context = "\n\n---\n\n".join(d.page_content for d in documents)
    
    print(f"[generate] generating answer (attempt {retry + 1})...")
    
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "If the context does not contain enough information, say so clearly. "
            "Do not add information not present in the context."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ])
    
    answer = response.content
    print(f"[generate] answer: {answer[:100]}...")
    return {"generation": answer, "retry_count": state.get("retry_count", 0) + 1}


def hallucination_check_node(state: RAGState) -> dict:
    """Check whether the generated answer is grounded in the documents."""
    generation = state["generation"]
    documents  = state["documents"]
    context    = "\n\n".join(d.page_content for d in documents)
    
    print(f"[hallucination_check] verifying answer is grounded...")
    
    response = llm.invoke([
        SystemMessage(content=(
            "You are a fact-checker. Given a context and a generated answer, "
            "determine if the answer is grounded in the context (no made-up facts). "
            "Respond with exactly one word: 'grounded' or 'hallucinating'."
        )),
        HumanMessage(content=f"Context:\n{context}\n\nAnswer:\n{generation}"),
    ])
    
    verdict = response.content.strip().lower()
    if "ground" in verdict:
        verdict = "grounded"
    elif "hallucin" in verdict:
        verdict = "hallucinating"
    else:
        verdict = "grounded"   # default to grounded if unclear
    
    print(f"[hallucination_check] verdict: {verdict}")
    return {"hallucination": verdict}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_grading(state: RAGState) -> Literal["web_fallback", "generate"]:
    """Route to web fallback if no relevant docs, otherwise generate."""
    return "web_fallback" if state["relevance"] == "irrelevant" else "generate"


def route_after_hallucination_check(state: RAGState) -> Literal["generate", END]:
    """Retry generation if hallucinating (max 2 retries), else return answer."""
    if state["hallucination"] == "grounded":
        return END
    if state.get("retry_count", 0) >= 2:
        print("[hallucination_check] max retries reached — returning best effort answer")
        return END
    print("[hallucination_check] hallucination detected — retrying generation")
    return "generate"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

builder = StateGraph(RAGState)

builder.add_node("retrieve",            retrieve_node)
builder.add_node("grade_docs",          grade_documents_node)
builder.add_node("web_fallback",        web_fallback_node)
builder.add_node("generate",            generate_node)
builder.add_node("hallucination_check", hallucination_check_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade_docs")

builder.add_conditional_edges(
    "grade_docs",
    route_after_grading,
    {"web_fallback": "web_fallback", "generate": "generate"},
)

builder.add_edge("web_fallback", "generate")
builder.add_edge("generate",     "hallucination_check")

builder.add_conditional_edges(
    "hallucination_check",
    route_after_hallucination_check,
    {"generate": "generate", END: END},
)

graph = builder.compile()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

print("=" * 60)
print("DEMO 09 — Corrective RAG Pipeline as a LangGraph")
print("=" * 60)
print()

questions = [
    "What is LangGraph and what are its key primitives?",
    "What is Corrective RAG and how does it differ from standard RAG?",
    "What is the capital of France?",   # Off-topic — should trigger web fallback
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print("=" * 60)
    
    initial: RAGState = {
        "question":      question,
        "documents":     [],
        "relevance":     "",
        "generation":    "",
        "hallucination": "",
        "retry_count":   0,
    }
    
    result = graph.invoke(initial)
    
    print(f"\n✅ FINAL ANSWER:\n{result['generation']}")
    print(f"\n   Relevance: {result['relevance']} | Hallucination check: {result['hallucination']}")

print("\n" + "="*60)
print("RAG Graph structure:")
try:
    print(graph.get_graph().draw_mermaid())
except Exception:
    pass
