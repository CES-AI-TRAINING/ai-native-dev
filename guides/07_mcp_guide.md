# Model Context Protocol (MCP) - Comprehensive Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Why Context Management Becomes Critical at Scale](#why-context-management-becomes-critical-at-scale)
3. [Limitations of Prompt-Only Context Sharing](#limitations-of-prompt-only-context-sharing)
4. [Introduction to Model Context Protocol (MCP)](#introduction-to-model-context-protocol-mcp)
5. [MCP Architecture and Key Concepts](#mcp-architecture-and-key-concepts)
6. [MCP vs RAG vs Fine-tuning](#mcp-vs-rag-vs-fine-tuning)
7. [Standardising Tool Calling with MCP](#standardising-tool-calling-with-mcp)
8. [Transport Protocols: Stdio vs SSE vs Streamable HTTP](#transport-protocols-stdio-vs-sse-vs-streamable-http)
9. [Integrating MCP with Agents](#integrating-mcp-with-agents)
10. [Integration MCP with LangChain](#integration-mcp-with-langchain)
11. [How to Build Custom MCP Servers](#how-to-build-custom-mcp-servers)
12. [Best Practices and Use Cases](#best-practices-and-use-cases)

---

## Introduction

Model Context Protocol (MCP) is an open protocol developed by Anthropic that standardizes how applications provide context to Large Language Models (LLMs). As AI applications grow in complexity and scale, managing context efficiently becomes crucial for building robust, production-ready systems.

This guide explores MCP's role in modern AI development, its architecture, and how to integrate it with existing tools and frameworks.

---

## Why Context Management Becomes Critical at Scale

### The Scale Challenge

As AI applications mature and handle more complex tasks, several context management challenges emerge:

#### 1. **Context Window Limitations**

- Modern LLMs have finite context windows (typically 4K-200K tokens)
- At scale, applications need to process documents, codebases, and datasets far exceeding these limits
- Poor context management leads to information loss and degraded performance

#### 2. **Cost Implications**

- Every token sent to the model incurs API costs
- Redundant context in repeated requests multiplies expenses
- Inefficient context sharing can increase costs by 10-100x

#### 3. **Latency Issues**

- Larger context windows require more processing time
- Network overhead for transmitting large prompts
- User experience suffers with slow response times

#### 4. **Data Freshness**

- Applications need access to real-time data (APIs, databases, file systems)
- Embedding all possible context upfront is impractical
- Dynamic context retrieval becomes necessary

#### 5. **Security and Privacy**

- Not all context should be sent to external models
- Need for fine-grained access control
- Compliance requirements (GDPR, HIPAA) demand careful context handling

#### 6. **Multi-Source Integration**

- Modern applications integrate multiple data sources
- Each source has different access patterns and formats
- Standardized context management prevents integration chaos

### Example Scenario

Consider a customer support AI that needs to:

- Access customer order history
- Check inventory systems
- Review previous support tickets
- Apply company policies
- Generate personalized responses

Without proper context management:

- Each request might send the entire customer history
- API calls multiply unnecessarily
- Response times increase
- Costs spiral out of control

---

## Limitations of Prompt-Only Context Sharing

### Traditional Approach

The conventional method of sharing context involves embedding everything directly into prompts:

```python
# Traditional prompt-only approach
prompt = f"""
You are a customer support agent.

Customer Information:
{customer_data}

Order History:
{order_history}

Previous Tickets:
{previous_tickets}

Company Policies:
{policies}

Current Question: {user_question}
"""
```

### Key Limitations

#### 1. **Static Context**

- All context must be prepared before the LLM call
- Cannot dynamically fetch information based on the model's needs
- Results in over-fetching (retrieving unnecessary data)

#### 2. **No Tool Interaction**

- LLMs cannot call external APIs or functions
- Cannot perform actions (e.g., creating tickets, updating databases)
- Limited to passive information retrieval

#### 3. **Token Waste**

- Same context repeated across multiple turns in conversations
- Redundant information sent with every request
- No caching or reuse mechanisms

#### 4. **Lack of Structure**

- Context is unstructured text
- No metadata about data sources
- Models cannot reason about context provenance

#### 5. **Versioning Problems**

- No way to track which version of data was used
- Inconsistencies when data changes between requests
- Difficult to debug context-related issues

#### 6. **Poor Composability**

- Hard to combine multiple context sources
- Manual orchestration required
- No standard interface for extensions

#### 7. **Security Risks**

- All context visible in prompts
- No granular access control
- Potential data leakage in logs

### Example: The Problem in Practice

```python
# Inefficient: Loading everything upfront
def answer_question(user_id, question):
    # Loading ALL user data
    user_profile = db.get_user(user_id)
    all_orders = db.get_all_orders(user_id)  # Could be thousands
    all_tickets = db.get_all_tickets(user_id)  # Could be hundreds
    all_policies = load_all_policies()  # Large document corpus

    # Stuffing everything into prompt
    prompt = build_massive_prompt(
        user_profile, all_orders, all_tickets, all_policies, question
    )

    # Sending 50K+ tokens when model might only need 500
    response = llm.generate(prompt)
    return response
```

**Problems:**

- Loads far more data than needed
- Sends unnecessary tokens to the model (expensive)
- Slow due to data loading and large context
- Cannot dynamically adjust based on the question

---

## Introduction to Model Context Protocol (MCP)

### What is MCP?

Model Context Protocol (MCP) is an open standard that enables seamless integration between LLM applications and external data sources. It provides a universal protocol for:

- **Dynamic Context Sharing**: Providing context on-demand
- **Tool Integration**: Enabling LLMs to call functions and APIs
- **Standardized Communication**: Common interface for all integrations
- **Bidirectional Flow**: Both reading data and performing actions

### Core Philosophy

MCP is built on several key principles:

1. **Separation of Concerns**: Context sources are independent of applications
2. **Composability**: Multiple MCP servers can work together
3. **Security First**: Built-in authentication and authorization
4. **Protocol Agnostic**: Works over different transport layers
5. **Extensibility**: Easy to add new capabilities

### The MCP Model

```
┌─────────────────┐
│   LLM Client    │  (Claude Desktop, IDEs, Custom Apps)
│   Application   │
└────────┬────────┘
         │
         │ MCP Protocol
         │
    ┌────┴─────┐
    │          │
┌───▼───┐  ┌──▼────┐
│ MCP   │  │ MCP   │  (Servers provide context/tools)
│Server │  │Server │
│  #1   │  │  #2   │
└───┬───┘  └──┬────┘
    │         │
┌───▼───┐  ┌──▼────┐
│Database│ │ Files │  (Actual data sources)
└────────┘ └───────┘
```

### Key Components

1. **MCP Hosts (Clients)**: Applications that consume context (e.g., Claude Desktop, IDEs)
2. **MCP Servers**: Services that provide context and tools
3. **Protocol Layer**: Standardized communication format
4. **Transport Layer**: How messages are transmitted (stdio, HTTP, SSE)

### Simple Example

```python
# MCP Server exposing a simple tool
from mcp.server import Server
from mcp.types import Tool

server = Server("my-first-server")

@server.tool()
async def get_weather(city: str) -> str:
    """Get current weather for a city"""
    # Call weather API
    weather_data = await weather_api.get(city)
    return f"Weather in {city}: {weather_data.description}, {weather_data.temp}°C"

# LLM can now call this tool dynamically
# No need to embed weather data in prompts
```

### Benefits Over Prompt-Only Approach

| Aspect          | Prompt-Only             | MCP                       |
| --------------- | ----------------------- | ------------------------- |
| Context Loading | Upfront, all-at-once    | On-demand, selective      |
| Tool Calling    | Manual orchestration    | Native support            |
| Reusability     | Copy-paste code         | Plug-and-play servers     |
| Security        | Everything in prompt    | Granular permissions      |
| Cost            | High (redundant tokens) | Lower (efficient context) |
| Scalability     | Poor                    | Excellent                 |

---

## MCP Architecture and Key Concepts

### Architectural Overview

MCP follows a client-server architecture with three main layers:

```
┌───────────────────────────────────────────────────────┐
│                  Application Layer                     │
│  (IDEs, Chat Apps, Agents, Custom Applications)       │
└─────────────────────┬─────────────────────────────────┘
                      │
┌─────────────────────▼─────────────────────────────────┐
│                  MCP Host (Client)                     │
│  • Manages connections to MCP servers                  │
│  • Handles protocol negotiation                        │
│  • Routes requests to appropriate servers              │
│  • Manages authentication & permissions                │
└─────────────────────┬─────────────────────────────────┘
                      │
                      │ MCP Protocol (JSON-RPC 2.0)
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼─────┐ ┌────▼─────┐ ┌────▼─────┐
│ MCP Server  │ │ MCP      │ │ MCP      │
│  (Database) │ │ Server   │ │ Server   │
│             │ │ (Files)  │ │ (API)    │
└─────────────┘ └──────────┘ └──────────┘
```

### Core Concepts

#### 1. **Resources**

Resources represent data that can be read by the LLM. They are similar to files or database records.

**Characteristics:**

- URI-based identification (e.g., `file:///data/customers.json`)
- Typed content (text, binary, structured data)
- Optional metadata
- Read-only or read-write

**Example:**

```python
{
  "uri": "customer://user-12345/profile",
  "name": "Customer Profile",
  "mimeType": "application/json",
  "description": "Full customer profile including orders and preferences"
}
```

#### 2. **Prompts**

Prompts are reusable templates that MCP servers can provide. They help standardize common interaction patterns.

**Characteristics:**

- Named templates with parameters
- Can include dynamic content
- Composable and reusable
- Version-controlled

**Example:**

```python
{
  "name": "analyze_customer_issue",
  "description": "Analyze a customer support issue",
  "arguments": [
    {
      "name": "ticket_id",
      "description": "Support ticket ID",
      "required": true
    }
  ]
}
```

#### 3. **Tools**

Tools are functions that LLMs can invoke to perform actions or retrieve dynamic data.

**Characteristics:**

- Function signature with typed parameters
- Input validation
- Execution with side effects allowed
- Return values sent back to LLM

**Example:**

```python
{
  "name": "create_support_ticket",
  "description": "Create a new support ticket",
  "inputSchema": {
    "type": "object",
    "properties": {
      "customer_id": {"type": "string"},
      "issue": {"type": "string"},
      "priority": {"type": "string", "enum": ["low", "medium", "high"]}
    },
    "required": ["customer_id", "issue"]
  }
}
```

#### 4. **Sampling**

Sampling allows MCP servers to request completions from the LLM, enabling agentic patterns.

**Use Cases:**

- Multi-step reasoning
- Validation and error correction
- Recursive problem solving
- Agent-to-agent communication

### Protocol Messages

MCP uses JSON-RPC 2.0 for message format:

#### **Initialization**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "roots": { "listChanged": true },
      "sampling": {}
    },
    "clientInfo": {
      "name": "my-app",
      "version": "1.0.0"
    }
  }
}
```

#### **List Tools**

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}
```

#### **Call Tool**

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "city": "San Francisco"
    }
  }
}
```

### Security Model

#### **Authentication**

- Server-side API keys
- OAuth 2.0 support
- Client certificates

#### **Authorization**

- Tool-level permissions
- Resource access control
- Rate limiting

#### **Data Privacy**

- Servers don't see LLM prompts
- Clients control what's sent to models
- End-to-end encryption support

### Lifecycle Management

```
1. Discovery → 2. Connection → 3. Initialization → 4. Operation → 5. Shutdown
```

1. **Discovery**: Client discovers available servers
2. **Connection**: Establishes transport connection
3. **Initialization**: Negotiates capabilities
4. **Operation**: Exchanges messages (tools, resources, prompts)
5. **Shutdown**: Graceful disconnection

---

## MCP vs RAG vs Fine-tuning

### Overview of Approaches

When building LLM applications that need domain-specific knowledge, you have three main approaches:

| Approach        | What It Does                   | When to Use                                |
| --------------- | ------------------------------ | ------------------------------------------ |
| **Fine-tuning** | Modifies model weights         | Specialized knowledge, behavior patterns   |
| **RAG**         | Retrieves relevant documents   | Large knowledge bases, factual information |
| **MCP**         | Provides dynamic tools/context | Real-time data, actions, integrations      |

### Detailed Comparison

#### 1. **Fine-tuning**

**How it works:**

- Train the model on domain-specific data
- Updates model weights
- Knowledge becomes part of the model

**Pros:**

- No retrieval latency
- Knowledge deeply integrated
- Can learn style and patterns
- No external dependencies at inference

**Cons:**

- Expensive to train and update
- Knowledge becomes stale (requires retraining)
- Risk of overfitting
- Hard to update frequently
- Cannot access real-time data
- Black box (hard to debug)

**Best for:**

- Specialized domains (medical, legal)
- Consistent formatting/style
- Private/proprietary data that shouldn't leave infrastructure
- When latency is critical

**Example:**

```python
# Fine-tuned model for medical diagnosis
model = load_model("gpt-4-medical-fine-tuned")
diagnosis = model.generate("Patient symptoms: fever, cough...")
```

#### 2. **RAG (Retrieval-Augmented Generation)**

**How it works:**

- Documents are embedded and stored in vector database
- At query time, retrieve relevant documents
- Inject retrieved content into prompt
- Model generates response

**Pros:**

- Easy to update knowledge (add documents)
- Transparent (can see what was retrieved)
- Cost-effective for large knowledge bases
- No model retraining needed
- Can cite sources

**Cons:**

- Retrieval can be slow
- Quality depends on embedding/chunking
- May retrieve irrelevant content
- Still static (snapshot in time)
- Cannot perform actions
- Extra infrastructure (vector DB)

**Best for:**

- Large document collections
- FAQs and knowledge bases
- When citations are important
- Frequently updated content
- Search-like queries

**Example:**

```python
# RAG system for documentation
query = "How do I configure SSL?"
relevant_docs = vector_db.search(query, k=5)
prompt = f"Based on these docs:\n{relevant_docs}\n\nQuestion: {query}"
answer = llm.generate(prompt)
```

#### 3. **MCP (Model Context Protocol)**

**How it works:**

- MCP servers expose tools and resources
- LLM can call tools dynamically
- Real-time data and actions
- Bidirectional communication

**Pros:**

- Real-time, live data
- Can perform actions (not just read)
- Modular and composable
- Standard protocol (reusable)
- Fine-grained control
- Supports complex workflows

**Cons:**

- Requires server infrastructure
- More complex to set up
- API call latency
- Need error handling for tool failures
- Requires tool calling support in LLM

**Best for:**

- Real-time data (stocks, weather, inventory)
- Actions and transactions
- Multiple system integrations
- Dynamic workflows
- Agentic applications
- Database queries

**Example:**

```python
# MCP server for live inventory
@server.tool()
async def check_inventory(product_id: str) -> dict:
    """Check current inventory level"""
    return await inventory_db.get_current_stock(product_id)

# LLM calls tool when needed
# "What's the stock level for product ABC-123?"
# → LLM calls check_inventory("ABC-123")
# → Returns real-time data
```

### Combining Approaches

The most powerful systems often combine multiple approaches:

#### **Scenario 1: E-commerce Assistant**

```
Fine-tuning: Brand voice and personality
RAG: Product catalog and descriptions
MCP: Real-time inventory, order status, checkout actions
```

#### **Scenario 2: Code Assistant**

```
Fine-tuning: Programming patterns and idioms
RAG: Documentation and API references
MCP: File system access, running tests, Git operations
```

#### **Scenario 3: Customer Support**

```
Fine-tuning: Company communication style
RAG: Help articles and FAQs
MCP: CRM lookups, ticket creation, order management
```

### Decision Framework

```
┌─────────────────────────────────────────────────┐
│ Does the knowledge change frequently?           │
│ Yes → RAG or MCP                                │
│ No → Consider fine-tuning                       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Do you need to perform actions?                 │
│ Yes → MCP                                       │
│ No → RAG or fine-tuning                         │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Is the data structured (database)?              │
│ Yes → MCP                                       │
│ No (documents) → RAG                            │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ Do you need real-time data?                     │
│ Yes → MCP                                       │
│ No → RAG or fine-tuning                         │
└─────────────────────────────────────────────────┘
```

### Cost Comparison

Assuming 1M requests/month:

| Approach    | Setup Cost          | Ongoing Cost            | Update Cost   |
| ----------- | ------------------- | ----------------------- | ------------- |
| Fine-tuning | $$$$ (training)     | $ (inference)           | $$$ (retrain) |
| RAG         | $$ (embedding)      | $$ (retrieval + tokens) | $ (re-embed)  |
| MCP         | $$ (infrastructure) | $$ (API calls)          | $ (deploy)    |

---

## Standardising Tool Calling with MCP

### The Problem with Non-Standard Tool Calling

Before MCP, every LLM provider had different tool calling formats:

#### **OpenAI Format**

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}]
```

#### **Anthropic Format (pre-MCP)**

```python
tools = [{
    "name": "get_weather",
    "description": "Get weather",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"]
    }
}]
```

**Problems:**

- Different schemas for each provider
- No reusability across platforms
- Manual conversion required
- Inconsistent behavior

### MCP's Standardization

MCP provides a unified tool definition format:

```python
# Standard MCP tool definition
{
    "name": "get_weather",
    "description": "Get current weather for a city",
    "inputSchema": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["city"]
    }
}
```

### Key Benefits

#### 1. **Provider Independence**

Write once, use with any MCP-compatible LLM:

```python
# Same tool works with Claude, GPT, Gemini, etc.
@server.tool()
async def get_weather(city: str, units: str = "celsius") -> str:
    """Get current weather for a city"""
    return await weather_api.fetch(city, units)
```

#### 2. **Type Safety**

MCP enforces JSON Schema validation:

```python
# Automatic validation
inputSchema = {
    "type": "object",
    "properties": {
        "age": {"type": "integer", "minimum": 0, "maximum": 150},
        "email": {"type": "string", "format": "email"}
    }
}
```

#### 3. **Rich Metadata**

```python
{
    "name": "search_products",
    "description": "Search product catalog",
    "inputSchema": {...},
    "metadata": {
        "category": "ecommerce",
        "rateLimit": "100/minute",
        "requiresAuth": true,
        "version": "2.0"
    }
}
```

#### 4. **Discovery**

Clients can list all available tools:

```python
# List all tools from all servers
tools = await client.list_tools()
for tool in tools:
    print(f"{tool.name}: {tool.description}")
```

### Creating Standardized Tools

#### **Basic Tool**

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("my-tools")

@server.tool()
async def calculator(operation: str, a: float, b: float) -> float:
    """
    Perform basic arithmetic operations

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number
    """
    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else None
    }

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    result = operations[operation](a, b)
    return result
```

#### **Tool with Complex Types**

```python
from typing import List, Dict
from pydantic import BaseModel

class Customer(BaseModel):
    id: str
    name: str
    email: str

@server.tool()
async def create_customer(
    name: str,
    email: str,
    tags: List[str] = [],
    metadata: Dict[str, str] = {}
) -> Customer:
    """Create a new customer record"""
    customer = await db.customers.insert({
        "name": name,
        "email": email,
        "tags": tags,
        "metadata": metadata
    })
    return Customer(**customer)
```

#### **Tool with Error Handling**

```python
@server.tool()
async def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email to a recipient"""
    try:
        result = await email_service.send(
            to=to,
            subject=subject,
            body=body
        )
        return {
            "success": True,
            "message_id": result.id
        }
    except EmailValidationError as e:
        return {
            "success": False,
            "error": f"Invalid email address: {str(e)}"
        }
    except EmailServiceError as e:
        return {
            "success": False,
            "error": f"Failed to send: {str(e)}"
        }
```

### Tool Composition

MCP tools can be composed to create higher-level operations:

```python
@server.tool()
async def get_order_details(order_id: str) -> dict:
    """Get detailed information about an order"""
    # Combines multiple data sources
    order = await db.get_order(order_id)
    customer = await db.get_customer(order.customer_id)
    products = await db.get_products(order.product_ids)

    return {
        "order": order,
        "customer": customer,
        "products": products,
        "total": sum(p.price * p.quantity for p in products)
    }

@server.tool()
async def process_refund(order_id: str, reason: str) -> dict:
    """Process a refund for an order"""
    # Uses other tools internally
    order_details = await get_order_details(order_id)

    refund = await payment_service.create_refund(
        amount=order_details["total"],
        order_id=order_id
    )

    await notification_service.send(
        to=order_details["customer"]["email"],
        template="refund_processed",
        data={"refund": refund, "reason": reason}
    )

    return {
        "success": True,
        "refund_id": refund.id,
        "amount": refund.amount
    }
```

### Best Practices

#### 1. **Clear Descriptions**

```python
# ❌ Poor
async def get_data(id: str) -> dict:
    """Get data"""

# ✅ Good
async def get_customer_profile(customer_id: str) -> dict:
    """
    Retrieve complete customer profile including contact info,
    order history, and preferences.

    Args:
        customer_id: Unique customer identifier (UUID format)

    Returns:
        Dictionary containing profile, orders, and preferences
    """
```

#### 2. **Input Validation**

```python
from pydantic import validator

@server.tool()
async def create_appointment(date: str, duration_minutes: int) -> dict:
    """Schedule an appointment"""

    # Validate date format
    try:
        appointment_date = datetime.fromisoformat(date)
    except ValueError:
        raise ValueError("Date must be in ISO format (YYYY-MM-DD)")

    # Validate duration
    if duration_minutes not in [15, 30, 45, 60]:
        raise ValueError("Duration must be 15, 30, 45, or 60 minutes")

    # Check availability
    if not await calendar.is_available(appointment_date, duration_minutes):
        raise ValueError("Time slot not available")

    return await calendar.book(appointment_date, duration_minutes)
```

#### 3. **Idempotency**

```python
@server.tool()
async def create_order(
    customer_id: str,
    items: List[dict],
    idempotency_key: str
) -> dict:
    """
    Create a new order (idempotent)

    Args:
        idempotency_key: Unique key to prevent duplicate orders
    """
    # Check if already processed
    existing = await db.orders.find_one({"idempotency_key": idempotency_key})
    if existing:
        return existing

    # Create new order
    order = await db.orders.insert({
        "customer_id": customer_id,
        "items": items,
        "idempotency_key": idempotency_key,
        "created_at": datetime.now()
    })

    return order
```

---

## Transport Protocols: Stdio vs SSE vs Streamable HTTP

MCP supports multiple transport protocols for different deployment scenarios. Understanding when to use each is crucial for building efficient systems.

### Overview

| Transport | Use Case                        | Latency | Complexity | Streaming          |
| --------- | ------------------------------- | ------- | ---------- | ------------------ |
| **Stdio** | Local, same-machine             | Lowest  | Simple     | ✅                 |
| **SSE**   | Browser apps, real-time updates | Low     | Medium     | ✅                 |
| **HTTP**  | Web APIs, microservices         | Medium  | Medium     | ✅ (with chunking) |

---

### 1. Stdio (Standard Input/Output)

#### **What It Is**

- Communication via standard input/output streams
- Process-to-process communication
- Parent process spawns child MCP server

#### **Architecture**

```
┌──────────────┐
│  MCP Client  │
│   (Parent)   │
└──────┬───────┘
       │ spawn
       ▼
┌──────────────┐
│  MCP Server  │
│   (Child)    │
└──────────────┘
     ↕ stdin/stdout
Messages (JSON-RPC)
```

#### **When to Use**

- ✅ Desktop applications (VS Code, Claude Desktop)
- ✅ CLI tools
- ✅ Local development
- ✅ Same-machine deployments
- ❌ Web applications
- ❌ Remote servers
- ❌ Cloud deployments

#### **Advantages**

- Ultra-low latency
- No network overhead
- Automatic process lifecycle management
- Simple security model (process isolation)
- No authentication needed

#### **Disadvantages**

- Not suitable for remote access
- Tied to specific machine
- Process management complexity
- Platform-specific considerations

#### **Implementation Example**

**Server (Python):**

```python
import asyncio
import sys
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("local-file-server")

@server.tool()
async def read_file(path: str) -> str:
    """Read local file contents"""
    with open(path, 'r') as f:
        return f.read()

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

**Client Configuration:**

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "python",
      "args": ["server.py"],
      "env": {
        "HOME": "/Users/username"
      }
    }
  }
}
```

---

### 2. SSE (Server-Sent Events)

#### **What It Is**

- HTTP-based unidirectional streaming
- Server pushes events to client
- Long-lived connection
- Built on HTTP/1.1

#### **Architecture**

```
┌──────────────┐
│  Web Client  │
│  (Browser)   │
└──────┬───────┘
       │ HTTP GET (SSE)
       ▼
┌──────────────┐
│  MCP Server  │
│   (HTTP)     │
└──────────────┘
     ↓ Event Stream
Messages flow server→client
```

#### **When to Use**

- ✅ Web applications
- ✅ Browser-based tools
- ✅ Real-time dashboards
- ✅ Notification systems
- ✅ One-way data flow (server→client)
- ❌ Bidirectional communication needs
- ❌ Binary data transfer

#### **Advantages**

- Native browser support
- Automatic reconnection
- Simple protocol
- Works through proxies
- HTTP-based (familiar)

#### **Disadvantages**

- One-directional (server → client)
- Limited binary support
- Browser connection limits
- Not ideal for bidirectional tools

#### **Implementation Example**

**Server (FastAPI):**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from mcp.server import Server
import json
import asyncio

app = FastAPI()
mcp_server = Server("sse-server")

@mcp_server.tool()
async def get_stock_price(symbol: str) -> float:
    """Get current stock price"""
    return await stock_api.get_price(symbol)

async def event_generator():
    """Generate SSE events"""
    while True:
        # Send tool updates
        tools = mcp_server.list_tools()
        event_data = json.dumps({
            "type": "tools_updated",
            "tools": [t.dict() for t in tools]
        })
        yield f"data: {event_data}\n\n"

        await asyncio.sleep(5)

@app.get("/events")
async def sse_endpoint():
    """SSE endpoint for MCP messages"""
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.post("/tools/call")
async def call_tool(name: str, arguments: dict):
    """HTTP endpoint for tool calls"""
    result = await mcp_server.call_tool(name, arguments)
    return {"result": result}
```

**Client (JavaScript):**

```javascript
// Connect to SSE endpoint
const eventSource = new EventSource("http://localhost:8000/events");

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === "tools_updated") {
    console.log("Available tools:", data.tools);
  }
};

// Call tools via HTTP POST
async function callTool(name, args) {
  const response = await fetch("http://localhost:8000/tools/call", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, arguments: args }),
  });
  return await response.json();
}

// Use the tool
const price = await callTool("get_stock_price", { symbol: "AAPL" });
console.log("Stock price:", price.result);
```

---

### 3. Streamable HTTP

#### **What It Is**

- Standard HTTP request/response
- Supports chunked transfer encoding
- Can stream responses
- RESTful or RPC-style

#### **Architecture**

```
┌──────────────┐
│  MCP Client  │
│  (Any HTTP)  │
└──────┬───────┘
       │ HTTP Request
       ▼
┌──────────────┐
│  MCP Server  │
│  (HTTP API)  │
└──────┬───────┘
       │ HTTP Response (streamed)
       ▼
Chunked response
```

#### **When to Use**

- ✅ Microservices architecture
- ✅ Cloud deployments
- ✅ Mobile apps
- ✅ Cross-platform clients
- ✅ Load balancing needed
- ✅ Caching requirements
- ✅ Existing HTTP infrastructure

#### **Advantages**

- Universal compatibility
- Works with all HTTP clients
- Standard security (TLS, OAuth)
- Scalable (load balancers, CDN)
- Caching support
- Familiar to developers

#### **Disadvantages**

- Higher latency than stdio
- More overhead
- Connection setup cost
- Requires network infrastructure

#### **Implementation Example**

**Server (FastAPI with Streaming):**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from mcp.server import Server
from mcp.types import TextContent
import json

app = FastAPI()
server = Server("http-server")

@server.tool()
async def analyze_document(content: str) -> str:
    """Analyze document content (returns streaming response)"""
    # Simulate streaming analysis
    results = []
    for i, sentence in enumerate(content.split('.')):
        sentiment = await analyze_sentiment(sentence)
        results.append(f"Sentence {i}: {sentiment}")
        yield {"progress": i, "result": sentiment}

    return "\n".join(results)

@app.post("/mcp/tools/call")
async def call_tool_streaming(name: str, arguments: dict):
    """Call MCP tool with streaming response"""

    async def generate():
        """Stream tool execution results"""
        try:
            # Call the tool
            async for chunk in server.call_tool_streaming(name, arguments):
                # Stream chunks as JSON lines
                yield json.dumps(chunk) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"  # Newline-delimited JSON
    )

@app.get("/mcp/tools/list")
async def list_tools():
    """List all available tools"""
    tools = server.list_tools()
    return {
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema
            }
            for t in tools
        ]
    }

@app.post("/mcp/tools/call-sync")
async def call_tool_sync(name: str, arguments: dict):
    """Non-streaming tool call"""
    result = await server.call_tool(name, arguments)
    return {"result": result}
```

**Client (Python):**

```python
import httpx
import json

class MCPHTTPClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient()

    async def list_tools(self):
        """List available tools"""
        response = await self.client.get(f"{self.base_url}/mcp/tools/list")
        return response.json()["tools"]

    async def call_tool(self, name: str, arguments: dict):
        """Call tool (non-streaming)"""
        response = await self.client.post(
            f"{self.base_url}/mcp/tools/call-sync",
            json={"name": name, "arguments": arguments}
        )
        return response.json()["result"]

    async def call_tool_streaming(self, name: str, arguments: dict):
        """Call tool with streaming response"""
        async with self.client.stream(
            "POST",
            f"{self.base_url}/mcp/tools/call",
            json={"name": name, "arguments": arguments}
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    yield json.loads(line)

# Usage
client = MCPHTTPClient("http://localhost:8000")

# List tools
tools = await client.list_tools()
print("Available tools:", tools)

# Call tool (non-streaming)
result = await client.call_tool("get_weather", {"city": "Tokyo"})
print("Weather:", result)

# Call tool (streaming)
async for chunk in client.call_tool_streaming(
    "analyze_document",
    {"content": "Long document text..."}
):
    print("Progress:", chunk)
```

**Client (TypeScript/Node.js):**

```typescript
import fetch from "node-fetch";

class MCPHTTPClient {
  constructor(private baseUrl: string) {}

  async listTools() {
    const response = await fetch(`${this.baseUrl}/mcp/tools/list`);
    const data = await response.json();
    return data.tools;
  }

  async callTool(name: string, arguments: any) {
    const response = await fetch(`${this.baseUrl}/mcp/tools/call-sync`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, arguments }),
    });
    const data = await response.json();
    return data.result;
  }

  async *callToolStreaming(name: string, arguments: any) {
    const response = await fetch(`${this.baseUrl}/mcp/tools/call`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, arguments }),
    });

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split("\n").filter((l) => l.trim());

      for (const line of lines) {
        yield JSON.parse(line);
      }
    }
  }
}

// Usage
const client = new MCPHTTPClient("http://localhost:8000");

// List tools
const tools = await client.listTools();
console.log("Available tools:", tools);

// Streaming call
for await (const chunk of client.callToolStreaming("analyze_document", {
  content: "Long document...",
})) {
  console.log("Chunk:", chunk);
}
```

---

### Choosing the Right Transport

#### **Decision Matrix**

```
Deployment Model:
├─ Same machine/local → Stdio
├─ Web browser client → SSE or HTTP
├─ Mobile app → HTTP
├─ Microservices → HTTP
└─ Desktop app → Stdio or HTTP

Performance Priority:
├─ Ultra-low latency → Stdio
├─ Real-time updates → SSE
└─ Standard API → HTTP

Infrastructure:
├─ Existing HTTP services → HTTP
├─ Simple deployment → Stdio
└─ Browser required → SSE or HTTP
```

#### **Hybrid Approach**

Some applications use multiple transports:

```python
# Server supporting multiple transports
from mcp.server import Server

server = Server("multi-transport-server")

# Stdio mode
if sys.stdin.isatty():
    await server.run_stdio()

# HTTP mode
else:
    app = FastAPI()
    app.include_router(server.create_http_router())
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Integrating MCP with Agents

Agents are autonomous systems that can plan, execute, and adapt their behavior. MCP provides the perfect foundation for building powerful agents by giving them access to tools and dynamic context.

### Agent Architecture with MCP

```
┌─────────────────────────────────────┐
│         Agent Core                   │
│  ┌──────────────────────────┐       │
│  │  LLM (Reasoning Engine)  │       │
│  └───────────┬──────────────┘       │
│              │                       │
│  ┌───────────▼──────────────┐       │
│  │   Agent Loop              │       │
│  │  • Plan                   │       │
│  │  • Execute                │       │
│  │  • Observe                │       │
│  │  • Reflect                │       │
│  └───────────┬──────────────┘       │
│              │                       │
└──────────────┼──────────────────────┘
               │
               │ MCP Protocol
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼────┐  ┌──▼────┐
│ Tools │  │Context│  │Actions│
└───────┘  └───────┘  └───────┘
```

### Core Concepts

#### 1. **Agent Loop**

Agents follow a continuous loop:

```python
while not task_complete:
    # 1. Observe: Get current state
    state = await observe_environment()

    # 2. Plan: Decide what to do
    plan = await llm.plan(state, goal)

    # 3. Execute: Use MCP tools
    results = await execute_plan(plan)

    # 4. Reflect: Evaluate progress
    progress = await llm.reflect(results)

    # 5. Adapt: Update strategy if needed
    if not progress.satisfactory:
        goal = await llm.revise_goal(progress)
```

#### 2. **Tool Selection**

Agents must choose appropriate tools:

```python
# Agent receives task
task = "Find customers who haven't ordered in 90 days and send reminder"

# Agent breaks down into steps
steps = [
    "1. Query database for inactive customers",
    "2. Filter by last order date > 90 days",
    "3. For each customer, send personalized email"
]

# Agent selects MCP tools
tools_needed = [
    "query_customers",      # MCP tool for DB access
    "send_email"           # MCP tool for email
]

# Agent executes
customers = await mcp.call_tool("query_customers", {
    "criteria": {"last_order_days_ago": {"$gt": 90}}
})

for customer in customers:
    await mcp.call_tool("send_email", {
        "to": customer.email,
        "template": "reengagement",
        "data": {"name": customer.name}
    })
```

### Building an Agent with MCP

#### **Basic Agent Implementation**

```python
from mcp.client import Client
from typing import List, Dict
import anthropic

class MCPAgent:
    def __init__(self, mcp_servers: List[str], llm_model: str = "claude-3-5-sonnet-20241022"):
        self.mcp_client = Client()
        self.llm = anthropic.Anthropic()
        self.model = llm_model
        self.conversation_history = []

        # Connect to MCP servers
        for server_config in mcp_servers:
            self.mcp_client.connect(server_config)

    async def run(self, task: str) -> str:
        """Execute a task using available MCP tools"""

        # Get available tools from MCP servers
        tools = await self.mcp_client.list_tools()

        # Initialize conversation with task
        self.conversation_history.append({
            "role": "user",
            "content": task
        })

        # Agent loop
        max_iterations = 10
        for iteration in range(max_iterations):
            # Ask LLM what to do next
            response = self.llm.messages.create(
                model=self.model,
                messages=self.conversation_history,
                tools=tools,
                max_tokens=4096
            )

            # Check if done
            if response.stop_reason == "end_turn":
                final_response = response.content[0].text
                return final_response

            # Execute tool calls
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        # Call MCP tool
                        result = await self.mcp_client.call_tool(
                            block.name,
                            block.input
                        )

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                # Add tool results to conversation
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })

        return "Task not completed within iteration limit"

# Usage
agent = MCPAgent(mcp_servers=[
    {"type": "stdio", "command": "python", "args": ["database_server.py"]},
    {"type": "stdio", "command": "python", "args": ["email_server.py"]}
])

result = await agent.run(
    "Find all orders from last week and generate a sales report"
)
print(result)
```

#### **Advanced Agent with Planning**

```python
class PlanningAgent(MCPAgent):
    async def create_plan(self, task: str) -> List[Dict]:
        """Create a step-by-step plan for the task"""

        tools = await self.mcp_client.list_tools()
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}" for t in tools
        ])

        planning_prompt = f"""
        Task: {task}

        Available tools:
        {tool_descriptions}

        Create a detailed step-by-step plan to complete this task.
        For each step, specify:
        1. Action description
        2. Which tool to use
        3. Expected outcome

        Return the plan as JSON.
        """

        response = self.llm.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": planning_prompt}],
            max_tokens=2000
        )

        plan = self.parse_plan(response.content[0].text)
        return plan

    async def execute_plan(self, plan: List[Dict]) -> Dict:
        """Execute a plan step by step"""

        results = {}

        for step in plan:
            print(f"Executing: {step['description']}")

            # Execute the tool
            result = await self.mcp_client.call_tool(
                step['tool'],
                step['arguments']
            )

            # Store result
            results[step['id']] = result

            # Check if step was successful
            if not self.is_step_successful(result, step['expected_outcome']):
                # Agent can adapt and try alternative approach
                print(f"Step failed, attempting recovery...")
                recovery_result = await self.recover_from_failure(step, result)
                results[step['id']] = recovery_result

        return results

    async def run(self, task: str) -> str:
        """Execute task with planning"""

        # Create plan
        plan = await self.create_plan(task)
        print(f"Created plan with {len(plan)} steps")

        # Execute plan
        results = await self.execute_plan(plan)

        # Synthesize final answer
        synthesis_prompt = f"""
        Task: {task}
        Plan execution results: {results}

        Synthesize a final answer to the original task.
        """

        response = self.llm.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=2000
        )

        return response.content[0].text
```

### Common Agent Patterns with MCP

#### **1. ReAct Agent (Reason + Act)**

```python
class ReActAgent:
    """Agent that alternates between reasoning and acting"""

    async def run(self, task: str):
        thoughts = []
        actions = []

        while not self.is_task_complete():
            # Reason: Think about what to do
            thought = await self.llm.generate(
                f"Task: {task}\nThought:"
            )
            thoughts.append(thought)

            # Act: Take action using MCP tool
            action_name, action_args = self.parse_action(thought)
            action_result = await self.mcp.call_tool(action_name, action_args)
            actions.append(action_result)

            # Observe: Process result
            observation = self.process_result(action_result)

            # Continue reasoning with new information
            context = {
                "thoughts": thoughts,
                "actions": actions,
                "observation": observation
            }
```

#### **2. Hierarchical Agent**

```python
class ManagerAgent:
    """High-level agent that delegates to specialized agents"""

    def __init__(self):
        self.specialist_agents = {
            "data": DataAnalysisAgent(),
            "communication": CommunicationAgent(),
            "automation": AutomationAgent()
        }

    async def run(self, task: str):
        # Decompose task
        subtasks = await self.decompose_task(task)

        # Assign to specialists
        results = {}
        for subtask in subtasks:
            agent_type = self.identify_agent(subtask)
            agent = self.specialist_agents[agent_type]
            results[subtask.id] = await agent.run(subtask)

        # Synthesize results
        return await self.synthesize(results)

class DataAnalysisAgent(MCPAgent):
    """Specialized agent for data analysis tasks"""
    pass
```

#### **3. Feedback Loop Agent**

```python
class FeedbackAgent(MCPAgent):
    """Agent that learns from user feedback"""

    async def run_with_feedback(self, task: str):
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            # Generate result
            result = await self.run(task)

            # Get user feedback
            feedback = await self.get_user_feedback(result)

            if feedback.approved:
                return result

            # Improve based on feedback
            task = self.incorporate_feedback(task, feedback)
            attempts += 1

        return result
```

### MCP Tools for Agents

#### **Essential Agent Tools**

```python
# Memory/State Management
@server.tool()
async def save_state(key: str, value: dict) -> bool:
    """Save agent state for later retrieval"""
    await state_store.set(key, value)
    return True

@server.tool()
async def load_state(key: str) -> dict:
    """Load previously saved agent state"""
    return await state_store.get(key)

# Information Gathering
@server.tool()
async def search_knowledge_base(query: str, top_k: int = 5) -> List[dict]:
    """Search internal knowledge base"""
    results = await vector_db.search(query, k=top_k)
    return results

@server.tool()
async def web_search(query: str) -> List[dict]:
    """Search the web for information"""
    results = await search_api.search(query)
    return results

# Action Execution
@server.tool()
async def execute_code(code: str, language: str = "python") -> dict:
    """Execute code in sandboxed environment"""
    result = await code_executor.run(code, language)
    return {"output": result.stdout, "error": result.stderr}

@server.tool()
async def create_task(title: str, description: str) -> str:
    """Create a new task in task management system"""
    task = await task_manager.create(title, description)
    return task.id

# Verification
@server.tool()
async def verify_result(expected: str, actual: str) -> dict:
    """Verify if result matches expectations"""
    similarity = await compute_similarity(expected, actual)
    return {
        "matches": similarity > 0.9,
        "similarity": similarity,
        "differences": find_differences(expected, actual)
    }
```

### Best Practices

1. **Error Handling**: Agents should gracefully handle tool failures
2. **Timeouts**: Set reasonable timeouts for long-running tools
3. **Logging**: Track agent decisions for debugging
4. **Safety**: Implement guardrails to prevent harmful actions
5. **Cost Control**: Monitor API usage and set limits

---

## Integration MCP with LangChain

LangChain is a popular framework for building LLM applications. Integrating MCP with LangChain combines the best of both worlds: LangChain's powerful orchestration with MCP's standardized tool protocol.

### Why Integrate MCP with LangChain?

1. **Standardization**: Use MCP servers across different frameworks
2. **Reusability**: Write tools once, use in LangChain and other MCP clients
3. **Interoperability**: Connect LangChain agents to MCP ecosystem
4. **Simplification**: Reduce custom integration code

### Architecture

```
┌─────────────────────────────────┐
│   LangChain Application         │
│   • Agents                      │
│   • Chains                      │
│   • Tools                       │
└───────────┬─────────────────────┘
            │
            │ MCP Adapter
            │
┌───────────▼─────────────────────┐
│   MCP Client                    │
└───────────┬─────────────────────┘
            │
    ┌───────┼───────┐
    │       │       │
┌───▼───┐ ┌▼────┐ ┌▼────┐
│ MCP   │ │ MCP │ │ MCP │
│Server │ │Srv  │ │Srv  │
└───────┘ └─────┘ └─────┘
```

### Implementation

#### **1. MCP Tool Wrapper for LangChain**

```python
from langchain.tools import BaseTool
from mcp.client import Client
from typing import Optional, Type
from pydantic import BaseModel, Field

class MCPTool(BaseTool):
    """Wrapper to use MCP tools in LangChain"""

    mcp_client: Client
    mcp_tool_name: str
    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None

    def __init__(self, mcp_client: Client, mcp_tool: dict):
        """Initialize from MCP tool definition"""

        super().__init__(
            mcp_client=mcp_client,
            mcp_tool_name=mcp_tool["name"],
            name=mcp_tool["name"],
            description=mcp_tool["description"]
        )

        # Convert JSON Schema to Pydantic model
        if "inputSchema" in mcp_tool:
            self.args_schema = self._create_pydantic_model(
                mcp_tool["inputSchema"]
            )

    def _run(self, **kwargs) -> str:
        """Synchronous execution (required by LangChain)"""
        import asyncio
        return asyncio.run(self._arun(**kwargs))

    async def _arun(self, **kwargs) -> str:
        """Asynchronous execution"""
        result = await self.mcp_client.call_tool(
            self.mcp_tool_name,
            kwargs
        )

        # Convert result to string for LangChain
        if isinstance(result, dict):
            return str(result)
        return result

    @staticmethod
    def _create_pydantic_model(json_schema: dict) -> Type[BaseModel]:
        """Convert JSON Schema to Pydantic model"""
        from pydantic import create_model

        properties = json_schema.get("properties", {})
        required = set(json_schema.get("required", []))

        fields = {}
        for prop_name, prop_schema in properties.items():
            python_type = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict
            }.get(prop_schema.get("type"), str)

            default = ... if prop_name in required else None
            fields[prop_name] = (python_type, Field(default=default))

        return create_model("DynamicModel", **fields)


class MCPToolkit:
    """Load all tools from MCP servers as LangChain tools"""

    def __init__(self, mcp_servers: list):
        self.client = Client()

        # Connect to all MCP servers
        for server_config in mcp_servers:
            self.client.connect(server_config)

    async def get_tools(self) -> list[MCPTool]:
        """Get all MCP tools as LangChain tools"""
        mcp_tools = await self.client.list_tools()

        langchain_tools = []
        for mcp_tool in mcp_tools:
            lc_tool = MCPTool(
                mcp_client=self.client,
                mcp_tool=mcp_tool
            )
            langchain_tools.append(lc_tool)

        return langchain_tools
```

#### **2. Using MCP Tools in LangChain Agents**

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

async def create_agent_with_mcp():
    """Create LangChain agent with MCP tools"""

    # Initialize MCP toolkit
    toolkit = MCPToolkit(mcp_servers=[
        {
            "type": "stdio",
            "command": "python",
            "args": ["servers/database_server.py"]
        },
        {
            "type": "stdio",
            "command": "python",
            "args": ["servers/email_server.py"]
        }
    ])

    # Get all MCP tools as LangChain tools
    tools = await toolkit.get_tools()

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4", temperature=0)

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant with access to various tools."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # Create executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True
    )

    return agent_executor

# Usage
agent = await create_agent_with_mcp()

result = await agent.ainvoke({
    "input": "Find all customers from California and send them a promotional email"
})

print(result["output"])
```

#### **3. MCP Chain Integration**

```python
from langchain.chains import LLMChain

class MCPChain(LLMChain):
    """LangChain chain that uses MCP tools"""

    mcp_client: Client

    async def _call(self, inputs: dict) -> dict:
        """Execute chain with MCP tool calls"""

        # Get LLM response
        llm_output = await self.llm.agenerate([self.prompt.format(**inputs)])

        # Parse tool calls from response
        tool_calls = self.parse_tool_calls(llm_output)

        # Execute MCP tools
        tool_results = []
        for tool_call in tool_calls:
            result = await self.mcp_client.call_tool(
                tool_call["name"],
                tool_call["arguments"]
            )
            tool_results.append(result)

        # Generate final response with tool results
        final_inputs = {
            **inputs,
            "tool_results": tool_results
        }
        final_output = await self.llm.agenerate([
            self.prompt.format(**final_inputs)
        ])

        return {"output": final_output.generations[0][0].text}
```

#### **4. RAG with MCP**

Combine LangChain's RAG capabilities with MCP tools:

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

class MCPRAGChain:
    """RAG chain with MCP tool access"""

    def __init__(self, mcp_servers: list):
        self.mcp_toolkit = MCPToolkit(mcp_servers)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None

    async def setup(self, documents: list):
        """Initialize vector store"""
        self.vectorstore = await FAISS.afrom_documents(
            documents,
            self.embeddings
        )

    async def query(self, question: str) -> str:
        """Answer question using RAG + MCP tools"""

        # Get MCP tools
        tools = await self.mcp_toolkit.get_tools()

        # Create retrieval QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4"),
            retriever=self.vectorstore.as_retriever(),
            chain_type="stuff"
        )

        # First, try to answer from documents
        doc_answer = await qa_chain.ainvoke({"query": question})

        # If document answer is insufficient, use MCP tools
        if self.needs_real_time_data(doc_answer):
            # Use agent with tools
            agent = await create_agent_with_mcp()
            final_answer = await agent.ainvoke({"input": question})
            return final_answer["output"]

        return doc_answer["result"]

# Usage
rag_chain = MCPRAGChain(mcp_servers=[...])
await rag_chain.setup(documents)

answer = await rag_chain.query(
    "What's the current inventory level for product SKU-123?"
)
```

#### **5. LangGraph with MCP**

Integrate MCP with LangGraph for complex workflows:

```python
from langgraph.graph import Graph, END
from typing import TypedDict

class AgentState(TypedDict):
    input: str
    plan: list
    executed_steps: list
    final_answer: str

async def create_mcp_graph():
    """Create LangGraph workflow with MCP tools"""

    # Initialize MCP
    toolkit = MCPToolkit(mcp_servers=[...])
    tools = await toolkit.get_tools()

    # Create graph
    workflow = Graph()

    # Define nodes
    async def plan_node(state: AgentState):
        """Create execution plan"""
        llm = ChatOpenAI(model="gpt-4")
        plan = await llm.ainvoke(
            f"Create plan for: {state['input']}\nAvailable tools: {tools}"
        )
        return {"plan": parse_plan(plan)}

    async def execute_node(state: AgentState):
        """Execute plan using MCP tools"""
        results = []
        for step in state["plan"]:
            tool_result = await toolkit.client.call_tool(
                step["tool"],
                step["args"]
            )
            results.append(tool_result)
        return {"executed_steps": results}

    async def synthesize_node(state: AgentState):
        """Create final answer"""
        llm = ChatOpenAI(model="gpt-4")
        answer = await llm.ainvoke(
            f"Synthesize answer from: {state['executed_steps']}"
        )
        return {"final_answer": answer}

    # Build graph
    workflow.add_node("plan", plan_node)
    workflow.add_node("execute", execute_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.add_edge("plan", "execute")
    workflow.add_edge("execute", "synthesize")
    workflow.add_edge("synthesize", END)

    workflow.set_entry_point("plan")

    return workflow.compile()

# Usage
graph = await create_mcp_graph()
result = await graph.ainvoke({
    "input": "Analyze sales data and send report"
})
print(result["final_answer"])
```

### Best Practices

1. **Error Handling**: Wrap MCP calls in try-except blocks
2. **Async/Sync**: Handle LangChain's sync requirements with asyncio
3. **Tool Discovery**: Dynamically load MCP tools at runtime
4. **Caching**: Cache MCP client connections for performance
5. **Logging**: Log MCP tool calls for debugging

---

## How to Build Custom MCP Servers

Building custom MCP servers allows you to expose any data source or API as MCP-compatible tools. This section provides a comprehensive guide.

### Setup

```bash
# Install MCP SDK
pip install mcp

# Or for development
pip install "mcp[dev]"
```

### Basic Server Structure

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

# Create server instance
server = Server("my-custom-server")

# Define tools
@server.tool()
async def my_tool(param: str) -> str:
    """Tool description"""
    return f"Result: {param}"

# Run server
if __name__ == "__main__":
    import asyncio
    import mcp.server.stdio

    async def main():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )

    asyncio.run(main())
```

### Example 1: Database Server

```python
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource
import asyncpg
import json

server = Server("postgres-server")

# Database connection
db_pool = None

@server.initialize()
async def initialize():
    """Initialize database connection"""
    global db_pool
    db_pool = await asyncpg.create_pool(
        host="localhost",
        database="mydb",
        user="user",
        password="password"
    )

@server.tool()
async def query_database(sql: str, parameters: list = []) -> str:
    """
    Execute SQL query

    Args:
        sql: SQL query string
        parameters: Query parameters for safe parameterization
    """
    async with db_pool.acquire() as conn:
        try:
            rows = await conn.fetch(sql, *parameters)
            results = [dict(row) for row in rows]
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

@server.tool()
async def get_table_schema(table_name: str) -> str:
    """Get schema information for a table"""
    async with db_pool.acquire() as conn:
        schema = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = $1
            ORDER BY ordinal_position
        """, table_name)

        return json.dumps([dict(row) for row in schema])

@server.tool()
async def list_tables() -> str:
    """List all tables in the database"""
    async with db_pool.acquire() as conn:
        tables = await conn.fetch("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
        """)

        return json.dumps([row["tablename"] for row in tables])

# Expose tables as resources
@server.list_resources()
async def list_resources():
    """List database tables as resources"""
    async with db_pool.acquire() as conn:
        tables = await conn.fetch("""
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = 'public'
        """)

        return [
            Resource(
                uri=f"postgres://table/{table['tablename']}",
                name=table["tablename"],
                mimeType="application/json",
                description=f"Database table: {table['tablename']}"
            )
            for table in tables
        ]

@server.read_resource()
async def read_resource(uri: str):
    """Read table data"""
    # Extract table name from URI
    table_name = uri.split("/")[-1]

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(f"SELECT * FROM {table_name} LIMIT 100")
        data = [dict(row) for row in rows]

        return TextContent(
            type="text",
            text=json.dumps(data, default=str)
        )

@server.shutdown()
async def shutdown():
    """Clean up database connection"""
    if db_pool:
        await db_pool.close()
```

### Example 2: File System Server

```python
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource
import os
import aiofiles
from pathlib import Path

server = Server("filesystem-server")

BASE_PATH = Path("/safe/directory")

@server.tool()
async def read_file(path: str) -> str:
    """Read file contents"""
    full_path = BASE_PATH / path

    # Security check
    if not full_path.is_relative_to(BASE_PATH):
        raise ValueError("Access denied: path outside allowed directory")

    async with aiofiles.open(full_path, 'r') as f:
        content = await f.read()

    return content

@server.tool()
async def write_file(path: str, content: str) -> str:
    """Write content to file"""
    full_path = BASE_PATH / path

    # Security check
    if not full_path.is_relative_to(BASE_PATH):
        raise ValueError("Access denied")

    # Create parent directories
    full_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(full_path, 'w') as f:
        await f.write(content)

    return f"Written {len(content)} bytes to {path}"

@server.tool()
async def list_directory(path: str = ".") -> str:
    """List directory contents"""
    full_path = BASE_PATH / path

    if not full_path.is_relative_to(BASE_PATH):
        raise ValueError("Access denied")

    items = []
    for item in full_path.iterdir():
        items.append({
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else None
        })

    return json.dumps(items)

@server.tool()
async def search_files(pattern: str, path: str = ".") -> str:
    """Search for files matching pattern"""
    full_path = BASE_PATH / path

    if not full_path.is_relative_to(BASE_PATH):
        raise ValueError("Access denied")

    matches = list(full_path.rglob(pattern))

    return json.dumps([
        str(m.relative_to(BASE_PATH)) for m in matches
    ])

@server.list_resources()
async def list_resources():
    """List files as resources"""
    resources = []

    for file_path in BASE_PATH.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(BASE_PATH)
            resources.append(Resource(
                uri=f"file:///{rel_path}",
                name=file_path.name,
                mimeType=get_mime_type(file_path),
                description=f"File: {rel_path}"
            ))

    return resources
```

### Example 3: API Integration Server

```python
from mcp.server import Server
import httpx
import os

server = Server("api-integration-server")

API_KEY = os.getenv("API_KEY")

@server.tool()
async def get_weather(city: str, units: str = "metric") -> str:
    """Get current weather for a city"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": API_KEY,
                "units": units
            }
        )
        response.raise_for_status()
        data = response.json()

        return json.dumps({
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"]
        })

@server.tool()
async def geocode_address(address: str) -> str:
    """Convert address to coordinates"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json"}
        )
        data = response.json()

        if data:
            return json.dumps({
                "lat": data[0]["lat"],
                "lon": data[0]["lon"],
                "display_name": data[0]["display_name"]
            })
        return json.dumps({"error": "Location not found"})
```

### Example 4: Multi-Transport Server

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from fastapi import FastAPI
import uvicorn
import sys

server = Server("multi-transport-server")

# Define tools (same as before)
@server.tool()
async def example_tool(param: str) -> str:
    """Example tool"""
    return f"Result: {param}"

# Stdio mode
async def run_stdio():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

# HTTP mode
def run_http():
    app = FastAPI()

    @app.post("/mcp/tools/call")
    async def call_tool(name: str, arguments: dict):
        result = await server.call_tool(name, arguments)
        return {"result": result}

    @app.get("/mcp/tools/list")
    async def list_tools():
        tools = server.list_tools()
        return {"tools": tools}

    uvicorn.run(app, host="0.0.0.0", port=8000)

# Main entry point
if __name__ == "__main__":
    import asyncio

    if len(sys.argv) > 1 and sys.argv[1] == "--http":
        run_http()
    else:
        asyncio.run(run_stdio())
```

### Testing Your Server

```python
import pytest
from mcp.client import Client

@pytest.mark.asyncio
async def test_my_server():
    """Test custom MCP server"""

    # Start server
    client = Client()
    await client.connect({
        "type": "stdio",
        "command": "python",
        "args": ["my_server.py"]
    })

    # Test listing tools
    tools = await client.list_tools()
    assert len(tools) > 0

    # Test calling tool
    result = await client.call_tool(
        "my_tool",
        {"param": "test"}
    )
    assert "Result" in result

    # Cleanup
    await client.disconnect()
```

### Deployment

#### **Docker**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .

CMD ["python", "server.py", "--http"]
```

#### **Systemd Service**

```ini
[Unit]
Description=MCP Server
After=network.target

[Service]
Type=simple
User=mcpuser
WorkingDirectory=/opt/mcp-server
ExecStart=/usr/bin/python3 server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Best Practices and Use Cases

### Security Best Practices

1. **Input Validation**: Always validate and sanitize inputs
2. **Access Control**: Implement proper authentication/authorization
3. **Rate Limiting**: Prevent abuse with rate limits
4. **Audit Logging**: Log all tool calls for security auditing
5. **Principle of Least Privilege**: Grant minimal necessary permissions

### Performance Optimization

1. **Connection Pooling**: Reuse database connections
2. **Caching**: Cache frequently accessed data
3. **Async Operations**: Use async/await throughout
4. **Batching**: Batch multiple operations when possible
5. **Lazy Loading**: Load data only when needed

### Common Use Cases

#### 1. **Customer Support Agent**

```
MCP Servers:
- CRM (customer data)
- Ticketing system
- Email service
- Knowledge base

Agent capabilities:
- Look up customer history
- Create/update tickets
- Send responses
- Search documentation
```

#### 2. **Data Analysis Assistant**

```
MCP Servers:
- Database server
- Analytics API
- Visualization service
- Report generator

Agent capabilities:
- Query databases
- Generate analytics
- Create visualizations
- Export reports
```

#### 3. **DevOps Automation**

```
MCP Servers:
- Git operations
- CI/CD systems
- Cloud provider APIs
- Monitoring tools

Agent capabilities:
- Deploy applications
- Check system health
- Manage infrastructure
- Analyze logs
```

#### 4. **Content Management**

```
MCP Servers:
- CMS API
- File storage
- Image processing
- SEO tools

Agent capabilities:
- Create/edit content
- Process media
- Optimize for SEO
- Publish content
```

### Conclusion

Model Context Protocol represents a paradigm shift in how we build AI applications. By standardizing context management and tool calling, MCP enables more scalable, maintainable, and powerful AI systems.

Key takeaways:

- Choose the right approach (MCP, RAG, fine-tuning) based on your needs
- Leverage MCP's standardization for better interoperability
- Build reusable MCP servers for common integrations
- Integrate with existing tools like LangChain for maximum flexibility
- Follow security and performance best practices

As the MCP ecosystem grows, we'll see more tools, integrations, and patterns emerge. Stay updated with the official MCP documentation and community resources.

---

## Additional Resources

- [MCP Official Documentation](https://spec.modelcontextprotocol.io/)
- [MCP GitHub Repository](https://github.com/modelcontextprotocol)
- [Anthropic MCP Announcement](https://www.anthropic.com/news/model-context-protocol)
- [LangChain Documentation](https://python.langchain.com/)
- [MCP Community Examples](https://github.com/modelcontextprotocol/servers)

---

**Last Updated**: February 2026  
**Author**: AI Native Development Course
