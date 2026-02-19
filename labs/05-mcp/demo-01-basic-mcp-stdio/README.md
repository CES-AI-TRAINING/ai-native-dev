# Demo 01: Basic MCP Server with FastMCP 🚀

Learn the fundamentals of Model Context Protocol (MCP) with FastMCP - a Pythonic, beginner-friendly framework for building MCP servers.

## 🎯 What You'll Learn

- Understanding MCP protocol basics
- How to use FastMCP for simplified server creation
- Creating MCP tools with decorators
- Stdio transport for local communication
- Tool discovery and invocation
- Running servers in demo and production modes

## 📦 What's Inside

✅ **FastMCP Server** - Simple server setup with decorators  
✅ **Two Demo Tools** - Greet and get_server_info tools  
✅ **Stdio Transport** - Local process communication  
✅ **Demo Mode** - Educational walkthrough of concepts  
✅ **Server Mode** - Production-ready server (--server flag)  
✅ **Clear Documentation** - Extensive inline explanations

## 🚀 Quick Start

### 1. Install Dependencies

```bash
uv sync
```

### 2. Choose Your Mode

#### Option A: Educational Demo (Simplified)

Run the educational demo that explains concepts without MCP protocol overhead:

```bash
uv run python main.py
```

**What this does:**

- Shows educational content about MCP
- Calls tools directly (NOT through MCP protocol)
- Demonstrates what tools do
- No client-server communication

**⚠️ Important:** This mode does NOT test actual MCP protocol communication. It's purely educational.

#### Option B: Test Actual MCP Protocol (Recommended for Testing)

Test the real MCP client-server communication:

```bash
uv run python test_client.py
```

**What this does:**

- ✓ Starts MCP server as subprocess
- ✓ Creates MCP client
- ✓ Connects via stdio transport
- ✓ Sends JSON-RPC messages
- ✓ Discovers tools through `tools/list`
- ✓ Calls tools through `tools/call`
- ✓ Tests actual protocol behavior

**This is the proper way to test your MCP server!**

#### Option C: Run as Production Server

Run as an MCP server for external clients (like Claude Desktop):

```bash
uv run python main.py --server
```

The server will wait for connections on stdin/stdout.

## 📚 How It Works

### MCP Architecture

```
┌──────────────┐
│ MCP Client   │ (This process - parent)
└──────┬───────┘
       │ spawn
       ▼
┌──────────────┐
│ MCP Server   │ (Child process)
└──────────────┘
     ↕ stdin/stdout
  JSON-RPC Messages
```

### Tool Definition with FastMCP

```python
from fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def greet(name: str) -> str:
    """Greet someone by name"""
    return f"Hello, {name}! Welcome to MCP!"

# Run the server
mcp.run()
```

**Key Advantages of FastMCP:**

- ✅ No async/await required for simple tools
- ✅ Simple decorator pattern: `@mcp.tool()`
- ✅ Automatic JSON Schema generation from type hints
- ✅ One-line server startup: `mcp.run()`
- ✅ 60% less boilerplate code than raw MCP

### Message Flow

1. **Client → Server**: `tools/list` - Request available tools
2. **Server → Client**: Tool definitions (name, description, schema)
3. **Client → Server**: `tools/call` - Invoke specific tool
4. **Server → Client**: Tool result

## 🧠 Key Concepts

### What is MCP?

Model Context Protocol (MCP) is an open standard for connecting AI applications to data sources and tools. It provides:

- **Standardized Communication**: Common protocol for tool calling
- **Tool Discovery**: Dynamic listing of available capabilities
- **Type Safety**: JSON Schema for parameter validation
- **Transport Flexibility**: Stdio, HTTP, SSE support

### Stdio Transport

- **Pros**: Ultra-low latency, simple setup, automatic lifecycle
- **Cons**: Same machine only, not suitable for web/remote access
- **Best For**: Desktop apps, CLI tools, local development

### Tool Structure

Every MCP tool has:

1. **Name**: Unique identifier
2. **Description**: What the tool does
3. **Input Schema**: Parameter types and validation
4. **Handler**: Async function implementing the logic

## 📁 Project Structure

```
demo-01-basic-mcp-stdio/
├── .python-version      # Python 3.12
├── .gitignore          # Python/UV ignores
├── pyproject.toml      # Dependencies (fastmcp + mcp)
├── README.md           # This file
├── main.py             # FastMCP server with demo mode
└── test_client.py      # Proper MCP client test (NEW!)
```

## 🧪 Testing Your MCP Server

### Understanding the Difference

**❌ Demo Mode (`python main.py`)**

- Calls helper functions directly
- NO MCP protocol involved
- Educational purposes only
- Shows tool behavior without protocol overhead

**✅ Protocol Test (`python test_client.py`)**

- Real MCP client-server communication
- Server runs as subprocess
- Client connects via stdio
- JSON-RPC messages exchanged
- Tests actual MCP behavior

### Running Protocol Tests

```bash
# Test with proper MCP client
uv run python test_client.py
```

**What gets tested:**

1. ✓ Server startup in subprocess
2. ✓ Client connection via stdio
3. ✓ Tool discovery (`tools/list` request)
4. ✓ Tool invocation (`tools/call` request)
5. ✓ JSON-RPC message format
6. ✓ Result serialization/deserialization

### Expected Output

```
╔════════════════════════════════════╗
║         MCP CLIENT TEST            ║
║    Testing Actual Protocol         ║
╚════════════════════════════════════╝

✓ Connected to MCP server

TEST 1: List Available Tools
---
✓ Found 2 tools:
  • greet
  • get_server_info

TEST 2: Call greet Tool
---
Calling: greet(name='Alice')
✓ Result: Hello, Alice! Welcome to MCP...

✨ All Tests Passed!
```

### Viewing JSON-RPC Messages

To see the actual JSON-RPC messages, you can modify `test_client.py` to add logging or run with debug mode.

### Writing Custom Tests

You can extend `test_client.py` to add more test cases:

```python
async def test_custom(session: ClientSession):
    """Your custom test."""
    result = await session.call_tool("greet", arguments={"name": "Custom"})
    assert "Custom" in result.content[0].text
    print("✓ Custom test passed")
```

## 🔧 Troubleshooting

### Import Error: No module named 'fastmcp'

```bash
# Make sure dependencies are installed
uv sync

# Verify FastMCP is installed
uv run python -c "import fastmcp; print('FastMCP OK')"
```

### Running as MCP Server

To run as an actual MCP server (for Claude Desktop or other MCP clients):

```bash
uv run python main.py --server
```

### Claude Desktop Integration

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "demo-server": {
      "command": "uv",
      "args": ["run", "python", "main.py", "--server"],
      "cwd": "/path/to/demo-01-basic-mcp-stdio"
    }
  }
}
```

## 🎓 Learning Notes

### Why FastMCP?

**FastMCP** simplifies MCP development compared to the raw protocol:

| Feature         | Raw MCP            | FastMCP              |
| --------------- | ------------------ | -------------------- |
| Server setup    | 15+ lines          | 1 line               |
| Tool definition | async + decorators | Simple decorator     |
| Type validation | Manual JSON Schema | Automatic from hints |
| Boilerplate     | High               | Minimal              |
| Learning curve  | Steep              | Gentle               |

### Why Stdio?

Stdio (standard input/output) is the simplest MCP transport because:

- No network configuration needed
- Automatic process management
- Built-in security (process isolation)
- Perfect for local tools and desktop apps

### JSON-RPC 2.0

MCP uses JSON-RPC 2.0 for message format:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "echo",
    "arguments": { "message": "Hello" }
  }
}
```

### Tool Decorator

The `@server.tool()` decorator:

- Automatically registers the function
- Extracts type hints for schema generation
- Handles serialization/deserialization
- Manages error responses

## 📚 Next Steps

1. **Demo 02** - Multiple tools and parameter validation
2. **Demo 03** - External API integration
3. **Demo 04** - Filesystem operations with security
4. **Demo 06** - HTTP transport for web applications
5. **Demo 07** - Integration with LangChain

## 🤝 Need Help?

- Check the [MCP Documentation](https://spec.modelcontextprotocol.io/)
- Review the error messages - they're designed to be helpful
- Try modifying the echo tool to understand the flow

---

**Happy Learning! 🚀**
