"""
Test client for MCP server using the mcp library.

This script demonstrates proper MCP client-server communication:
1. Spawns the MCP server as a subprocess
2. Creates an MCP client
3. Connects via stdio transport
4. Lists available tools
5. Calls tools with parameters
6. Shows actual JSON-RPC communication
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def create_mcp_client():
    """
    Create and initialize an MCP client connected to the demo server.
    
    Yields:
        Tuple of (read, write, session) for interacting with the server
    """
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "main.py", "--server"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            yield session


async def test_list_tools(session: ClientSession):
    """
    Test tool listing functionality.
    
    Args:
        session: Active MCP client session
    """
    print("=" * 70)
    print("TEST 1: List Available Tools")
    print("=" * 70)
    print()
    
    # List available tools
    response = await session.list_tools()
    
    print(f"✓ Found {len(response.tools)} tools:")
    print()
    
    for tool in response.tools:
        print(f"Tool: {tool.name}")
        print(f"  Description: {tool.description}")
        print(f"  Input Schema: {tool.inputSchema}")
        print()


async def test_call_greet(session: ClientSession):
    """
    Test calling the greet tool.
    
    Args:
        session: Active MCP client session
    """
    print("=" * 70)
    print("TEST 2: Call greet Tool")
    print("=" * 70)
    print()
    
    print("Calling: greet(name='Alice')")
    
    result = await session.call_tool("greet", arguments={"name": "Alice"})
    
    print(f"✓ Result: {result.content[0].text}")
    print()


async def test_call_get_server_info(session: ClientSession):
    """
    Test calling the get_server_info tool.
    
    Args:
        session: Active MCP client session
    """
    print("=" * 70)
    print("TEST 3: Call get_server_info Tool")
    print("=" * 70)
    print()
    
    print("Calling: get_server_info()")
    
    result = await session.call_tool("get_server_info", arguments={})
    
    print(f"✓ Result: {result.content[0].text}")
    print()


async def test_multiple_calls(session: ClientSession):
    """
    Test multiple sequential tool calls.
    
    Args:
        session: Active MCP client session
    """
    print("=" * 70)
    print("TEST 4: Multiple Sequential Calls")
    print("=" * 70)
    print()
    
    names = ["Bob", "Charlie", "Diana"]
    
    for name in names:
        print(f"Calling: greet(name='{name}')")
        result = await session.call_tool("greet", arguments={"name": name})
        print(f"  ✓ {result.content[0].text}")
    
    print()


async def main():
    """Run all MCP client tests."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "MCP CLIENT TEST" + " " * 33 + "║")
    print("║" + " " * 15 + "Testing Actual Protocol" + " " * 30 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("📡 This test demonstrates REAL MCP communication:")
    print("   ✓ Server runs as subprocess")
    print("   ✓ Client connects via stdio")
    print("   ✓ JSON-RPC messages over stdin/stdout")
    print("   ✓ Tool discovery through protocol")
    print("   ✓ Tool invocation through protocol")
    print()
    
    try:
        async with create_mcp_client() as session:
            print("✓ Connected to MCP server")
            print()
            
            # Run tests
            await test_list_tools(session)
            await test_call_greet(session)
            await test_call_get_server_info(session)
            await test_multiple_calls(session)
            
            print("=" * 70)
            print("✨ All Tests Passed!")
            print("=" * 70)
            print()
            print("💡 KEY OBSERVATIONS:")
            print("   • Server ran as separate process")
            print("   • Client discovered tools via tools/list")
            print("   • Tool calls went through JSON-RPC protocol")
            print("   • Results were properly serialized/deserialized")
            print()
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
