"""
Test client for MCP server using FastMCP's built-in Client.

This script demonstrates MCP testing using only the FastMCP library:
1. Uses FastMCP's Client class instead of the mcp library
2. Can connect in-process or via subprocess
3. Simpler API and fewer dependencies
4. Same protocol testing capabilities
"""

import asyncio
import sys
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def test_list_tools(client: Client):
    """
    Test tool listing functionality.
    
    Args:
        client: Active FastMCP client
    """
    print("=" * 70)
    print("TEST 1: List Available Tools")
    print("=" * 70)
    print()
    
    # List available tools - FastMCP returns list directly
    tools = await client.list_tools()
    
    print(f"✓ Found {len(tools)} tools:")
    print()
    
    for tool in tools:
        print(f"Tool: {tool.name}")
        print(f"  Description: {tool.description}")
        print(f"  Input Schema: {tool.inputSchema}")
        print()


async def test_call_greet(client: Client):
    """
    Test calling the greet tool.
    
    Args:
        client: Active FastMCP client
    """
    print("=" * 70)
    print("TEST 2: Call greet Tool")
    print("=" * 70)
    print()
    
    print("Calling: greet(name='Alice')")
    
    result = await client.call_tool("greet", {"name": "Alice"})
    
    print(f"✓ Result: {result.content[0].text}")
    print()


async def test_call_get_server_info(client: Client):
    """
    Test calling the get_server_info tool.
    
    Args:
        client: Active FastMCP client
    """
    print("=" * 70)
    print("TEST 3: Call get_server_info Tool")
    print("=" * 70)
    print()
    
    print("Calling: get_server_info()")
    
    result = await client.call_tool("get_server_info", {})
    
    print(f"✓ Result: {result.content[0].text}")
    print()


async def test_multiple_calls(client: Client):
    """
    Test multiple sequential tool calls.
    
    Args:
        client: Active FastMCP client
    """
    print("=" * 70)
    print("TEST 4: Multiple Sequential Calls")
    print("=" * 70)
    print()
    
    names = ["Bob", "Charlie", "Diana"]
    
    for name in names:
        print(f"Calling: greet(name='{name}')")
        result = await client.call_tool("greet", {"name": name})
        print(f"  ✓ {result.content[0].text}")
    
    print()


async def main():
    """Run all MCP client tests using FastMCP Client."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 16 + "FASTMCP CLIENT TEST" + " " * 29 + "║")
    print("║" + " " * 15 + "Testing with FastMCP Only" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("📡 This test uses FastMCP's built-in Client:")
    print("   ✓ Single library for both server and client")
    print("   ✓ No need for separate 'mcp' package")
    print("   ✓ Simpler API and configuration")
    print("   ✓ Full MCP protocol testing")
    print()
    
    try:
        # Create StdioTransport to connect to the server via subprocess
        transport = StdioTransport(
            command="uv",
            args=["run", "python", "main.py", "--server"]
        )
        
        # Create FastMCP client using the transport
        async with Client(transport) as client:
            print("✓ Connected to MCP server")
            print()
            
            # Run tests
            await test_list_tools(client)
            await test_call_greet(client)
            await test_call_get_server_info(client)
            await test_multiple_calls(client)
            
            print("=" * 70)
            print("✨ All Tests Passed!")
            print("=" * 70)
            print()
            print("💡 KEY OBSERVATIONS:")
            print("   • FastMCP Client automatically managed server process")
            print("   • Client discovered tools via tools/list")
            print("   • Tool calls went through JSON-RPC protocol")
            print("   • Only FastMCP library needed (no 'mcp' dependency)")
            print()
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
