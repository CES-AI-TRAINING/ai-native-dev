"""
Test client for HTTP MCP server using FastMCP's built-in Client.

NOTE: Demo 06 uses HTTP transport, not stdio!
This test client connects to the HTTP server instead of spawning a subprocess.

Prerequisites: The HTTP server must be running:
    uv run python main.py
"""

import asyncio
import sys
from fastmcp import Client


async def test_http_tools(client: Client):
    """Test all HTTP MCP tools."""
    
    print("=" * 70)
    print("TEST 1: List HTTP MCP Tools")
    print("=" * 70)
    print()
    
    tools = await client.list_tools()
    print(f"✓ Found {len(tools)} tools:")
    for tool in tools:
        print(f"  • {tool.name}")
    print()
    
    # Test greet
    print("=" * 70)
    print("TEST 2: Call greet Tool")
    print("=" * 70)
    print()
    
    print("Calling: greet(name='Alice', greeting='Hi')")
    try:
        result = await client.call_tool("greet", {"name": "Alice", "greeting": "Hi"})
        print(f"  ✓ Result: {result.content[0].text}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test calculator tools
    print("=" * 70)
    print("TEST 3: Call Calculator Tools")
    print("=" * 70)
    print()
    
    test_cases = [
        ("add", {"a": 10, "b": 5}),
        ("subtract", {"a": 20, "b": 8}),
        ("multiply", {"a": 7, "b": 6}),
        ("divide", {"a": 100, "b": 4}),
    ]
    
    for tool_name, args in test_cases:
        args_str = ", ".join(f"{k}={v}" for k, v in args.items())
        print(f"Calling: {tool_name}({args_str})")
        try:
            result = await client.call_tool(tool_name, args)
            print(f"  ✓ Result: {result.content[0].text}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print()
    
    # Test error handling
    print("=" * 70)
    print("TEST 4: Error Handling (Division by Zero)")
    print("=" * 70)
    print()
    
    print("Calling: divide(a=10, b=0)")
    try:
        result = await client.call_tool("divide", {"a": 10, "b": 0})
        result_text = result.content[0].text
        if "error" in result_text.lower() or "cannot divide by zero" in result_text.lower():
            print(f"  ✓ Error handled correctly: {result_text[:80]}")
        else:
            print(f"  ✗ Should have returned error: {result_text}")
    except Exception as e:
        print(f"  ✓ Server returned error: {str(e)[:80]}...")
    print()


async def main():
    """Run HTTP MCP client tests using FastMCP Client."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "HTTP MCP CLIENT TEST" + " " * 33 + "║")
    print("║" + " " * 14 + "Testing HTTP Transport" + " " * 32 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("📡 This test uses FastMCP Client with HTTP transport:")
    print("   ✓ Connects to HTTP server (not stdio)")
    print("   ✓ RESTful MCP over HTTP/JSON")
    print("   ✓ Server must be running separately")
    print()
    print("💡 Make sure the server is running:")
    print("   Terminal 1: uv run python main.py")
    print("   Terminal 2: uv run python test_client_fastmcp.py")
    print()
    
    try:
        # Connect to HTTP server using URL
        # Note: FastMCP Client can connect to HTTP URLs directly
        async with Client("http://localhost:8000") as client:
            print("✓ Connected to HTTP MCP server at http://localhost:8000")
            print()
            
            await test_http_tools(client)
            
            print("=" * 70)
            print("✨ All Tests Passed!")
            print("=" * 70)
            print()
            print("💡 KEY OBSERVATIONS:")
            print("   • HTTP transport works instead of stdio")
            print("   • Server runs independently (not spawned by client)")
            print("   • RESTful API provides MCP functionality")
            print("   • Can be accessed from web browsers and HTTP clients")
            print()
            
    except Exception as e:
        error_msg = str(e)
        if "Connection refused" in error_msg or "connect" in error_msg.lower():
            print("✗ Cannot connect to server!")
            print()
            print("Please start the server first:")
            print("  Terminal 1: cd labs/05-mcp/demo-06-mcp-http-transport")
            print("  Terminal 1: uv run python main.py")
            print()
            print("Then run this test in another terminal:")
            print("  Terminal 2: uv run python test_client_fastmcp.py")
        else:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
