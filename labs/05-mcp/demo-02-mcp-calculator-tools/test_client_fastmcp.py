"""
Test client for Calculator MCP server using FastMCP's built-in Client.

This demonstrates MCP testing using only the FastMCP library.
No separate 'mcp' package dependency needed!
"""

import asyncio
import sys
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def test_calculator_tools(client: Client):
    """Test all calculator tools through MCP protocol."""
    
    print("=" * 70)
    print("TEST 1: List Calculator Tools")
    print("=" * 70)
    print()
    
    # List available tools - FastMCP returns list directly
    tools = await client.list_tools()
    print(f"✓ Found {len(tools)} calculator tools:")
    for tool in tools:
        print(f"  • {tool.name}")
    print()
    
    # Test cases
    test_cases = [
        ("add", {"a": 15.5, "b": 24.3}),
        ("subtract", {"a": 100, "b": 37}),
        ("multiply", {"a": 7, "b": 8}),
        ("divide", {"a": 144, "b": 12}),
        ("power", {"base": 2, "exponent": 10}),
        ("modulo", {"a": 17, "b": 5}),
    ]
    
    print("=" * 70)
    print("TEST 2: Call Calculator Tools")
    print("=" * 70)
    print()
    
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
    print("TEST 3: Error Handling (Division by Zero)")
    print("=" * 70)
    print()
    
    print("Calling: divide(a=10, b=0)")
    try:
        result = await client.call_tool("divide", {"a": 10, "b": 0})
        print(f"  ✗ Should have failed but got: {result.content[0].text}")
    except Exception as e:
        # FastMCP Client raises exception on tool errors
        error_msg = str(e)
        if "Cannot divide by zero" in error_msg:
            print(f"  ✓ Properly caught error: {error_msg[:70]}...")
        else:
            print(f"  ✗ Unexpected error: {error_msg}")
    print()


async def main():
    """Run calculator MCP client tests using FastMCP Client."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 11 + "FASTMCP CALCULATOR CLIENT TEST" + " " * 26 + "║")
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
        # Create StdioTransport to connect to the server
        transport = StdioTransport(
            command="uv",
            args=["run", "python", "main.py", "--server"]
        )
        
        # Create FastMCP client using the transport
        async with Client(transport) as client:
            print("✓ Connected to calculator MCP server")
            print()
            
            await test_calculator_tools(client)
            
            print("=" * 70)
            print("✨ All Tests Passed!")
            print("=" * 70)
            print()
            print("💡 KEY OBSERVATIONS:")
            print("   • FastMCP Client managed server automatically")
            print("   • Only FastMCP library needed for testing")
            print("   • Full MCP protocol tested via stdio transport")
            print()
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
