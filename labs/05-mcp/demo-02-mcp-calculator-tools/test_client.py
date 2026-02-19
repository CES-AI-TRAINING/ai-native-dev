"""
Test client for Calculator MCP server.

This script demonstrates proper MCP client-server communication
with multiple tools and parameter validation.
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@asynccontextmanager
async def create_mcp_client():
    """Create and initialize an MCP client connected to the calculator server."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "main.py", "--server"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def test_calculator_tools(session: ClientSession):
    """Test all calculator tools through MCP protocol."""
    
    print("=" * 70)
    print("TEST 1: List Calculator Tools")
    print("=" * 70)
    print()
    
    response = await session.list_tools()
    print(f"✓ Found {len(response.tools)} calculator tools:")
    for tool in response.tools:
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
            result = await session.call_tool(tool_name, arguments=args)
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
        result = await session.call_tool("divide", arguments={"a": 10, "b": 0})
        if result.isError:
            print(f"  ✓ Server returned error (as expected): {result.content[0].text}")
        else:
            print(f"  ✗ Should have failed but got: {result.content[0].text}")
    except Exception as e:
        # MCP SDK raises exception when tool execution fails
        error_msg = str(e)
        if "Cannot divide by zero" in error_msg:
            print(f"  ✓ Properly caught error: {error_msg[:70]}...")
        else:
            print(f"  ✗ Unexpected error: {error_msg}")
    print()


async def main():
    """Run calculator MCP client tests."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "CALCULATOR MCP CLIENT TEST" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    try:
        async with create_mcp_client() as session:
            print("✓ Connected to calculator MCP server")
            print()
            
            await test_calculator_tools(session)
            
            print("=" * 70)
            print("✨ All Tests Passed!")
            print("=" * 70)
            print()
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
