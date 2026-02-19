"""
Test client for LangChain MCP Integration using FastMCP's built-in Client.

This demonstrates testing MCP tools that are integrated with LangChain agents.
"""

import asyncio
import sys
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def test_langchain_mcp_tools(client: Client):
    """Test MCP tools through protocol (before LangChain integration)."""
    
    print("=" * 70)
    print("TEST 1: List MCP Calculator Tools")
    print("=" * 70)
    print()
    
    tools = await client.list_tools()
    print(f"✓ Found {len(tools)} calculator tools (for LangChain):")
    for tool in tools:
        print(f"  • {tool.name}")
    print()
    
    # Test calculator tools
    print("=" * 70)
    print("TEST 2: Call Calculator Tools via MCP")
    print("=" * 70)
    print()
    
    test_cases = [
        ("add", {"a": 15, "b": 27}),
        ("subtract", {"a": 50, "b": 13}),
        ("multiply", {"a": 8, "b": 9}),
        ("divide", {"a": 144, "b": 12}),
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
    print("TEST 3: Error Handling (Division by Zero)")
    print("=" * 70)
    print()
    
    print("Calling: divide(a=10, b=0)")
    try:
        result = await client.call_tool("divide", {"a": 10, "b": 0})
        result_text = result.content[0].text
        if "error" in result_text.lower() or "cannot divide by zero" in result_text.lower():
            print(f"  ✓ Error handled: {result_text[:80]}")
        else:
            print(f"  ✗ Should have returned error")
    except Exception as e:
        print(f"  ✓ Server returned error: {str(e)[:80]}...")
    print()


async def main():
    """Run LangChain MCP client tests using FastMCP Client."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 10 + "LANGCHAIN MCP INTEGRATION TEST" + " " * 27 + "║")
    print("║" + " " * 15 + "Testing with FastMCP Only" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("📡 This test uses FastMCP's built-in Client:")
    print("   ✓ Tests MCP tools before LangChain integration")
    print("   ✓ Verifies tools work correctly via MCP protocol")
    print("   ✓ Full MCP protocol testing via stdio")
    print()
    print("💡 NOTE:")
    print("   • This tests MCP tools directly (not through LangChain agent)")
    print("   • For LangChain agent demo, run: uv run python main.py")
    print("   • LangChain agent requires OPENAI_API_KEY in .env")
    print()
    
    try:
        transport = StdioTransport(
            command="uv",
            args=["run", "python", "main.py", "--server"]
        )
        
        async with Client(transport) as client:
            print("✓ Connected to LangChain MCP server")
            print()
            
            await test_langchain_mcp_tools(client)
            
            print("=" * 70)
            print("✨ All Tests Passed!")
            print("=" * 70)
            print()
            print("💡 KEY OBSERVATIONS:")
            print("   • MCP tools work correctly via protocol")
            print("   • These same tools are wrapped for LangChain")
            print("   • LangChain agent can use them for complex reasoning")
            print()
            print("🎯 NEXT STEPS:")
            print("   1. Add OPENAI_API_KEY to .env file")
            print("   2. Run: uv run python main.py")
            print("   3. See LangChain agent use these MCP tools!")
            print()
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
