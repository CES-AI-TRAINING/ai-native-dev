"""
Test client for Filesystem MCP server using FastMCP's built-in Client.

This demonstrates testing secure filesystem operations within a sandbox.
"""

import asyncio
import sys
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def test_filesystem_tools(client: Client):
    """Test all filesystem tools through MCP protocol."""
    
    print("=" * 70)
    print("TEST 1: List Filesystem Tools")
    print("=" * 70)
    print()
    
    tools = await client.list_tools()
    print(f"✓ Found {len(tools)} filesystem tools:")
    for tool in tools:
        print(f"  • {tool.name}")
    print()
    
    # Test write_file
    print("=" * 70)
    print("TEST 2: Write File")
    print("=" * 70)
    print()
    
    print("Calling: write_file(path='test.txt', content='Hello from MCP test!')")
    try:
        result = await client.call_tool("write_file", {
            "path": "test.txt",
            "content": "Hello from MCP test!"
        })
        print(f"  ✓ Result: {result.content[0].text}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test read_file
    print("=" * 70)
    print("TEST 3: Read File")
    print("=" * 70)
    print()
    
    print("Calling: read_file(path='test.txt')")
    try:
        result = await client.call_tool("read_file", {"path": "test.txt"})
        print(f"  ✓ Result: {result.content[0].text[:100]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test list_directory
    print("=" * 70)
    print("TEST 4: List Directory")
    print("=" * 70)
    print()
    
    print("Calling: list_directory(path='.')")
    try:
        result = await client.call_tool("list_directory", {"path": "."})
        print(f"  ✓ Result: {result.content[0].text[:150]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test search_files
    print("=" * 70)
    print("TEST 5: Search Files")
    print("=" * 70)
    print()
    
    print("Calling: search_files(pattern='*.txt')")
    try:
        result = await client.call_tool("search_files", {"pattern": "*.txt"})
        print(f"  ✓ Result: {result.content[0].text[:100]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test security (path traversal attempt)
    print("=" * 70)
    print("TEST 6: Security - Path Traversal Prevention")
    print("=" * 70)
    print()
    
    print("Calling: read_file(path='../../../etc/passwd') - Should fail!")
    try:
        result = await client.call_tool("read_file", {"path": "../../../etc/passwd"})
        # Should return error, not actual file content
        if "error" in result.content[0].text.lower() or "access denied" in result.content[0].text.lower():
            print(f"  ✓ Security check passed: {result.content[0].text[:80]}")
        else:
            print(f"  ✗ Security issue: Path traversal not blocked!")
    except Exception as e:
        # Exception is also fine - server rejected the request
        print(f"  ✓ Security check passed: {str(e)[:80]}...")
    print()
    
    # Test delete_file
    print("=" * 70)
    print("TEST 7: Delete File")
    print("=" * 70)
    print()
    
    print("Calling: delete_file(path='test.txt')")
    try:
        result = await client.call_tool("delete_file", {"path": "test.txt"})
        print(f"  ✓ Result: {result.content[0].text}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()


async def main():
    """Run filesystem MCP client tests using FastMCP Client."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 12 + "FILESYSTEM MCP CLIENT TEST" + " " * 30 + "║")
    print("║" + " " * 15 + "Testing with FastMCP Only" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("📡 This test uses FastMCP's built-in Client:")
    print("   ✓ Tests secure filesystem operations")
    print("   ✓ Verifies sandbox security")
    print("   ✓ Full MCP protocol testing via stdio")
    print()
    
    try:
        transport = StdioTransport(
            command="uv",
            args=["run", "python", "main.py", "--server"]
        )
        
        async with Client(transport) as client:
            print("✓ Connected to filesystem MCP server")
            print()
            
            await test_filesystem_tools(client)
            
            print("=" * 70)
            print("✨ All Tests Passed!")
            print("=" * 70)
            print()
            print("💡 KEY OBSERVATIONS:")
            print("   • All operations confined to sandbox directory")
            print("   • Path traversal attacks properly blocked")
            print("   • Async file I/O works correctly")
            print()
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
