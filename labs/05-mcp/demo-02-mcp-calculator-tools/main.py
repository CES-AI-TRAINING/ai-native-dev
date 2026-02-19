"""
Demo 02: MCP Calculator Tools

This demo shows how to create multiple tools in one MCP server:
- 6 calculator operations (add, subtract, multiply, divide, power, modulo)
- Parameter validation with type hints
- Error handling for edge cases
- Tool discovery and listing

Key Concepts:
- Multiple tool registration
- Type safety with JSON Schema
- Error handling patterns
- Tool organization
"""

from fastmcp import FastMCP

# ============================================================================
# CONFIGURATION
# ============================================================================

mcp = FastMCP("calculator-server")

# Print statements suppressed to avoid interfering with JSON-RPC protocol

# ============================================================================
# CALCULATOR TOOLS
# ============================================================================


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of a and b
    """
    result = a + b
    return result


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """
    Subtract b from a.
    
    Args:
        a: Number to subtract from
        b: Number to subtract
        
    Returns:
        Difference (a - b)
    """
    result = a - b
    return result


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of a and b
    """
    result = a * b
    return result


@mcp.tool()
def divide(a: float, b: float) -> float:
    """
    Divide a by b.
    
    Args:
        a: Numerator
        b: Denominator (must not be zero)
        
    Returns:
        Quotient (a / b)
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    
    result = a / b
    return result


@mcp.tool()
def power(base: float, exponent: float) -> float:
    """
    Raise base to the power of exponent.
    
    Args:
        base: The base number
        exponent: The exponent
        
    Returns:
        base raised to the power of exponent
    """
    result = base ** exponent
    return result


@mcp.tool()
def modulo(a: float, b: float) -> float:
    """
    Calculate the remainder of a divided by b.
    
    Args:
        a: Dividend
        b: Divisor (must not be zero)
        
    Returns:
        Remainder of a / b
        
    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot calculate modulo with zero divisor")
    
    result = a % b
    return result


# ============================================================================
# HELPER FUNCTIONS FOR DEMO (not MCP tools)
# ============================================================================
# ⚠️ IMPORTANT: These helper functions are used ONLY for educational demos.
# They do NOT go through the MCP protocol - they're direct Python calls.
# 
# To test actual MCP protocol communication, use: uv run python test_client.py
# ============================================================================

def _demo_add(a: float, b: float) -> float:
    """Demo version of add for educational purposes (NOT via MCP protocol)."""
    return a + b


def _demo_subtract(a: float, b: float) -> float:
    """Demo version of subtract for testing."""
    return a - b


def _demo_multiply(a: float, b: float) -> float:
    """Demo version of multiply for testing."""
    return a * b


def _demo_divide(a: float, b: float) -> float:
    """Demo version of divide for testing."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def _demo_power(base: float, exponent: float) -> float:
    """Demo version of power for testing."""
    return base ** exponent


def _demo_modulo(a: float, b: float) -> float:
    """Demo version of modulo for testing."""
    if b == 0:
        raise ValueError("Cannot calculate modulo with zero divisor")
    return a % b


# ============================================================================
# MAIN DEMO
# ============================================================================

def demo_calculator():
    """Run calculator tool demonstrations."""
    
    print("=" * 70)
    print("CALCULATOR SERVER: Tool Demonstrations")
    print("=" * 70)
    print()
    
    # Demonstrate each operation
    print("=" * 70)
    print("📋 AVAILABLE TOOLS")
    print("=" * 70)
    print()
    print("✓ Calculator server has 6 tools registered:")
    print()
    print("1. add(a, b) - Add two numbers")
    print("2. subtract(a, b) - Subtract b from a")
    print("3. multiply(a, b) - Multiply two numbers")
    print("4. divide(a, b) - Divide a by b (validates b ≠ 0)")
    print("5. power(base, exponent) - Raise base to exponent")
    print("6. modulo(a, b) - Calculate remainder of a/b")
    print()
    
    print("=" * 70)
    print("🧪 DEMONSTRATIONS")
    print("=" * 70)
    print()
    
    test_cases = [
        ("add", {"a": 15.5, "b": 24.3}, None),
        ("subtract", {"a": 100, "b": 37}, None),
        ("multiply", {"a": 7, "b": 8}, None),
        ("divide", {"a": 144, "b": 12}, None),
        ("divide", {"a": 10, "b": 0}, "Expected to fail: division by zero"),
        ("power", {"base": 2, "exponent": 10}, None),
        ("modulo", {"a": 17, "b": 5}, None),
        ("modulo", {"a": 10, "b": 0}, "Expected to fail: modulo by zero"),
    ]
    
    for i, (tool_name, args, note) in enumerate(test_cases, 1):
        print(f"Test {i}: {tool_name}({', '.join(f'{k}={v}' for k, v in args.items())})")
        if note:
            print(f"   Note: {note}")
        
        try:
            # Find the tool function
            tool_func = None
            if tool_name == "add":
                tool_func = _demo_add
            elif tool_name == "subtract":
                tool_func = _demo_subtract
            elif tool_name == "multiply":
                tool_func = _demo_multiply
            elif tool_name == "divide":
                tool_func = _demo_divide
            elif tool_name == "power":
                tool_func = _demo_power
            elif tool_name == "modulo":
                tool_func = _demo_modulo
            
            if tool_func:
                result = tool_func(**args)
                print(f"   ✓ Result: {result}")
            else:
                print(f"   ✗ Tool not found: {tool_name}")
                
        except ValueError as e:
            print(f"   ✗ Error (expected): {str(e)}")
        except Exception as e:
            print(f"   ✗ Unexpected error: {str(e)}")
        
        print()
    
    print("=" * 70)


def main():
    """Run the calculator demo."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "WELCOME TO MCP DEMO 02" + " " * 31 + "║")
    print("║" + " " * 16 + "Calculator Tools Server" + " " * 29 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("📚 This demo demonstrates:")
    print("   • Multiple tools in one MCP server")
    print("   • Parameter validation with type hints")
    print("   • Error handling for edge cases")
    print("   • Tool discovery and listing")
    print("   • JSON Schema automatic generation")
    print()
    
    # Run calculator demo
    demo_calculator()
    
    print("=" * 70)
    print("💡 KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("✓ One server can host multiple tools")
    print("✓ Type hints automatically create JSON Schema")
    print("✓ MCP validates parameters before calling tools")
    print("✓ Tools should handle errors gracefully")
    print("✓ Clear descriptions help LLMs choose right tools")
    print()
    
    print("=" * 70)
    print("🧪 UNDERSTANDING PARAMETER VALIDATION")
    print("=" * 70)
    print()
    print("When you define a tool like this:")
    print()
    print("  @mcp.tool()")
    print("  def add(a: float, b: float) -> float:")
    print('      """Add two numbers."""')
    print("      return a + b")
    print()
    print("FastMCP automatically generates this JSON Schema:")
    print()
    print("  {")
    print('    "type": "object",')
    print('    "properties": {')
    print('      "a": {"type": "number"},')
    print('      "b": {"type": "number"}')
    print("    },")
    print('    "required": ["a", "b"]')
    print("  }")
    print()
    print("This ensures:")
    print("  • Parameters must be numbers")
    print("  • Both a and b are required")
    print("  • Invalid inputs are rejected automatically")
    print()
    
    print("=" * 70)
    print("🔐 ERROR HANDLING BEST PRACTICES")
    print("=" * 70)
    print()
    print("1. Validate edge cases (like division by zero)")
    print("2. Raise descriptive errors (ValueError, TypeError, etc.)")
    print("3. Let MCP handle the error response to client")
    print("4. Don't catch errors unless you can recover")
    print()
    
    print("=" * 70)
    print("🎯 NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Try demo-03: External API integration")
    print("2. Try demo-04: Filesystem operations with security")
    print("3. Try demo-07: Use calculator in LangChain agent")
    print()
    
    print("💡 Exercise: Add these tools yourself:")
    print("   • square_root(x) - with negative number validation")
    print("   • percentage(value, percent) - calculate percentage")
    print("   • factorial(n) - with range validation (0-20)")
    print()
    
    print("=" * 70)
    print("✨ Demo Complete!")
    print("=" * 70)
    print()


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # Run as server only (for production use)
        # NOTE: Print statements are suppressed here because stdout MUST contain
        # ONLY JSON-RPC messages for the MCP protocol to work correctly.
        mcp.run()
    else:
        # Run demo (educational only - NO MCP protocol testing)
        print("⚠️  DEMO MODE: This shows tool behavior but does NOT test MCP protocol.")
        print("    To test actual MCP communication, run: uv run python test_client.py")
        print()
        main()
        print()
        print("=" * 70)
        print("🔧 OTHER RUN MODES")
        print("=" * 70)
        print()
        print("  Server Mode:  uv run python main.py --server")
        print("  MCP Test:     uv run python test_client.py")
        print()
        print("  Server Mode: Starts server and waits for client connections")
        print("  MCP Test:    Properly tests MCP protocol with client-server communication")
        print()
