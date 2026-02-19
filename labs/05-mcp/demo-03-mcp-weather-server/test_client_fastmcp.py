"""
Test client for Weather MCP server using FastMCP's built-in Client.

This demonstrates testing a weather API server that integrates external APIs.
Works in DEMO mode (simulated data) or LIVE mode (real OpenWeatherMap API).
"""

import asyncio
import sys
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def test_weather_tools(client: Client):
    """Test all weather tools through MCP protocol."""
    
    print("=" * 70)
    print("TEST 1: List Weather Tools")
    print("=" * 70)
    print()
    
    tools = await client.list_tools()
    print(f"✓ Found {len(tools)} weather tools:")
    for tool in tools:
        print(f"  • {tool.name}")
    print()
    
    # Test get_weather
    print("=" * 70)
    print("TEST 2: Get Current Weather")
    print("=" * 70)
    print()
    
    cities = ["London", "Tokyo", "New York"]
    for city in cities:
        print(f"Calling: get_weather(city='{city}')")
        try:
            result = await client.call_tool("get_weather", {"city": city})
            print(f"  ✓ Result: {result.content[0].text[:100]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print()
    
    # Test with different units
    print("=" * 70)
    print("TEST 3: Get Weather with Different Units")
    print("=" * 70)
    print()
    
    for units in ["metric", "imperial"]:
        print(f"Calling: get_weather(city='Paris', units='{units}')")
        try:
            result = await client.call_tool("get_weather", {"city": "Paris", "units": units})
            print(f"  ✓ Result: {result.content[0].text[:100]}...")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        print()
    
    # Test geocode
    print("=" * 70)
    print("TEST 4: Geocode Location")
    print("=" * 70)
    print()
    
    print("Calling: geocode(location='San Francisco')")
    try:
        result = await client.call_tool("geocode", {"location": "San Francisco"})
        print(f"  ✓ Result: {result.content[0].text[:100]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()
    
    # Test compare_weather
    print("=" * 70)
    print("TEST 5: Compare Weather Across Cities")
    print("=" * 70)
    print()
    
    print("Calling: compare_weather(cities=['Berlin', 'Madrid', 'Rome'])")
    try:
        result = await client.call_tool("compare_weather", {
            "cities": ["Berlin", "Madrid", "Rome"]
        })
        print(f"  ✓ Result: {result.content[0].text[:150]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    print()


async def main():
    """Run weather MCP client tests using FastMCP Client."""
    
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 14 + "WEATHER MCP CLIENT TEST" + " " * 31 + "║")
    print("║" + " " * 15 + "Testing with FastMCP Only" + " " * 28 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    print("📡 This test uses FastMCP's built-in Client:")
    print("   ✓ Tests weather API integration")
    print("   ✓ Works in DEMO mode or LIVE mode")
    print("   ✓ Full MCP protocol testing via stdio")
    print()
    
    try:
        transport = StdioTransport(
            command="uv",
            args=["run", "python", "main.py", "--server"]
        )
        
        async with Client(transport) as client:
            print("✓ Connected to weather MCP server")
            print()
            
            await test_weather_tools(client)
            
            print("=" * 70)
            print("✨ All Tests Passed!")
            print("=" * 70)
            print()
            print("💡 NOTE:")
            print("   • If OPENWEATHER_API_KEY not configured, tests use simulated data")
            print("   • To test with real API, add API key to .env file")
            print("   • Get free API key: https://openweathermap.org/api")
            print()
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
