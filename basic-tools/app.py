"""
OpenAI Agents SDK - Comprehensive Multi-Tool Usage Example
Main application orchestrator and demo scenarios

IMPORTANT: This implementation uses ONLY function tools and agents as tools.
NO MCP (Model Context Protocol) is implemented here.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from agents import Agent, Runner, RunContextWrapper
from agents.model_settings import ModelSettings

# Import tools and data models
from _tools import (
    UserPreferences,
    calculate,
    get_weather,
    read_from_file,
    risky_operation,
    save_to_file,
    web_search,
)

# Import agents and conditional functions
from _agents import (
    content_writer_agent,
    data_analyzer_agent,
    research_agent,
    search_enabled,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")


# ============================================================================
# ORCHESTRATOR AGENT
# ============================================================================

def create_orchestrator_agent() -> Agent:
    """Create the main orchestrator agent with all tools.

    Returns:
        Configured orchestrator agent with function tools and agent tools
    """
    logger.info("Creating orchestrator agent...")

    agent = Agent(
        name="OrchestratorAgent",
        instructions=(
            "You are a helpful AI assistant with access to multiple specialized tools and agents. "
            "You can:\n"
            "- Perform calculations\n"
            "- Check weather information\n"
            "- Search the web for current information\n"
            "- Analyze data using specialized agents\n"
            "- Write content using specialized agents\n"
            "- Conduct research using specialized agents\n"
            "- Save and read files\n\n"
            "When the user asks a question:\n"
            "1. Determine which tools or agents are needed\n"
            "2. Use them efficiently (call multiple tools in parallel when possible)\n"
            "3. Provide a comprehensive answer\n\n"
            "Always be helpful, accurate, and concise."
        ),
        tools=[
            # Function tools
            calculate,
            get_weather,
            save_to_file,
            read_from_file,
            risky_operation,
            web_search,

            # Agent tools (agents used as tools)
            data_analyzer_agent.as_tool(
                tool_name="analyze_data",
                tool_description="Analyze data and provide statistical insights",
                is_enabled=True
            ),
            content_writer_agent.as_tool(
                tool_name="write_content",
                tool_description="Write professional content, articles, or documentation",
                is_enabled=True
            ),
            research_agent.as_tool(
                tool_name="research_topic",
                tool_description="Research a topic using web search and compile findings",
                is_enabled=search_enabled
            ),
        ],
        model_settings=ModelSettings(
            temperature=0.7,
            max_tokens=2000,
        )
    )

    logger.info(f"✓ Orchestrator agent created with {len(agent.tools)} tools")
    return agent


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

async def demo_basic_calculation():
    """Demo: Basic calculation using function tool."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Calculation (Function Tool)")
    print("="*70)
    logger.info("Starting Demo 1: Basic Calculation")

    try:
        agent = create_orchestrator_agent()
        logger.info("Agent created, executing query...")

        result = await Runner.run(
            agent,
            "What is 15 multiplied by 7?"
        )

        logger.info("Demo 1 completed successfully")
        print(f"\n✓ Result: {result.final_output}")
        return True

    except Exception as e:
        logger.error(f"Demo 1 failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return False


async def demo_weather_check():
    """Demo: Weather check with structured data."""
    print("\n" + "="*70)
    print("DEMO 2: Weather Information (TypedDict)")
    print("="*70)
    logger.info("Starting Demo 2: Weather Check")

    try:
        agent = create_orchestrator_agent()
        logger.info("Agent created, executing query...")

        result = await Runner.run(
            agent,
            "What's the weather in San Francisco (lat: 37.7749, long: -122.4194)?"
        )

        logger.info("Demo 2 completed successfully")
        print(f"\n✓ Result: {result.final_output}")
        return True

    except Exception as e:
        logger.error(f"Demo 2 failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return False


async def demo_web_search():
    """Demo: Web search using Tavily."""
    print("\n" + "="*70)
    print("DEMO 3: Web Search with Tavily (Function Tool)")
    print("="*70)
    logger.info("Starting Demo 3: Web Search")

    try:
        agent = create_orchestrator_agent()

        # Create context with preferences
        context = RunContextWrapper(UserPreferences(
            search_enabled=True,
            verbose=True
        ))
        logger.info("Context created with search enabled")

        result = await Runner.run(
            agent,
            "Search for the latest news about Python programming in 2024",
            context=context.context
        )

        logger.info("Demo 3 completed successfully")
        print(f"\n✓ Result: {result.final_output}")
        return True

    except Exception as e:
        logger.error(f"Demo 3 failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return False


async def demo_multiple_tools():
    """Demo: Using multiple tools in parallel."""
    print("\n" + "="*70)
    print("DEMO 4: Multiple Tools in Parallel")
    print("="*70)
    logger.info("Starting Demo 4: Multiple Tools")

    try:
        agent = create_orchestrator_agent()
        logger.info("Agent created, executing query with multiple tools...")

        result = await Runner.run(
            agent,
            "Calculate 25 + 17 and also tell me the weather in Tokyo (lat: 35.6762, long: 139.6503)"
        )

        logger.info("Demo 4 completed successfully")
        print(f"\n✓ Result: {result.final_output}")
        return True

    except Exception as e:
        logger.error(f"Demo 4 failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return False


async def demo_agent_as_tool():
    """Demo: Using specialized agents as tools."""
    print("\n" + "="*70)
    print("DEMO 5: Agent as Tool - Data Analysis")
    print("="*70)
    logger.info("Starting Demo 5: Agent as Tool")

    try:
        agent = create_orchestrator_agent()
        logger.info("Agent created, calling sub-agent for data analysis...")

        result = await Runner.run(
            agent,
            "Analyze this dataset and provide insights: [10, 25, 30, 15, 40, 35, 20]"
        )

        logger.info("Demo 5 completed successfully")
        print(f"\n✓ Result: {result.final_output}")
        return True

    except Exception as e:
        logger.error(f"Demo 5 failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return False


async def demo_file_operations():
    """Demo: File operations with context."""
    print("\n" + "="*70)
    print("DEMO 6: File Operations (with Context)")
    print("="*70)
    logger.info("Starting Demo 6: File Operations")

    try:
        agent = create_orchestrator_agent()

        context = RunContextWrapper(UserPreferences(verbose=True))
        logger.info("Context created with verbose mode")

        result = await Runner.run(
            agent,
            "Save the text 'Hello, OpenAI Agents SDK!' to a file called 'test_output.txt'",
            context=context.context
        )

        logger.info("Demo 6 completed successfully")
        print(f"\n✓ Result: {result.final_output}")
        return True

    except Exception as e:
        logger.error(f"Demo 6 failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return False


async def demo_error_handling():
    """Demo: Error handling in tools."""
    print("\n" + "="*70)
    print("DEMO 7: Error Handling (Custom Error Function)")
    print("="*70)
    logger.info("Starting Demo 7: Error Handling")

    try:
        agent = create_orchestrator_agent()
        logger.info("Agent created, executing query that will trigger error...")

        result = await Runner.run(
            agent,
            "Perform a risky operation with value -5"
        )

        logger.info("Demo 7 completed (error was handled gracefully)")
        print(f"\n✓ Result: {result.final_output}")
        return True

    except Exception as e:
        logger.error(f"Demo 7 failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return False


async def demo_conditional_tools():
    """Demo: Conditional tool enabling."""
    print("\n" + "="*70)
    print("DEMO 8: Conditional Tool Enabling")
    print("="*70)
    logger.info("Starting Demo 8: Conditional Tools")

    try:
        agent = create_orchestrator_agent()

        # Test with search disabled
        context = RunContextWrapper(UserPreferences(search_enabled=False))
        logger.info("Context created with search DISABLED")

        result = await Runner.run(
            agent,
            "Can you research the latest AI developments?",
            context=context.context
        )

        logger.info("Demo 8 completed successfully")
        print(f"\n✓ Result (search disabled): {result.final_output}")
        return True

    except Exception as e:
        logger.error(f"Demo 8 failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return False


# ============================================================================
# VALIDATION
# ============================================================================

def validate_environment() -> bool:
    """Validate environment setup and API keys.

    Returns:
        True if validation passed, False otherwise
    """
    logger.info("Validating environment...")

    issues = []

    # Check OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        issues.append("OPENAI_API_KEY not found")
    else:
        logger.info("✓ OPENAI_API_KEY found")

    # Check TAVILY_API_KEY
    if not os.getenv("TAVILY_API_KEY"):
        issues.append("TAVILY_API_KEY not found")
    else:
        logger.info("✓ TAVILY_API_KEY found")

    if issues:
        logger.error(f"Environment validation failed: {', '.join(issues)}")
        print("\n" + "="*70)
        print("ENVIRONMENT VALIDATION FAILED")
        print("="*70)
        for issue in issues:
            print(f"✗ {issue}")
        print("\nPlease set the required API keys in your .env file")
        print("="*70)
        return False

    logger.info("✓ Environment validation passed")
    return True


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Run all demo scenarios."""

    print("\n" + "="*70)
    print("OpenAI Agents SDK - Multiple Tool Usage Demonstration")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    logger.info("="*50)
    logger.info("APPLICATION STARTED")
    logger.info("="*50)

    # Validate environment
    if not validate_environment():
        return

    print("\n✓ All API keys validated")
    print("\nNOTE: This demo uses:")
    print("  • Function Tools (@function_tool decorator)")
    print("  • Agents as Tools (agent.as_tool())")
    print("  • NO MCP (Model Context Protocol)")
    print("  • Modular structure: _tools.py, _agents.py, app.py")
    print("\n" + "="*70)

    # Track demo results
    results = {}

    # Run all demos
    demos = [
        ("Basic Calculation", demo_basic_calculation),
        ("Weather Check", demo_weather_check),
        ("Web Search", demo_web_search),
        ("Multiple Tools", demo_multiple_tools),
        ("Agent as Tool", demo_agent_as_tool),
        ("File Operations", demo_file_operations),
        ("Error Handling", demo_error_handling),
        ("Conditional Tools", demo_conditional_tools),
    ]

    for name, demo_func in demos:
        try:
            logger.info(f"Starting demo: {name}")
            success = await demo_func()
            results[name] = success
        except Exception as e:
            logger.error(f"Demo '{name}' crashed: {e}", exc_info=True)
            results[name] = False
            print(f"\n✗ Demo '{name}' crashed: {e}")

    # Summary
    print("\n" + "="*70)
    print("DEMO SUMMARY")
    print("="*70)

    successful = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    print("="*70)
    print(f"Results: {successful}/{total} demos passed")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    logger.info("="*50)
    logger.info(f"APPLICATION COMPLETED: {successful}/{total} demos passed")
    logger.info("="*50)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Application crashed: {e}", exc_info=True)
        print(f"\n\nCRITICAL ERROR: {e}")
        sys.exit(1)
