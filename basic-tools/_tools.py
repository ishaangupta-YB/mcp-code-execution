"""
Function Tools for OpenAI Agents SDK
All tool implementations using @function_tool decorator
"""

import logging
import os
from typing import Any, TypedDict

from pydantic import BaseModel
from tavily import TavilyClient

from agents import RunContextWrapper, function_tool

# Module logger
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class UserPreferences(BaseModel):
    """User preferences for the agent session."""
    language: str = "english"
    verbose: bool = False
    search_enabled: bool = True


class Location(TypedDict):
    """Geographic location coordinates."""
    lat: float
    long: float
    city: str


# ============================================================================
# ERROR HANDLERS
# ============================================================================

def custom_error_handler(context: RunContextWrapper[Any], error: Exception) -> str:
    """Custom error handler for tool failures."""
    error_type = type(error).__name__
    error_msg = f"Tool execution failed [{error_type}]: {str(error)}. Please try a different approach."
    logger.error(f"[ERROR HANDLER] {error_msg}")
    return error_msg


# ============================================================================
# FUNCTION TOOLS
# ============================================================================

@function_tool
async def calculate(operation: str, a: float, b: float) -> str:
    """Perform mathematical calculations.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: The first number
        b: The second number

    Returns:
        The result of the calculation
    """
    logger.info(f"[TOOL: calculate] Called with: operation={operation}, a={a}, b={b}")

    operations = {
        "add": lambda x, y: x + y,
        "subtract": lambda x, y: x - y,
        "multiply": lambda x, y: x * y,
        "divide": lambda x, y: x / y if y != 0 else "Error: Division by zero"
    }

    if operation not in operations:
        error_msg = f"Error: Unknown operation '{operation}'. Use: add, subtract, multiply, divide"
        logger.error(f"[TOOL: calculate] {error_msg}")
        return error_msg

    result = operations[operation](a, b)
    result_str = f"{a} {operation} {b} = {result}"
    logger.info(f"[TOOL: calculate] Result: {result_str}")
    return result_str


@function_tool
async def get_weather(location: Location) -> str:
    """Fetch current weather for a given location.

    Args:
        location: The location to fetch weather for (with lat, long, city)

    Returns:
        Weather information as a string
    """
    logger.info(f"[TOOL: get_weather] Called for: {location.get('city', 'Unknown')}")

    # Simulated weather data (in production, call a real API)
    import random
    weather_conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]

    condition = random.choice(weather_conditions)
    temp = random.randint(15, 30)

    result = f"Weather in {location['city']}: {condition}, {temp}°C (Lat: {location['lat']}, Long: {location['long']})"
    logger.info(f"[TOOL: get_weather] Result: {result}")
    return result


@function_tool
async def save_to_file(ctx: RunContextWrapper[UserPreferences], filename: str, content: str) -> str:
    """Save content to a file.

    Args:
        filename: The name of the file to save
        content: The content to write to the file

    Returns:
        Confirmation message
    """
    logger.info(f"[TOOL: save_to_file] Saving to: {filename}")

    try:
        # Access user preferences from context
        verbose = ctx.context.verbose if ctx.context else False

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)

        if verbose:
            result = f"Successfully saved {len(content)} characters to '{filename}'"
        else:
            result = f"File '{filename}' saved successfully"

        logger.info(f"[TOOL: save_to_file] {result}")
        return result

    except Exception as e:
        error_msg = f"Error saving file: {str(e)}"
        logger.error(f"[TOOL: save_to_file] {error_msg}")
        return error_msg


@function_tool
async def read_from_file(filename: str) -> str:
    """Read content from a file.

    Args:
        filename: The name of the file to read

    Returns:
        The file contents
    """
    logger.info(f"[TOOL: read_from_file] Reading from: {filename}")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        result = content if content else "File is empty"
        logger.info(f"[TOOL: read_from_file] Read {len(content)} characters")
        return result

    except FileNotFoundError:
        error_msg = f"Error: File '{filename}' not found"
        logger.error(f"[TOOL: read_from_file] {error_msg}")
        return error_msg
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        logger.error(f"[TOOL: read_from_file] {error_msg}")
        return error_msg


@function_tool(failure_error_function=custom_error_handler)
async def risky_operation(value: int) -> str:
    """A tool that might fail to demonstrate error handling.

    Args:
        value: An integer value (must be positive)

    Returns:
        Success message
    """
    logger.info(f"[TOOL: risky_operation] Called with value: {value}")

    if value < 0:
        logger.warning(f"[TOOL: risky_operation] Negative value received: {value}")
        raise ValueError("Value must be positive")

    result = f"Operation successful with value: {value}"
    logger.info(f"[TOOL: risky_operation] {result}")
    return result


@function_tool
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)

    Returns:
        Search results as formatted text
    """
    logger.info(f"[TOOL: web_search] Searching for: '{query}' (max_results={max_results})")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        error_msg = "TAVILY_API_KEY not found in environment variables"
        logger.error(f"[TOOL: web_search] {error_msg}")
        return f"Error: {error_msg}"

    try:
        client = TavilyClient(api_key=api_key)
        logger.info(f"[TOOL: web_search] Tavily client initialized, performing search...")

        # Perform the search
        response = client.search(
            query=query,
            max_results=max_results,
            include_answer=True
        )

        logger.info(f"[TOOL: web_search] Received {len(response.get('results', []))} results")

        # Format results
        results = []

        # Add the AI-generated answer if available
        if response.get("answer"):
            results.append(f"Summary: {response['answer']}\n")
            logger.info(f"[TOOL: web_search] AI summary available")

        # Add individual search results
        results.append("Search Results:")
        for idx, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No description")
            results.append(f"\n{idx}. {title}")
            results.append(f"   URL: {url}")
            results.append(f"   {content[:200]}...")

        formatted_results = "\n".join(results)
        logger.info(f"[TOOL: web_search] Successfully formatted {len(response.get('results', []))} results")
        return formatted_results

    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        logger.error(f"[TOOL: web_search] {error_msg}")
        return error_msg


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Data models
    'UserPreferences',
    'Location',
    # Error handlers
    'custom_error_handler',
    # Function tools
    'calculate',
    'get_weather',
    'save_to_file',
    'read_from_file',
    'risky_operation',
    'web_search',
]
