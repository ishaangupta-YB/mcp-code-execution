"""
Specialized Agents for OpenAI Agents SDK
Agent definitions and conditional enabling functions
"""

import logging
from typing import Any

from agents import Agent, RunContextWrapper

# Import tools
from _tools import UserPreferences, calculate, web_search

# Module logger
logger = logging.getLogger(__name__)


# ============================================================================
# CONDITIONAL ENABLING FUNCTIONS
# ============================================================================

def search_enabled(ctx: RunContextWrapper[UserPreferences], agent: Any) -> bool:
    """Enable search based on user preferences.

    Args:
        ctx: Run context with user preferences
        agent: The agent instance

    Returns:
        True if search should be enabled, False otherwise
    """
    enabled = ctx.context.search_enabled if ctx.context else True
    logger.info(f"[CONDITIONAL] search_enabled check: {enabled}")
    return enabled


# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

logger.info("Initializing specialized agents...")

# Data Analysis Agent
data_analyzer_agent = Agent(
    name="DataAnalyzer",
    instructions=(
        "You are a data analysis expert. You analyze data and provide insights. "
        "When given data, you identify patterns, calculate statistics, and provide summaries. "
        "Always structure your analysis clearly with bullet points."
    ),
    tools=[calculate]
)
logger.info("✓ DataAnalyzer agent created")


# Content Writer Agent
content_writer_agent = Agent(
    name="ContentWriter",
    instructions=(
        "You are a professional content writer. You create well-structured, engaging content. "
        "You can write articles, summaries, and documentation. "
        "Always use proper formatting and clear language."
    ),
)
logger.info("✓ ContentWriter agent created")


# Research Agent (with web search)
research_agent = Agent(
    name="Researcher",
    instructions=(
        "You are a research specialist. You search for information and compile findings. "
        "Use web search to find current information. "
        "Cite your sources and provide comprehensive answers."
    ),
    tools=[web_search],
)
logger.info("✓ Researcher agent created")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Conditional functions
    'search_enabled',
    # Specialized agents
    'data_analyzer_agent',
    'content_writer_agent',
    'research_agent',
]
