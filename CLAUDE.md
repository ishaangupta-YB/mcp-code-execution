# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a demonstration project for the **OpenAI Agents SDK** that showcases proper implementation of function tools, agents-as-tools, and multi-agent orchestration patterns. The project deliberately does NOT use MCP (Model Context Protocol) - it focuses exclusively on function-based tools and agent composition.

## Key Commands

### Environment Setup
```bash
# Install dependencies
pip install openai-agents python-dotenv tavily-python pydantic aiohttp

# Verify setup (checks dependencies and API keys)
python verify_setup.py
```

### Running the Application
```bash
# Run all 8 demo scenarios
python app.py

# The app expects these environment variables in .env:
# - OPENAI_API_KEY
# - TAVILY_API_KEY
```

## Architecture

### Core Design Pattern: Orchestrator + Specialized Agents

The architecture follows a hierarchical agent pattern:

```
OrchestratorAgent (lines 342-397)
├── Function Tools (direct capabilities)
│   ├── calculate() - Math operations
│   ├── get_weather() - Weather data (TypedDict example)
│   ├── save_to_file() - Context-aware file operations
│   ├── read_from_file() - File reading
│   ├── risky_operation() - Error handling demo
│   └── web_search() - Tavily integration
│
└── Agent Tools (specialized sub-agents)
    ├── DataAnalyzer - Uses calculate tool
    ├── ContentWriter - No additional tools
    └── Researcher - Uses web_search tool (conditionally enabled)
```

The orchestrator coordinates all tools and decides which to invoke based on user queries.

### Three-Layer Agent System

1. **Function Tools Layer** (lines 64-282)
   - Direct tool implementations using `@function_tool` decorator
   - Each tool is a standalone async function
   - Logging prefix: `[TOOL: name]`

2. **Specialized Agents Layer** (lines 285-324)
   - Domain-specific agents (DataAnalyzer, ContentWriter, Researcher)
   - Created at module load time
   - Each has specific instructions and subset of tools

3. **Orchestrator Layer** (lines 342-397)
   - Main agent that user interacts with
   - Has access to all function tools AND all specialized agents (via `agent.as_tool()`)
   - Routes requests to appropriate tools/agents

### Context Management Pattern

Uses `RunContextWrapper[UserPreferences]` for dependency injection:

```python
UserPreferences (Pydantic model)
├── language: str
├── verbose: bool
└── search_enabled: bool
```

Context flows through:
- Tool calls (e.g., `save_to_file` checks `verbose` flag)
- Conditional enabling (e.g., `search_enabled()` checks `search_enabled` flag)
- Demo scenarios (each demo can create custom context)

### Tool Definition Patterns

When adding new tools, follow these established patterns:

**Pattern 1: Simple Function Tool**
```python
@function_tool
async def tool_name(param: Type) -> str:
    """Docstring is auto-parsed for tool description.

    Args:
        param: Description appears in schema
    """
    logger.info(f"[TOOL: tool_name] Called with: {param}")
    # implementation
    logger.info(f"[TOOL: tool_name] Result: {result}")
    return result
```

**Pattern 2: Context-Aware Tool**
```python
@function_tool
async def tool_name(ctx: RunContextWrapper[UserPreferences], param: str) -> str:
    """Tool with access to user context."""
    pref_value = ctx.context.some_field if ctx.context else default_value
    # implementation
```

**Pattern 3: Tool with Custom Error Handler**
```python
def tool_error_handler(context: RunContextWrapper[Any], error: Exception) -> str:
    logger.error(f"[ERROR HANDLER] {error}")
    return "User-friendly error message"

@function_tool(failure_error_function=tool_error_handler)
async def risky_tool(param: str) -> str:
    # implementation that might raise exceptions
```

**Pattern 4: TypedDict for Complex Inputs**
```python
class ComplexInput(TypedDict):
    field1: type1
    field2: type2

@function_tool
async def tool_name(data: ComplexInput) -> str:
    # Access data['field1'], data['field2']
```

### Logging Convention

All logging follows this format:
- Entry: `logger.info(f"[TOOL: name] Called with: {params}")`
- Result: `logger.info(f"[TOOL: name] Result: {output}")`
- Error: `logger.error(f"[TOOL: name] {error_msg}")`
- Specialized agents: `logger.info("✓ AgentName agent created")`

### Adding New Specialized Agents

1. Define agent at module level (before `create_orchestrator_agent()`):
```python
my_specialist = Agent(
    name="MySpecialist",
    instructions="Clear instructions about agent's role",
    tools=[list_of_relevant_tools]
)
logger.info("✓ MySpecialist agent created")
```

2. Add to orchestrator in `create_orchestrator_agent()`:
```python
my_specialist.as_tool(
    tool_name="do_specialized_task",
    tool_description="What this agent does",
    is_enabled=True  # or conditional function
)
```

### Conditional Tool Enabling

Use when tools should be dynamically available:

```python
def tool_enabled(ctx: RunContextWrapper[UserPreferences], agent: Any) -> bool:
    # Check context and return True/False
    return ctx.context.some_flag if ctx.context else default

# In agent creation:
agent.as_tool(
    tool_name="conditional_task",
    tool_description="...",
    is_enabled=tool_enabled  # Pass function reference
)
```

## Important Constraints

### What NOT to Use

This codebase deliberately **excludes** MCP (Model Context Protocol):
- Do NOT add `MCPServerStdio`, `MCPServerStreamableHttp`, `MCPServerSse`, or `HostedMCPTool`
- Do NOT add `mcp_servers` parameter to Agent constructor
- Focus remains on function tools and agent composition only

### Type Requirements

- All function tools must have type hints
- Complex inputs use TypedDict or Pydantic models
- Return type is always `str` (for tool output)
- Context parameter: `RunContextWrapper[UserPreferences]`

## Demo Suite Structure

The application includes 8 self-contained demos (lines 404-623):

Each demo follows this pattern:
```python
async def demo_name():
    """Description."""
    print(header)
    logger.info("Starting Demo X")

    try:
        agent = create_orchestrator_agent()
        result = await Runner.run(agent, query, context=...)
        logger.info("Demo X completed successfully")
        return True
    except Exception as e:
        logger.error(f"Demo X failed: {e}", exc_info=True)
        return False
```

Demos are orchestrated in `main()` which:
1. Validates environment (API keys)
2. Runs each demo sequentially
3. Tracks success/failure
4. Prints summary report

## External Dependencies

### Required API Keys
- `OPENAI_API_KEY` - For OpenAI Agents SDK
- `TAVILY_API_KEY` - For web search tool (get from https://tavily.com)

### Package Dependencies
- `openai-agents` - Core SDK
- `python-dotenv` - Environment variable management
- `tavily-python` - Web search API client
- `pydantic` - Data validation (UserPreferences model)
- `aiohttp` - Async HTTP (transitive dependency)

## Error Handling Philosophy

Three-layer error handling:

1. **Tool-level**: Custom error handlers return user-friendly strings
2. **Demo-level**: Try-catch wraps each demo, logs errors, continues to next
3. **Application-level**: Global handler in `main()` catches catastrophic failures

All errors are logged with `exc_info=True` for full tracebacks.

## Extending the Codebase

When adding functionality:

1. **New function tool**: Add before line 285, include logging, add to orchestrator
2. **New specialized agent**: Define at module level (285-324), add to orchestrator
3. **New demo**: Follow demo pattern (404-623), add to demos list in `main()`
4. **New context field**: Add to `UserPreferences` Pydantic model (lines 50-54)
5. **New data type**: Create TypedDict near line 57

Always maintain the logging convention and error handling patterns.
