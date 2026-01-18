# raven-skills

A library for building adaptive AI agents with a skill-based approach. The agent automatically selects the best skill for a task, or generates a new skill from successful conversations.

Unlike standard RAG or agent systems, `raven-skills` implements **learning through experience**: the agent remembers successful solutions and turns them into reusable skills.

## Key Features

- 🎯 **Automatic skill matching** — finds the most suitable skill via embedding similarity
- 🔧 **Skill generation** — creates new skills from successful dialogues
- 🧠 **Built-in LLM client** — all prompt logic and SGR (Schema-Guided Reasoning) included
- 🔄 **Optimization** — automatic merging of similar skills
- ⚡ **Adaptive improvement** — three strategies for fixing failed executions
- 💬 **Dialogue agent** — multi-turn conversations with clarifications and tool support

## Installation

```bash
pip install raven-skills
```

## Quick Start

The library is designed to be simple. You only need an OpenAI client and a skill storage.

```python
import asyncio
from openai import AsyncOpenAI
from raven_skills import SkillAgent, InMemoryStorage

async def main():
    # 1. Initialize
    agent = SkillAgent(
        client=AsyncOpenAI(),
        storage=InMemoryStorage(),
        llm_model="gpt-4o-mini",
    )

    # 2. Match a skill
    task, result = await agent.match("How to deploy an application?")

    if result.found:
        print(f"Found skill: {result.skill.name}")
        
        # 3. Execute the skill
        execution = await agent.execute(result.skill, task)
        print(f"Result: {execution.output}")
        
    else:
        print("No skill found, switching to dialogue mode...")
        # Here you handle the conversation yourself
        conversation_log = [
            {"role": "user", "content": "How to deploy?"},
            {"role": "assistant", "content": "First, build the image..."},
        ]
        
        # 4. Save experience as a new skill
        new_skill = await agent.generate_skill(
            task=task,
            conversation=conversation_log,
            final_result="Application deployed",
        )
        print(f"Created new skill: {new_skill.name}")

if __name__ == "__main__":
    asyncio.run(main())
```

## How It Works

`SkillAgent` is the single entry point. Internally, it encapsulates all LLM complexity:

1. **Prompts & Templates**: All system prompts are pre-written and optimized.
2. **Schema-Guided Reasoning (SGR)**: The agent doesn't just generate text — it fills strict Pydantic schemas. This ensures skill steps are always in the correct format.
3. **Embeddings**: Vector generation for search happens automatically via the OpenAI client.

No need to write your own prompts or parsers.

## Skill Storage

Skills are stored using the `SkillStorage` interface. The library includes `InMemoryStorage` for testing, but for production you can implement your own (e.g., Postgres + pgvector).

### Custom Storage Implementation

```python
from raven_skills import SkillStorage, Skill

class PostgresStorage(SkillStorage):
    def __init__(self, db_pool):
        self.db = db_pool

    async def save(self, skill: Skill) -> None:
        # Your SQL INSERT ...
        pass

    async def get(self, skill_id: str) -> Skill | None:
        # Your SQL SELECT ...
        pass
        
    async def get_all(self) -> list[Skill]:
        # Get all skills
        pass

    async def delete(self, skill_id: str) -> None:
        # Delete skill
        pass

    async def search_by_embedding(
        self, embedding: list[float], top_k: int = 5, min_score: float = 0.0
    ) -> list[tuple[Skill, float]]:
        # Vector similarity search
        pass
```

## Advanced Features

### Diagnosis and Self-Correction

If a skill performs poorly, the agent can diagnose the problem and suggest a fix.

```python
# Assume execution failed or user is unhappy
execution = await agent.execute(skill, task)

# Diagnose
action = await agent.diagnose(
    skill=skill,
    task=task,
    result=execution,
    user_feedback="Step 2 threw an auth error",
)

print(f"Diagnosis: {action.diagnosis}")
# e.g.: "wrong_steps" -> "Error in skill steps"

# Refine (creates new version of skill)
refined_skill = await agent.refine(skill, action)
```

### Merging and Optimization

Over time, duplicate skills may appear. The agent can find and merge them.

```python
# Find similar skills and merge them
# (dry_run=False to apply changes)
results = await agent.optimize(similarity_threshold=0.95, dry_run=False)

for originals, merged in results:
    print(f"Merged {len(originals)} skills into '{merged.name}'")
```

## Data Models

### Skill

```python
@dataclass
class Skill:
    id: str
    name: str                  # Name
    metadata: SkillMetadata    # description, goal, keywords, embedding
    steps: list[SkillStep]     # Execution steps
    version: int               # Skill version
    parent_id: str | None      # Parent ID (if forked)
```

### Task

```python
@dataclass
class Task:
    id: str
    query: str                # Original user query
    key_aspects: list[str]    # Key aspects extracted by LLM
    embedding: list[float]    # Query vector
```

## Dialogue Agent

For interactive scenarios with clarifying questions, use `SkillDialogueAgent`:

```python
from raven_skills import SkillDialogueAgent, Tool

# Define tools
weather_tool = Tool(
    name="get_weather",
    description="Get weather forecast",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    function=lambda city: f"Weather in {city}: +10°C",
)

agent = SkillDialogueAgent(
    client=AsyncOpenAI(),
    storage=storage,
    tools=[weather_tool],
    auto_generate_skills=True,  # Generate skills automatically
)

# Dialogue with clarifications
response = await agent.chat("What's the weather?")
# Agent: "Which city?" (needs_user_input=True)

response = await agent.chat("Moscow")
# Agent: "Weather in Moscow: +10°C" (called get_weather)

# The "weather" skill is now saved and will be reused
```

## LangChain / LangGraph Integration

The library is fully compatible with the LangChain ecosystem.

### LangChain Tool

```python
from raven_skills.integrations import SkillMatcherTool, SkillDialogueTool

# As a tool in a LangChain agent
skill_tool = SkillMatcherTool(agent=skill_agent)
dialogue_tool = SkillDialogueTool(agent=dialogue_agent)

# Use in AgentExecutor or chains
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=..., tools=[skill_tool])
```

### LangChain Runnable (LCEL)

```python
from raven_skills.integrations import SkillAgentRunnable

# As a Runnable in LCEL chain
skill_runnable = SkillAgentRunnable(agent=dialogue_agent)

chain = prompt | skill_runnable | output_parser
result = await chain.ainvoke({"query": "Book a restaurant"})
```

### LangGraph State Machine

```python
from langgraph.graph import StateGraph
from raven_skills.integrations import (
    create_skill_node,
    create_skill_router,
    SkillGraphState,
)

# Create state graph
graph = StateGraph(SkillGraphState)

# Add node with our agent
skill_node = create_skill_node(dialogue_agent)
graph.add_node("skill_agent", skill_node)

# Routing: if user input needed — wait, otherwise complete
router = create_skill_router(
    needs_input_node="wait_for_input",
    complete_node="__end__",
)
graph.add_conditional_edges("skill_agent", router)

# Compile and run
app = graph.compile()
result = await app.ainvoke({"current_message": "Order pizza"})
```

### Available Adapters

| Adapter | Description |
|---------|-------------|
| `SkillMatcherTool` | LangChain Tool for skill matching and execution |
| `SkillDialogueTool` | LangChain Tool for dialogue agent |
| `SkillAgentRunnable` | LCEL Runnable for use in chains |
| `create_skill_node()` | Node factory for LangGraph |
| `create_skill_router()` | Router for conditional edges in LangGraph |
| `SkillGraphState` | TypedDict state schema for LangGraph |

## MCP Server

Run raven-skills as an MCP server for use with Claude, Cursor, and other MCP clients.

### Installation

```bash
pip install raven-skills[mcp]
```

### Quick Start

```bash
# Default server with OpenAI
python -m raven_skills serve

# With Ollama
python -m raven_skills serve --base-url http://localhost:11434/v1 --model llama3
```

### All CLI Options

```bash
python -m raven_skills serve [APP] [OPTIONS]

# APP (optional): Custom app module:variable (e.g., my_app:mcp)

Options:
  --port PORT              HTTP port (default: 8000)
  --transport {http,stdio} Transport type (default: http)
  --storage PATH           Skills JSON file (default: ./skills.json)
  --model MODEL            LLM model (default: gpt-4o-mini)
  --embedding-model MODEL  Embedding model (default: text-embedding-3-small)
  --base-url URL           OpenAI-compatible API URL
  --api-key KEY            API key (overrides OPENAI_API_KEY)
  --embedding-base-url URL Separate URL for embeddings
```

### Custom App (FastAPI-style)

Create your fully configured server in Python, run with CLI:

```python
# my_server.py
from openai import AsyncOpenAI
from mcp.server.fastmcp import FastMCP
from raven_skills import SkillDialogueAgent, Tool, JSONStorage

# 1. Configure your tools
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 22°C"

def search_restaurants(location: str, cuisine: str = "any") -> str:
    return f"Found 5 {cuisine} restaurants near {location}"

tools = [
    Tool(
        name="get_weather",
        description="Get current weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        function=get_weather,
    ),
    Tool(
        name="search_restaurants",
        description="Find restaurants",
        parameters={"type": "object", "properties": {
            "location": {"type": "string"},
            "cuisine": {"type": "string"},
        }},
        function=search_restaurants,
    ),
]

# 2. Configure client and storage
client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
storage = JSONStorage("./my-skills.json")

# 3. Create agent with everything configured
agent = SkillDialogueAgent(
    client=client,
    storage=storage,
    tools=tools,
    llm_model="llama3",
    embedding_model="bge-m3:latest",
    auto_generate_skills=True,
)

# 4. Create MCP server and expose agent
mcp = FastMCP(name="my-skills-server")

@mcp.tool()
async def chat(message: str) -> str:
    """Chat with the skill agent."""
    response = await agent.chat(message)
    return response.message

@mcp.tool()
async def list_skills() -> list[dict]:
    """List all learned skills."""
    skills = await storage.get_all()
    return [{"name": s.name, "description": s.metadata.description} for s in skills]

@mcp.tool()
async def reset_conversation() -> str:
    """Reset the conversation state."""
    agent.reset()
    return "Conversation reset"
```

Run it:
```bash
python -m raven_skills serve my_server:mcp --port 9000
```

### Available Tools

| Tool | Description |
|------|-------------|
| `execute_skill` | Execute a skill for a query (auto-matches or uses specific skill) |
| `list_skills` | List all available skills |
| `create_skill` | Create a new skill manually |
| `search_skills` | Search skills by semantic similarity |

### Available Resources

| Resource | URI | Description |
|----------|-----|-------------|
| Skills Library | `skills://library` | JSON array of all skills |
| Skill Detail | `skills://skill/{id}` | Single skill details |

### Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "raven-skills": {
      "command": "python",
      "args": ["-m", "raven_skills", "serve", "--transport", "stdio"]
    }
  }
}
```

## License

Apache 2.0