"""Test script for SkillDialogueAgent with tools.

Demonstrates the dialogue agent that always operates through skills.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from raven_skills import SkillDialogueAgent, SkillStorage, Skill, Tool
from raven_skills.utils.similarity import cosine_similarity


# ─────────────────────────────────────────────────────────────────
# In-Memory Storage
# ─────────────────────────────────────────────────────────────────

class InMemoryStorage(SkillStorage):
    def __init__(self):
        self._skills: dict[str, Skill] = {}
    
    async def save(self, skill: Skill) -> None:
        self._skills[skill.id] = skill
    
    async def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)
    
    async def get_all(self) -> list[Skill]:
        return list(self._skills.values())
    
    async def delete(self, skill_id: str) -> None:
        self._skills.pop(skill_id, None)
    
    async def search_by_embedding(
        self, embedding: list[float], top_k: int = 5, min_score: float = 0.0
    ) -> list[tuple[Skill, float]]:
        results = []
        for skill in self._skills.values():
            if skill.metadata.embedding:
                score = cosine_similarity(embedding, skill.metadata.embedding)
                if score >= min_score:
                    results.append((skill, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ─────────────────────────────────────────────────────────────────
# Sample Tools
# ─────────────────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """Simulate web search."""
    return f"Search results for '{query}': [Result 1, Result 2, Result 3]"


def run_command(command: str) -> str:
    """Simulate running a command."""
    return f"$ {command}\nCommand executed successfully."


def read_file(path: str) -> str:
    """Simulate reading a file."""
    return f"Contents of {path}: [file content placeholder]"


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

async def main():
    print("\n" + "="*60)
    print("🤖 SkillDialogueAgent Test")
    print("="*60)
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return
    
    # Create clients
    llm_client = AsyncOpenAI()
    embedding_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    
    # Define tools
    tools = [
        Tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
            function=web_search,
        ),
        Tool(
            name="run_command",
            description="Run a shell command",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to run"}
                },
                "required": ["command"],
            },
            function=run_command,
        ),
        Tool(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"],
            },
            function=read_file,
        ),
    ]
    
    # Create agent
    storage = InMemoryStorage()
    agent = SkillDialogueAgent(
        client=llm_client,
        storage=storage,
        tools=tools,
        embedding_client=embedding_client,
        llm_model="gpt-4o-mini",
        embedding_model="bge-m3:latest",
        similarity_threshold=0.55,
        auto_generate_skills=True,
    )
    
    print("\n✅ Agent initialized with tools:", [t.name for t in tools])
    
    # ─────────────────────────────────────────────────────────────
    # Test 1: First message (generates new skill)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("📝 Test 1: First message (should generate skill)")
    print("-"*60)
    
    response = await agent.chat("Как задеплоить Python приложение в Kubernetes?")
    
    print(f"\n🤖 Response: {response.message[:300]}...")
    print(f"\n📊 Skill used: {response.skill_used.name if response.skill_used else 'None'}")
    print(f"   Generated new: {response.skill_generated}")
    print(f"   Tools called: {[t.tool_name for t in response.tools_called]}")
    
    # ─────────────────────────────────────────────────────────────
    # Test 2: Similar message (should match existing skill)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("📝 Test 2: Similar message (should match existing skill)")
    print("-"*60)
    
    response2 = await agent.chat("Деплой приложения в k8s кластер")
    
    print(f"\n🤖 Response: {response2.message[:300]}...")
    print(f"\n📊 Skill used: {response2.skill_used.name if response2.skill_used else 'None'}")
    print(f"   Generated new: {response2.skill_generated}")
    
    # ─────────────────────────────────────────────────────────────
    # Test 3: Different topic (should generate new skill)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "-"*60)
    print("📝 Test 3: Different topic (should generate new skill)")
    print("-"*60)
    
    response3 = await agent.chat("Настрой мониторинг с Prometheus и Grafana")
    
    print(f"\n🤖 Response: {response3.message[:300]}...")
    print(f"\n📊 Skill used: {response3.skill_used.name if response3.skill_used else 'None'}")
    print(f"   Generated new: {response3.skill_generated}")
    
    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("📋 Summary")
    print("="*60)
    
    all_skills = await storage.get_all()
    print(f"\n📚 Total skills in storage: {len(all_skills)}")
    for skill in all_skills:
        print(f"   - {skill.name} ({len(skill.steps)} steps)")
    
    print(f"\n💬 Conversation history: {len(agent.conversation_history)} messages")
    
    print("\n✅ SkillDialogueAgent test complete!")


if __name__ == "__main__":
    asyncio.run(main())
