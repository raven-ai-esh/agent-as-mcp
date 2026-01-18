"""Test script for weather scenario with clarifications.

Demonstrates the dialogue agent asking clarifying questions before calling tools.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from raven_skills import SkillDialogueAgent, SkillStorage, Skill, Tool
from raven_skills.models.skill import SkillMetadata, SkillStep
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
# Weather Tool
# ─────────────────────────────────────────────────────────────────

def get_weather(city: str, date: str = "сегодня", include_wind: bool = False) -> str:
    """Simulate getting weather data."""
    # Simulated weather data
    weather_data = {
        "temperature": 5,
        "condition": "облачно",
        "wind_speed": 10,
    }
    
    result = f"Погода в {city} на {date}: {weather_data['temperature']}°C, {weather_data['condition']}"
    if include_wind:
        result += f", ветер {weather_data['wind_speed']} м/с"
    
    return result


# ─────────────────────────────────────────────────────────────────
# Pre-trained Weather Skill (with clarification steps)
# ─────────────────────────────────────────────────────────────────

async def create_weather_skill(storage: InMemoryStorage, embedding_client) -> Skill:
    """Create a pre-trained weather skill with clarification steps."""
    from raven_skills.core.embeddings import EmbeddingsClient
    
    emb = EmbeddingsClient(embedding_client, model="bge-m3:latest")
    
    # Generate embedding for the skill
    skill_text = "погода прогноз температура ветер город дата"
    embedding = await emb.embed_text(skill_text)
    
    skill = Skill(
        id="weather-skill-001",
        name="Прогноз погоды",
        version=1,
        metadata=SkillMetadata(
            description="Узнать прогноз погоды для города с уточнением даты и параметров",
            goal="Пользователь получает информацию о погоде в нужном городе на нужную дату",
            keywords=["погода", "прогноз", "температура", "ветер", "город"],
            embedding=embedding,
        ),
        steps=[
            SkillStep(order=1, instruction="Уточни у пользователя, на какую дату нужен прогноз погоды"),
            SkillStep(order=2, instruction="Спроси, нужно ли показать скорость ветра помимо температуры"),
            SkillStep(order=3, instruction="Вызови инструмент get_weather с собранными параметрами"),
        ],
        created_at=datetime.now(),
    )
    
    await storage.save(skill)
    return skill


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

async def main():
    print("\n" + "="*60)
    print("🌤️ Weather Scenario - Clarification Demo")
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
    
    # Define weather tool
    weather_tool = Tool(
        name="get_weather",
        description="Get weather forecast for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "date": {"type": "string", "description": "Date (e.g. 'сегодня', 'завтра')"},
                "include_wind": {"type": "boolean", "description": "Include wind speed"},
            },
            "required": ["city"],
        },
        function=get_weather,
    )
    
    # Create storage and pre-load weather skill
    storage = InMemoryStorage()
    weather_skill = await create_weather_skill(storage, embedding_client)
    print(f"\n✅ Pre-loaded skill: '{weather_skill.name}' ({len(weather_skill.steps)} steps)")
    for step in weather_skill.steps:
        print(f"   {step.order}. {step.instruction}")
    
    # Create agent
    agent = SkillDialogueAgent(
        client=llm_client,
        storage=storage,
        tools=[weather_tool],
        embedding_client=embedding_client,
        llm_model="gpt-4o-mini",
        embedding_model="bge-m3:latest",
        similarity_threshold=0.5,
        auto_generate_skills=False,  # Use pre-trained skill only
    )
    
    print("\n" + "-"*60)
    print("📝 Simulating dialogue...")
    print("-"*60)
    
    # ─────────────────────────────────────────────────────────────
    # Turn 1: User asks about weather
    # ─────────────────────────────────────────────────────────────
    print("\n👤 User: Какая погода в Воронеже?")
    response = await agent.chat("Какая погода в Воронеже?")
    print(f"🤖 Agent: {response.message}")
    print(f"   [needs_user_input={response.needs_user_input}, skill={response.skill_used.name if response.skill_used else None}]")
    
    if not response.needs_user_input:
        print("⚠️ Expected agent to ask for clarification!")
        return
    
    # ─────────────────────────────────────────────────────────────
    # Turn 2: User provides date
    # ─────────────────────────────────────────────────────────────
    print("\n👤 User: На завтра")
    response = await agent.chat("На завтра")
    print(f"🤖 Agent: {response.message}")
    print(f"   [needs_user_input={response.needs_user_input}]")
    
    if not response.needs_user_input:
        print("⚠️ Expected agent to ask about wind!")
        return
    
    # ─────────────────────────────────────────────────────────────
    # Turn 3: User says yes to wind
    # ─────────────────────────────────────────────────────────────
    print("\n👤 User: Да, покажи ветер тоже")
    response = await agent.chat("Да, покажи ветер тоже")
    print(f"🤖 Agent: {response.message}")
    print(f"   [needs_user_input={response.needs_user_input}]")
    print(f"   [tools_called={[t.tool_name for t in response.tools_called]}]")
    
    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("📋 Dialogue 1 Summary")
    print("="*60)
    print(f"📚 Conversation length: {len(agent.conversation_history)} messages")
    print(f"🎯 Skill used: {response.skill_used.name if response.skill_used else 'None'}")
    print(f"🔧 Tools called: {len(response.tools_called)}")
    
    # ─────────────────────────────────────────────────────────────
    # NEW DIALOGUE - Testing skill reuse
    # ─────────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("🔄 NEW DIALOGUE - Testing Skill Reuse")
    print("="*60)
    
    agent.reset()
    
    print("\n👤 User: Какой прогноз погоды в Москве?")
    response = await agent.chat("Какой прогноз погоды в Москве?")
    print(f"🤖 Agent: {response.message}")
    print(f"   [needs_user_input={response.needs_user_input}, skill_generated={response.skill_generated}]")
    
    print("\n👤 User: На выходные")
    response = await agent.chat("На выходные")
    print(f"🤖 Agent: {response.message}")
    
    print("\n👤 User: Нет, только температуру")
    response = await agent.chat("Нет, только температуру")
    print(f"🤖 Agent: {response.message}")
    print(f"   [tools_called={[t.tool_name for t in response.tools_called]}]")
    
    print("\n" + "="*60)
    print("📋 Final Summary")
    print("="*60)
    all_skills = await storage.get_all()
    print(f"📚 Skills in storage: {len(all_skills)}")
    for skill in all_skills:
        print(f"   - {skill.name}")
    
    print("\n✅ Weather scenario complete!")


if __name__ == "__main__":
    asyncio.run(main())
