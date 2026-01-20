"""Test HyDE and Multi-vector matching features."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


async def test_hyde_multivector():
    from openai import AsyncOpenAI
    from raven_skills import SkillDialogueAgent, Tool, JSONStorage
    
    print("=" * 60)
    print("🧪 Testing HyDE + Multi-vector Matching")
    print("=" * 60)
    
    # Clean start
    storage_path = "./test_hyde_skills.json"
    if os.path.exists(storage_path):
        os.remove(storage_path)
    
    storage = JSONStorage(storage_path)
    client = AsyncOpenAI()
    
    # Simple tool
    def get_weather(city: str) -> str:
        return f"Weather in {city}: sunny, 22°C"
    
    weather_tool = Tool(
        name="get_weather",
        description="Get weather forecast for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        },
        function=get_weather,
    )
    
    agent = SkillDialogueAgent(
        client=client,
        storage=storage,
        tools=[weather_tool],
        auto_generate_skills=True,
        similarity_threshold=0.55,
    )
    
    # Test 1: Create skill
    print("\n" + "─" * 60)
    print("📝 TEST 1: Create skill with 'What is the weather in Moscow?'")
    print("─" * 60)
    
    response = await agent.chat("What is the weather in Moscow?")
    print(f"✅ Response: {response.message[:150]}...")
    print(f"   Skill: {response.skill_used.name if response.skill_used else 'None'}")
    print(f"   Generated: {response.skill_generated}")
    
    # Check multi-vector embeddings were created
    skills = await storage.get_all()
    if skills:
        skill = skills[0]
        print(f"\n📊 Multi-vector check:")
        print(f"   Primary embedding: {len(skill.metadata.embedding)} dims")
        print(f"   Multi-embeddings: {len(skill.metadata.embeddings)} vectors")
        for i, emb in enumerate(skill.metadata.embeddings):
            print(f"     Vector {i+1}: {len(emb)} dims")
    
    # Reset for fresh matching test
    agent.reset()
    
    # Test 2: Match with different phrasing
    print("\n" + "─" * 60)
    print("📝 TEST 2: Match with different phrasing")
    print("─" * 60)
    
    test_queries = [
        "Tell me the weather",  # No city
        "What's it like outside in Paris?",  # Different wording
        "Погода в Лондоне",  # Russian
        "Temperature forecast for Tokyo",  # Different aspect
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        response = await agent.chat(query)
        matched = response.skill_used.name if response.skill_used else "NEW SKILL"
        generated = "🆕" if response.skill_generated else "♻️"
        print(f"   {generated} {matched}")
        agent.reset()
    
    # Cleanup
    if os.path.exists(storage_path):
        os.remove(storage_path)
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_hyde_multivector())
