"""Test LangChain and LangGraph integrations.

Verifies that raven-skills works correctly with:
1. LangChain Tools
2. LangChain Runnable (LCEL)
3. LangGraph State Machine
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from raven_skills import SkillDialogueAgent, SkillStorage, Skill, Tool
from raven_skills.models.skill import SkillMetadata, SkillStep
from raven_skills.utils.similarity import cosine_similarity
from raven_skills.integrations import (
    SkillMatcherTool,
    SkillDialogueTool,
    SkillAgentRunnable,
    create_skill_node,
    create_skill_router,
    SkillGraphState,
)
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
# Storage & Setup
# ═══════════════════════════════════════════════════════════════════

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


def get_weather(city: str) -> str:
    return f"В {city} сейчас +5°C, облачно"


async def create_test_skill(storage, emb_client):
    from raven_skills.core.embeddings import EmbeddingsClient
    emb = EmbeddingsClient(emb_client, model="bge-m3:latest")
    embedding = await emb.embed_text("погода прогноз температура")
    
    skill = Skill(
        id="skill-weather",
        name="Прогноз погоды",
        version=1,
        metadata=SkillMetadata(
            description="Узнать погоду в городе",
            goal="Пользователь получает прогноз",
            keywords=["погода", "прогноз"],
            embedding=embedding,
        ),
        steps=[SkillStep(order=1, instruction="Вызови get_weather для города")],
        created_at=datetime.now(),
    )
    await storage.save(skill)
    return skill


# ═══════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════

async def test_langchain_tool(agent):
    """Test 1: LangChain Tool wrapper."""
    print("\n" + "─"*60)
    print("🔧 Test 1: LangChain Tool (SkillDialogueTool)")
    print("─"*60)
    
    tool = SkillDialogueTool(agent=agent)
    
    print(f"   Tool name: {tool.name}")
    print(f"   Description: {tool.description}")
    
    # Invoke tool
    result = await tool._arun("Какая погода в Москве?")
    print(f"\n   Input: 'Какая погода в Москве?'")
    print(f"   Output: {result[:100]}...")
    
    assert "SKILL" in result or "погода" in result.lower(), "Tool should use skill"
    print("\n   ✅ LangChain Tool works!")


async def test_langchain_runnable(agent):
    """Test 2: LangChain Runnable (LCEL compatible)."""
    print("\n" + "─"*60)
    print("⚡ Test 2: LangChain Runnable (SkillAgentRunnable)")
    print("─"*60)
    
    runnable = SkillAgentRunnable(agent=agent)
    
    # Invoke runnable
    response = await runnable.ainvoke("Погода в Питере?")
    
    print(f"   Input: 'Погода в Питере?'")
    print(f"   Response type: {type(response).__name__}")
    print(f"   Message: {response.message[:100]}...")
    print(f"   Skill used: {response.skill_used.name if response.skill_used else 'None'}")
    
    assert response.message, "Runnable should return response"
    print("\n   ✅ LangChain Runnable works!")


async def test_langgraph_node(agent):
    """Test 3: LangGraph State Machine."""
    print("\n" + "─"*60)
    print("📊 Test 3: LangGraph State Machine")
    print("─"*60)
    
    from langgraph.graph import StateGraph, END
    
    # Create graph
    graph = StateGraph(SkillGraphState)
    
    # Add skill node
    skill_node = create_skill_node(agent)
    graph.add_node("skill_agent", skill_node)
    
    # Add router
    router = create_skill_router(
        needs_input_node="skill_agent",  # Loop back if needs input
        complete_node=END,
    )
    
    # Define edges
    graph.set_entry_point("skill_agent")
    graph.add_conditional_edges("skill_agent", router)
    
    # Compile
    app = graph.compile()
    
    print("   Graph compiled successfully!")
    print(f"   Nodes: {list(graph.nodes.keys())}")
    
    # Run graph
    initial_state: SkillGraphState = {
        "messages": [],
        "current_message": "Какая погода в Сочи?",
        "response": "",
        "skill_name": None,
        "skill_generated": False,
        "needs_user_input": False,
        "tools_called": [],
    }
    
    result = await app.ainvoke(initial_state)
    
    print(f"\n   Input: 'Какая погода в Сочи?'")
    print(f"   Response: {result['response'][:100]}...")
    print(f"   Skill: {result['skill_name']}")
    print(f"   Tools called: {result['tools_called']}")
    print(f"   Messages in state: {len(result['messages'])}")
    
    assert result["response"], "Graph should produce response"
    print("\n   ✅ LangGraph integration works!")


async def test_langgraph_multi_turn(agent):
    """Test 4: LangGraph multi-turn with clarifications."""
    print("\n" + "─"*60)
    print("🔄 Test 4: LangGraph Multi-Turn (with clarifications)")
    print("─"*60)
    
    from langgraph.graph import StateGraph, END
    
    # Create agent with clarification skill
    agent.reset()
    
    graph = StateGraph(SkillGraphState)
    skill_node = create_skill_node(agent)
    graph.add_node("skill_agent", skill_node)
    
    # This time, we simulate continuing the conversation
    graph.set_entry_point("skill_agent")
    graph.add_edge("skill_agent", END)
    
    app = graph.compile()
    
    # First turn
    state1: SkillGraphState = {
        "messages": [],
        "current_message": "Хочу узнать погоду",
        "response": "",
        "skill_name": None,
        "skill_generated": False,
        "needs_user_input": False,
        "tools_called": [],
    }
    
    result1 = await app.ainvoke(state1)
    print(f"   Turn 1: 'Хочу узнать погоду'")
    print(f"   Response: {result1['response'][:80]}...")
    print(f"   needs_user_input: {result1['needs_user_input']}")
    
    # Second turn (continue conversation through agent state)
    state2: SkillGraphState = {
        **result1,
        "current_message": "В Воронеже",
    }
    
    result2 = await app.ainvoke(state2)
    print(f"\n   Turn 2: 'В Воронеже'")
    print(f"   Response: {result2['response'][:80]}...")
    print(f"   Tools called: {result2['tools_called']}")
    
    print("\n   ✅ LangGraph multi-turn works!")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

async def main():
    print("\n" + "═"*60)
    print("🔗 LANGCHAIN / LANGGRAPH INTEGRATION TESTS")
    print("═"*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return
    
    llm_client = AsyncOpenAI()
    emb_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    # Setup
    storage = InMemoryStorage()
    await create_test_skill(storage, emb_client)
    
    weather_tool = Tool(
        name="get_weather",
        description="Get weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        function=get_weather,
    )
    
    agent = SkillDialogueAgent(
        client=llm_client,
        storage=storage,
        tools=[weather_tool],
        embedding_client=emb_client,
        llm_model="gpt-4o-mini",
        embedding_model="bge-m3:latest",
        similarity_threshold=0.5,
        auto_generate_skills=True,
    )
    
    print("\n✅ Agent initialized with weather skill")
    
    # Run tests
    try:
        await test_langchain_tool(agent)
        agent.reset()
        
        await test_langchain_runnable(agent)
        agent.reset()
        
        await test_langgraph_node(agent)
        agent.reset()
        
        await test_langgraph_multi_turn(agent)
        
        print("\n\n" + "═"*60)
        print("🎉 ALL TESTS PASSED!")
        print("═"*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
