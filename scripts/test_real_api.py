"""Integration test with real OpenAI API + Ollama embeddings.

Usage:
    1. Create .env file with OPENAI_API_KEY=sk-...
    2. Make sure Ollama is running with bge-m3 model
    3. Run: python scripts/test_real_api.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from raven_skills import SkillAgent, SkillStorage, Skill
from raven_skills.utils.similarity import cosine_similarity


class InMemoryStorage(SkillStorage):
    """Simple in-memory storage for testing."""
    
    def __init__(self):
        self._skills: dict[str, Skill] = {}
    
    async def save(self, skill: Skill) -> None:
        self._skills[skill.id] = skill
        print(f"  💾 Saved skill: {skill.name} (id={skill.id[:8]}...)")
    
    async def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)
    
    async def get_all(self) -> list[Skill]:
        return list(self._skills.values())
    
    async def delete(self, skill_id: str) -> None:
        self._skills.pop(skill_id, None)
    
    async def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Skill, float]]:
        results = []
        for skill in self._skills.values():
            if skill.metadata.embedding:
                score = cosine_similarity(embedding, skill.metadata.embedding)
                if score >= min_score:
                    results.append((skill, score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


async def test_skill_generation(agent: SkillAgent):
    """Test generating a skill from conversation."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Skill Generation")
    print("="*60)
    
    # Prepare a task
    print("\n📝 Preparing task...")
    task = await agent.prepare_task("Как задеплоить Python приложение в Docker?")
    print(f"  Query: {task.query}")
    print(f"  Key aspects: {task.key_aspects}")
    print(f"  Embedding dims: {len(task.embedding)}")
    
    # Simulate a conversation
    conversation = [
        {"role": "user", "content": "Как задеплоить Python приложение в Docker?"},
        {"role": "assistant", "content": "Для деплоя Python приложения в Docker нужно:\n1. Создать Dockerfile\n2. Собрать образ\n3. Запустить контейнер"},
        {"role": "user", "content": "А как написать Dockerfile?"},
        {"role": "assistant", "content": "Вот пример Dockerfile:\n\nFROM python:3.11-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"python\", \"main.py\"]"},
    ]
    
    # Generate skill
    print("\n🔧 Generating skill from conversation...")
    skill = await agent.generate_skill(
        task=task,
        conversation=conversation,
        final_result="Приложение успешно запущено в Docker контейнере",
    )
    
    print(f"\n✅ Generated skill:")
    print(f"  Name: {skill.name}")
    print(f"  Description: {skill.metadata.description}")
    print(f"  Goal: {skill.metadata.goal}")
    print(f"  Keywords: {skill.metadata.keywords}")
    print(f"  Embedding dims: {len(skill.metadata.embedding)}")
    print(f"  Steps ({len(skill.steps)}):")
    for step in skill.steps:
        print(f"    {step.order}. {step.instruction}")
    
    return skill


async def test_skill_matching(agent: SkillAgent, skill: Skill):
    """Test matching a query to existing skills."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Skill Matching")
    print("="*60)
    
    # Try to match a similar query
    print("\n🔍 Matching query: 'Деплой приложения в контейнер'")
    task, result = await agent.match("Деплой приложения в контейнер")
    
    print(f"\n📊 Match result:")
    print(f"  Found: {result.found}")
    print(f"  Score: {result.score:.4f}")
    print(f"  Threshold passed: {result.threshold_passed}")
    
    if result.skill:
        print(f"  Matched skill: {result.skill.name}")
    
    # Try a very different query
    print("\n🔍 Matching query: 'Как приготовить борщ'")
    task2, result2 = await agent.match("Как приготовить борщ")
    
    print(f"\n📊 Match result:")
    print(f"  Found: {result2.found}")
    print(f"  Score: {result2.score:.4f}")
    
    return result


async def test_skill_execution(agent: SkillAgent, skill: Skill):
    """Test executing a skill."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Skill Execution")
    print("="*60)
    
    task = await agent.prepare_task("Хочу задеплоить свой Flask сервер")
    
    print(f"\n⚡ Executing skill: {skill.name}")
    print(f"   Steps to execute: {len(skill.steps)}")
    
    result = await agent.execute(skill, task)
    
    print(f"\n📊 Execution result:")
    print(f"  Success: {result.success}")
    print(f"  Steps completed: {len(result.steps_completed)}")
    
    if result.output:
        output_preview = result.output[:300] + "..." if len(result.output) > 300 else result.output
        print(f"  Output: {output_preview}")
    
    if result.error:
        print(f"  Error: {result.error}")
    
    return result


async def test_diagnosis(agent: SkillAgent, skill: Skill, task, exec_result):
    """Test diagnosis and refinement."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Diagnosis & Refinement")
    print("="*60)
    
    print("\n🔬 Diagnosing execution...")
    action = await agent.diagnose(
        skill=skill,
        task=task,
        result=exec_result,
        user_feedback="Нужно больше деталей про docker-compose",
    )
    
    print(f"\n📊 Diagnosis result:")
    print(f"  Type: {action.type}")
    print(f"  Diagnosis: {action.diagnosis}")
    print(f"  Suggested changes: {action.suggested_changes}")
    
    print("\n🔧 Refining skill...")
    refined = await agent.refine(skill, action)
    
    print(f"\n✅ Refined skill:")
    print(f"  Name: {refined.name}")
    print(f"  Version: {refined.version}")
    print(f"  Steps ({len(refined.steps)}):")
    for step in refined.steps[:3]:  # Show first 3 steps
        print(f"    {step.order}. {step.instruction}")
    if len(refined.steps) > 3:
        print(f"    ... and {len(refined.steps) - 3} more steps")


async def main():
    print("\n" + "🚀"*30)
    print("  raven-skills Integration Test")
    print("  OpenAI (LLM) + Ollama (Embeddings)")
    print("🚀"*30)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ Error: OPENAI_API_KEY not found in environment")
        print("   Please create .env file with your API key")
        return
    
    print(f"\n✅ OpenAI API key: {api_key[:15]}...")
    print("✅ Ollama embeddings: http://localhost:11434/v1")
    
    # Create clients
    llm_client = AsyncOpenAI()
    embedding_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    
    # Create agent
    storage = InMemoryStorage()
    agent = SkillAgent(
        client=llm_client,
        embedding_client=embedding_client,
        storage=storage,
        llm_model="gpt-4o-mini",
        embedding_model="bge-m3:latest",
        similarity_threshold=0.6,  # Lower threshold for testing
        validate_matches=False,  # Skip LLM validation for speed
    )
    
    print("\n✅ SkillAgent initialized")
    
    try:
        # Run tests
        skill = await test_skill_generation(agent)
        result = await test_skill_matching(agent, skill)
        
        task = await agent.prepare_task("Хочу задеплоить Flask")
        exec_result = await test_skill_execution(agent, skill)
        
        await test_diagnosis(agent, skill, task, exec_result)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
