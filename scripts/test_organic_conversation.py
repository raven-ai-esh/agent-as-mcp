"""Organic conversation test with dynamic skill generation.

Tests natural dialogue flow where:
1. User has casual conversation
2. Mid-conversation, user wants food delivery
3. Agent generates skills dynamically
4. Conversation continues
"""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from raven_skills import SkillDialogueAgent, SkillStorage, Skill, Tool
from raven_skills.utils.similarity import cosine_similarity


# ═══════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════

class InMemoryStorage(SkillStorage):
    def __init__(self):
        self._skills: dict[str, Skill] = {}
    
    async def save(self, skill: Skill) -> None:
        self._skills[skill.id] = skill
        print(f"   💾 НОВЫЙ НАВЫК: {skill.name}")
        for step in skill.steps:
            print(f"      {step.order}. {step.instruction[:60]}...")
    
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


# ═══════════════════════════════════════════════════════════════════
# Tools
# ═══════════════════════════════════════════════════════════════════

def search_food_delivery(cuisine: str, location: str = "рядом") -> str:
    """Search for food delivery options."""
    options = {
        "пицца": ["Додо Пицца (30 мин, от 500₽)", "Папа Джонс (40 мин, от 600₽)"],
        "суши": ["Тануки (45 мин, от 800₽)", "Сушивок (35 мин, от 650₽)"],
        "бургер": ["Вкусно и точка (25 мин, от 300₽)", "Black Star Burger (35 мин, от 450₽)"],
        "default": ["Яндекс Еда (разная кухня)", "Delivery Club (рядом с вами)"],
    }
    results = options.get(cuisine.lower(), options["default"])
    return f"🍕 Найдено {location}: " + ", ".join(results)


def order_food(restaurant: str, items: str) -> str:
    """Place a food order."""
    order_id = abs(hash(restaurant + items)) % 10000
    return f"✅ Заказ #{order_id} в {restaurant}: {items}. Ожидайте через 30-40 минут!"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

async def main():
    print("\n" + "═"*70)
    print("🗣️ ORGANIC CONVERSATION TEST")
    print("═"*70)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return
    
    llm_client = AsyncOpenAI()
    emb_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    # Tools for food delivery
    tools = [
        Tool(
            name="search_food_delivery",
            description="Search for food delivery options by cuisine type",
            parameters={
                "type": "object",
                "properties": {
                    "cuisine": {"type": "string", "description": "Type of food (пицца, суши, бургер)"},
                    "location": {"type": "string", "description": "Location or 'рядом'"},
                },
                "required": ["cuisine"],
            },
            function=search_food_delivery,
        ),
        Tool(
            name="order_food",
            description="Place a food delivery order",
            parameters={
                "type": "object",
                "properties": {
                    "restaurant": {"type": "string"},
                    "items": {"type": "string"},
                },
                "required": ["restaurant", "items"],
            },
            function=order_food,
        ),
    ]
    
    # Empty storage - skills will be generated dynamically
    storage = InMemoryStorage()
    
    # Create agent with auto_generate_skills=True
    agent = SkillDialogueAgent(
        client=llm_client,
        storage=storage,
        tools=tools,
        embedding_client=emb_client,
        llm_model="gpt-4o-mini",
        embedding_model="bge-m3:latest",
        similarity_threshold=0.6,
        auto_generate_skills=True,  # KEY: generate skills dynamically
    )
    
    print("\n✅ Agent initialized (empty skill storage)")
    print("🎯 auto_generate_skills=True\n")
    
    # ───────────────────────────────────────────────────────────────
    # Phase 1: Casual conversation
    # ───────────────────────────────────────────────────────────────
    print("─"*70)
    print("📍 ФАЗА 1: Обычный разговор")
    print("─"*70)
    
    messages_phase1 = [
        "Привет! Как дела?",
        "Что интересного посоветуешь сделать вечером?",
        "А какой фильм сейчас стоит посмотреть?",
    ]
    
    for msg in messages_phase1:
        print(f"\n👤 User: {msg}")
        response = await agent.chat(msg)
        print(f"🤖 Agent: {response.message[:200]}{'...' if len(response.message) > 200 else ''}")
        if response.skill_generated:
            print(f"   [NEW SKILL GENERATED]")
        elif response.skill_used:
            print(f"   [SKILL REUSED: {response.skill_used.name}]")
    
    # ───────────────────────────────────────────────────────────────
    # Phase 2: Food delivery need emerges
    # ───────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("📍 ФАЗА 2: Хочется перекусить")
    print("─"*70)
    
    messages_phase2 = [
        "Слушай, я что-то проголодался. Где можно заказать поесть?",
        "Хочу пиццу!",
        "Закажи мне пепперони в Додо",
    ]
    
    for msg in messages_phase2:
        print(f"\n👤 User: {msg}")
        response = await agent.chat(msg)
        print(f"🤖 Agent: {response.message[:200]}{'...' if len(response.message) > 200 else ''}")
        if response.skill_generated:
            print(f"   [NEW SKILL GENERATED]")
        elif response.skill_used:
            print(f"   [SKILL REUSED: {response.skill_used.name}]")
        if response.tools_called:
            print(f"   [TOOLS: {[t.tool_name for t in response.tools_called]}]")
    
    # ───────────────────────────────────────────────────────────────
    # Phase 3: Back to casual
    # ───────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("📍 ФАЗА 3: Продолжение разговора")
    print("─"*70)
    
    messages_phase3 = [
        "Спасибо! Пока жду пиццу, расскажи что-нибудь интересное",
        "А как думаешь, будет ли дождь завтра?",
    ]
    
    for msg in messages_phase3:
        print(f"\n👤 User: {msg}")
        response = await agent.chat(msg)
        print(f"🤖 Agent: {response.message[:200]}{'...' if len(response.message) > 200 else ''}")
        if response.skill_generated:
            print(f"   [NEW SKILL GENERATED]")
        elif response.skill_used:
            print(f"   [SKILL REUSED: {response.skill_used.name}]")
    
    # ───────────────────────────────────────────────────────────────
    # Phase 4: Another food request (should reuse skill)
    # ───────────────────────────────────────────────────────────────
    print("\n" + "─"*70)
    print("📍 ФАЗА 4: Ещё одна еда (проверка переиспользования)")
    print("─"*70)
    
    print("\n👤 User: Хочу заказать суши на вечер")
    response = await agent.chat("Хочу заказать суши на вечер")
    print(f"🤖 Agent: {response.message[:200]}{'...' if len(response.message) > 200 else ''}")
    if response.skill_generated:
        print(f"   [NEW SKILL GENERATED] ❌ Ожидался REUSE!")
    elif response.skill_used:
        print(f"   [SKILL REUSED: {response.skill_used.name}] ✅")
    if response.tools_called:
        print(f"   [TOOLS: {[t.tool_name for t in response.tools_called]}]")
    
    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "═"*70)
    print("📊 ИТОГИ")
    print("═"*70)
    
    all_skills = await storage.get_all()
    print(f"\n📚 Сгенерировано навыков: {len(all_skills)}")
    for skill in all_skills:
        print(f"\n   📌 {skill.name}")
        print(f"      Описание: {skill.metadata.description[:80]}...")
        print(f"      Шагов: {len(skill.steps)}")
        print(f"      Keywords: {skill.metadata.keywords[:5]}")
    
    print(f"\n💬 Сообщений в истории: {len(agent.conversation_history)}")
    
    print("\n" + "═"*70)
    print("✅ Тест завершён!")
    print("═"*70)


if __name__ == "__main__":
    asyncio.run(main())
