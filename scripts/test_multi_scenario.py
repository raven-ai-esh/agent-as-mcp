"""Comprehensive multi-scenario test for skill learning and adaptation.

Tests how the agent learns and reuses skills across different domains:
1. Weather forecast
2. Restaurant booking
3. Flight search
4. Currency conversion

Each scenario includes clarifying questions and tool calls.
"""

import asyncio
import os
import sys
from datetime import datetime
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from raven_skills import SkillDialogueAgent, SkillStorage, Skill, Tool
from raven_skills.models.skill import SkillMetadata, SkillStep
from raven_skills.utils.similarity import cosine_similarity


# ═══════════════════════════════════════════════════════════════════
# Storage
# ═══════════════════════════════════════════════════════════════════

class InMemoryStorage(SkillStorage):
    def __init__(self):
        self._skills: dict[str, Skill] = {}
    
    async def save(self, skill: Skill) -> None:
        self._skills[skill.id] = skill
        print(f"   💾 Saved skill: {skill.name}")
    
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

def get_weather(city: str, date: str = "сегодня", include_wind: bool = False) -> str:
    """Get weather forecast."""
    temps = {"Москва": 3, "Воронеж": 5, "Сочи": 15, "Новосибирск": -10}
    temp = temps.get(city, 10)
    result = f"Погода в {city} на {date}: {temp}°C, облачно"
    if include_wind:
        result += ", ветер 8 м/с"
    return result


def book_restaurant(restaurant: str, date: str, guests: int, time: str = "19:00") -> str:
    """Book a restaurant table."""
    return f"✅ Столик забронирован: {restaurant}, {date} в {time}, {guests} гостей. Номер брони: R{abs(hash(restaurant)) % 10000}"


def search_flights(origin: str, destination: str, date: str, passengers: int = 1) -> str:
    """Search for flights."""
    prices = {"Москва-Сочи": 5500, "Москва-Питер": 3200, "Воронеж-Москва": 4100}
    key = f"{origin}-{destination}"
    price = prices.get(key, 6000)
    return f"✈️ Найден рейс {origin} → {destination} на {date}: от {price}₽ ({passengers} пассажиров)"


def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency."""
    rates = {"USD-RUB": 92.5, "EUR-RUB": 100.2, "RUB-USD": 0.011, "EUR-USD": 1.08}
    key = f"{from_currency}-{to_currency}"
    rate = rates.get(key, 1.0)
    result = amount * rate
    return f"💱 {amount} {from_currency} = {result:.2f} {to_currency} (курс: {rate})"


# ═══════════════════════════════════════════════════════════════════
# Pre-trained Skills
# ═══════════════════════════════════════════════════════════════════

async def create_skills(storage: InMemoryStorage, emb_client) -> dict[str, Skill]:
    """Create pre-trained skills with clarification steps."""
    from raven_skills.core.embeddings import EmbeddingsClient
    emb = EmbeddingsClient(emb_client, model="bge-m3:latest")
    
    skills = {}
    
    # Skill 1: Weather
    weather_emb = await emb.embed_text("погода прогноз температура ветер город дата")
    skills["weather"] = Skill(
        id="skill-weather",
        name="Прогноз погоды",
        version=1,
        metadata=SkillMetadata(
            description="Узнать прогноз погоды с уточнением даты и параметров",
            goal="Пользователь получает прогноз погоды",
            keywords=["погода", "прогноз", "температура", "ветер"],
            embedding=weather_emb,
        ),
        steps=[
            SkillStep(order=1, instruction="Уточни у пользователя дату прогноза"),
            SkillStep(order=2, instruction="Спроси, нужна ли информация о ветре"),
            SkillStep(order=3, instruction="Вызови get_weather с параметрами"),
        ],
        created_at=datetime.now(),
    )
    
    # Skill 2: Restaurant booking
    rest_emb = await emb.embed_text("ресторан бронь столик кафе заказать")
    skills["restaurant"] = Skill(
        id="skill-restaurant",
        name="Бронирование ресторана",
        version=1,
        metadata=SkillMetadata(
            description="Забронировать столик в ресторане",
            goal="Столик успешно забронирован",
            keywords=["ресторан", "бронь", "столик", "кафе"],
            embedding=rest_emb,
        ),
        steps=[
            SkillStep(order=1, instruction="Уточни название ресторана"),
            SkillStep(order=2, instruction="Спроси дату и время бронирования"),
            SkillStep(order=3, instruction="Узнай количество гостей"),
            SkillStep(order=4, instruction="Вызови book_restaurant с параметрами"),
        ],
        created_at=datetime.now(),
    )
    
    # Skill 3: Flight search
    flight_emb = await emb.embed_text("рейс самолет билет авиа перелет")
    skills["flight"] = Skill(
        id="skill-flight",
        name="Поиск авиабилетов",
        version=1,
        metadata=SkillMetadata(
            description="Найти и забронировать авиабилеты",
            goal="Найден подходящий рейс",
            keywords=["авиа", "рейс", "билет", "самолет", "перелет"],
            embedding=flight_emb,
        ),
        steps=[
            SkillStep(order=1, instruction="Уточни откуда и куда лететь"),
            SkillStep(order=2, instruction="Спроси дату вылета"),
            SkillStep(order=3, instruction="Узнай количество пассажиров"),
            SkillStep(order=4, instruction="Вызови search_flights"),
        ],
        created_at=datetime.now(),
    )
    
    # Skill 4: Currency conversion
    currency_emb = await emb.embed_text("валюта курс конвертация обмен доллар евро рубль")
    skills["currency"] = Skill(
        id="skill-currency",
        name="Конвертация валюты",
        version=1,
        metadata=SkillMetadata(
            description="Конвертировать валюту по текущему курсу",
            goal="Сумма успешно сконвертирована",
            keywords=["валюта", "курс", "конвертация", "обмен"],
            embedding=currency_emb,
        ),
        steps=[
            SkillStep(order=1, instruction="Уточни сумму для конвертации"),
            SkillStep(order=2, instruction="Спроси из какой валюты в какую"),
            SkillStep(order=3, instruction="Вызови convert_currency"),
        ],
        created_at=datetime.now(),
    )
    
    # Save all skills
    for skill in skills.values():
        await storage.save(skill)
    
    return skills


# ═══════════════════════════════════════════════════════════════════
# Dialogue Runner
# ═══════════════════════════════════════════════════════════════════

async def run_dialogue(agent: SkillDialogueAgent, messages: list[str], name: str) -> dict:
    """Run a multi-turn dialogue and return stats."""
    print(f"\n{'─'*60}")
    print(f"📝 Диалог: {name}")
    print(f"{'─'*60}")
    
    agent.reset()
    stats = {
        "name": name,
        "turns": 0,
        "skill_used": None,
        "skill_reused": False,
        "tools_called": [],
        "clarifications": 0,
    }
    
    for i, msg in enumerate(messages):
        print(f"\n👤 User: {msg}")
        response = await agent.chat(msg)
        print(f"🤖 Agent: {response.message[:150]}{'...' if len(response.message) > 150 else ''}")
        
        stats["turns"] += 1
        if response.skill_used:
            stats["skill_used"] = response.skill_used.name
            stats["skill_reused"] = not response.skill_generated
        if response.needs_user_input:
            stats["clarifications"] += 1
        if response.tools_called:
            stats["tools_called"].extend([t.tool_name for t in response.tools_called])
    
    return stats


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

async def main():
    print("\n" + "═"*70)
    print("🧪 MULTI-SCENARIO SKILL LEARNING TEST")
    print("═"*70)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return
    
    llm_client = AsyncOpenAI()
    emb_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    
    # Define tools
    tools = [
        Tool(name="get_weather", description="Get weather forecast", 
             parameters={"type": "object", "properties": {
                 "city": {"type": "string"}, "date": {"type": "string"}, 
                 "include_wind": {"type": "boolean"}
             }, "required": ["city"]}, function=get_weather),
        Tool(name="book_restaurant", description="Book restaurant table",
             parameters={"type": "object", "properties": {
                 "restaurant": {"type": "string"}, "date": {"type": "string"},
                 "guests": {"type": "integer"}, "time": {"type": "string"}
             }, "required": ["restaurant", "date", "guests"]}, function=book_restaurant),
        Tool(name="search_flights", description="Search for flights",
             parameters={"type": "object", "properties": {
                 "origin": {"type": "string"}, "destination": {"type": "string"},
                 "date": {"type": "string"}, "passengers": {"type": "integer"}
             }, "required": ["origin", "destination", "date"]}, function=search_flights),
        Tool(name="convert_currency", description="Convert currency",
             parameters={"type": "object", "properties": {
                 "amount": {"type": "number"}, "from_currency": {"type": "string"},
                 "to_currency": {"type": "string"}
             }, "required": ["amount", "from_currency", "to_currency"]}, function=convert_currency),
    ]
    
    # Create storage and pre-load skills
    storage = InMemoryStorage()
    skills = await create_skills(storage, emb_client)
    print(f"\n✅ Loaded {len(skills)} skills")
    
    # Create agent
    agent = SkillDialogueAgent(
        client=llm_client, storage=storage, tools=tools,
        embedding_client=emb_client, llm_model="gpt-4o-mini",
        embedding_model="bge-m3:latest", similarity_threshold=0.5,
        auto_generate_skills=False,
    )
    
    all_stats = []
    
    # ───────────────────────────────────────────────────────────────
    # Scenario 1: Weather (Воронеж)
    # ───────────────────────────────────────────────────────────────
    stats = await run_dialogue(agent, [
        "Какая погода в Воронеже?",
        "На завтра",
        "Да, покажи ветер",
    ], "Погода в Воронеже")
    all_stats.append(stats)
    
    # ───────────────────────────────────────────────────────────────
    # Scenario 2: Restaurant booking
    # ───────────────────────────────────────────────────────────────
    stats = await run_dialogue(agent, [
        "Хочу забронировать столик в ресторане",
        "В Пушкине",
        "На субботу вечером, в 20:00",
        "4 человека",
    ], "Бронь ресторана")
    all_stats.append(stats)
    
    # ───────────────────────────────────────────────────────────────
    # Scenario 3: Flight search
    # ───────────────────────────────────────────────────────────────
    stats = await run_dialogue(agent, [
        "Нужен билет на самолет",
        "Из Москвы в Сочи",
        "15 января",
        "Один пассажир",
    ], "Поиск авиабилета")
    all_stats.append(stats)
    
    # ───────────────────────────────────────────────────────────────
    # Scenario 4: Currency conversion
    # ───────────────────────────────────────────────────────────────
    stats = await run_dialogue(agent, [
        "Сколько будет в рублях 100 долларов?",
        "100 долларов",
        "В рубли",
    ], "Конвертация валюты")
    all_stats.append(stats)
    
    # ───────────────────────────────────────────────────────────────
    # Scenario 5: Weather (другой город) - SKILL REUSE
    # ───────────────────────────────────────────────────────────────
    stats = await run_dialogue(agent, [
        "Какой прогноз погоды в Сочи?",
        "Сегодня",
        "Нет, ветер не нужен",
    ], "Погода в Сочи (повторное использование)")
    all_stats.append(stats)
    
    # ───────────────────────────────────────────────────────────────
    # Scenario 6: Another restaurant - SKILL REUSE
    # ───────────────────────────────────────────────────────────────
    stats = await run_dialogue(agent, [
        "Забронируй столик",
        "White Rabbit",
        "В пятницу в 19:00",
        "2 гостя",
    ], "Бронь другого ресторана (повторное использование)")
    all_stats.append(stats)
    
    # ═══════════════════════════════════════════════════════════════
    # REPORT
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "═"*70)
    print("📊 ОТЧЁТ О ТЕСТИРОВАНИИ")
    print("═"*70)
    
    print("\n┌─────────────────────────────────────────────┬───────────┬─────────────┬──────────┬─────────────┐")
    print("│ Сценарий                                    │ Ходов     │ Уточнений   │ Tools    │ Skill Reuse │")
    print("├─────────────────────────────────────────────┼───────────┼─────────────┼──────────┼─────────────┤")
    for s in all_stats:
        name = s["name"][:43].ljust(43)
        turns = str(s["turns"]).center(9)
        clars = str(s["clarifications"]).center(11)
        tools = str(len(s["tools_called"])).center(8)
        reuse = ("✅" if s["skill_reused"] else "—").center(11)
        print(f"│ {name} │ {turns} │ {clars} │ {tools} │ {reuse} │")
    print("└─────────────────────────────────────────────┴───────────┴─────────────┴──────────┴─────────────┘")
    
    print("\n📚 Навыки в хранилище:")
    all_skills = await storage.get_all()
    for skill in all_skills:
        print(f"   • {skill.name} ({len(skill.steps)} шагов)")
    
    print("\n" + "═"*70)
    print("✅ Тестирование завершено!")
    print("═"*70)


if __name__ == "__main__":
    asyncio.run(main())
