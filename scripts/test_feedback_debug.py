"""Test feedback mechanism with debug output."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def read_file(path: str) -> str:
    """Simulated file reader."""
    return """Sheet1: Alice=100, Bob=200, Carol=300 (3 rows)"""


async def test_feedback_debug():
    from openai import AsyncOpenAI
    from raven_skills import SkillDialogueAgent, Tool
    from raven_skills.storage.json_storage import JSONStorage
    
    print("=" * 60)
    print("🧪 FEEDBACK DEBUG TEST")
    print("=" * 60)
    
    storage = JSONStorage("./test_debug_skills.json")
    client = AsyncOpenAI()
    
    file_tool = Tool(
        name="read_excel",
        description="Read data from Excel file",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        },
        function=read_file,
    )
    
    agent = SkillDialogueAgent(
        client=client,
        storage=storage,
        tools=[file_tool],
        auto_generate_skills=True,
    )
    
    # Step 1
    print("\n📩 Request: Read sales.xlsx")
    r1 = await agent.chat("Read data from sales.xlsx")
    print(f"✅ Response: {r1.message[:100]}...")
    print(f"   Skill: {r1.skill_used.name if r1.skill_used else None}")
    print(f"   _last_skill: {agent._last_skill.name if agent._last_skill else None}")
    print(f"   _last_task: {agent._last_task.query if agent._last_task else None}")
    
    # Step 2 - feedback
    print("\n📩 Feedback: Wrong, missed second sheet")
    
    # Check feedback detection
    is_negative = await agent._detect_negative_feedback("No, that's wrong!")
    print(f"   Negative feedback detected: {is_negative}")
    
    r2 = await agent.chat("No, that's wrong. You missed the second sheet!")
    print(f"\n✅ Response: {r2.message[:200]}...")
    print(f"   Skill refined: {r2.skill_refined}")
    print(f"   Skill used: {r2.skill_used.name if r2.skill_used else None}")
    
    # Cleanup
    if os.path.exists("./test_debug_skills.json"):
        os.remove("./test_debug_skills.json")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    asyncio.run(test_feedback_debug())
