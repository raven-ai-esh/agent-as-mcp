"""Test feedback mechanism with real dialogue.

Scenario:
1. Ask agent to read data from a file
2. Agent creates skill and executes
3. Provide negative feedback (missing data)
4. Agent refines skill and retries
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


# Simulated file reading tool
def read_file(path: str) -> str:
    """Simulated file reader - reads only first sheet."""
    return """Sheet1 Data:
- Row 1: Alice, 100
- Row 2: Bob, 200
- Row 3: Carol, 300
(3 rows from Sheet1)"""


def read_file_all_sheets(path: str) -> str:
    """Simulated file reader - reads all sheets."""
    return """Sheet1 Data:
- Row 1: Alice, 100
- Row 2: Bob, 200
- Row 3: Carol, 300

Sheet2 Data:
- Row 1: Dave, 400
- Row 2: Eve, 500
(5 rows total from 2 sheets)"""


async def test_feedback_flow():
    from openai import AsyncOpenAI
    from raven_skills import SkillDialogueAgent, Tool
    from raven_skills.storage.json_storage import JSONStorage
    
    print("=" * 60)
    print("🧪 FEEDBACK MECHANISM TEST")
    print("=" * 60)
    
    # Setup
    storage = JSONStorage("./test_feedback_skills.json")
    client = AsyncOpenAI()
    
    # Tool that simulates reading file (initially only first sheet)
    file_tool = Tool(
        name="read_excel",
        description="Read data from Excel file",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"}
            },
            "required": ["path"]
        },
        function=read_file,  # Initially reads only first sheet
    )
    
    agent = SkillDialogueAgent(
        client=client,
        storage=storage,
        tools=[file_tool],
        auto_generate_skills=True,
        llm_model="gpt-4o-mini",
    )
    
    # Step 1: Initial request
    print("\n" + "─" * 60)
    print("👤 USER: Read the data from sales.xlsx")
    print("─" * 60)
    
    response = await agent.chat("Read the data from sales.xlsx")
    
    print(f"\n🤖 AGENT: {response.message}")
    print(f"\n   Skill used: {response.skill_used.name if response.skill_used else 'None'}")
    print(f"   Skill generated: {response.skill_generated}")
    print(f"   Can retry: {response.can_retry}")
    print(f"   Tools called: {[tc.tool_name for tc in response.tools_called]}")
    
    # Step 2: Negative feedback
    print("\n" + "─" * 60)
    print("👤 USER: No, that's wrong. You only read the first sheet, "
          "but there are 2 sheets in this file!")
    print("─" * 60)
    
    response2 = await agent.chat(
        "No, that's wrong. You only read the first sheet, "
        "but there are 2 sheets in this file!"
    )
    
    print(f"\n🤖 AGENT: {response2.message}")
    print(f"\n   Skill used: {response2.skill_used.name if response2.skill_used else 'None'}")
    print(f"   Skill refined: {response2.skill_refined}")
    
    # Check results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    
    # Check if skill was updated
    skills = await storage.get_all()
    print(f"\nSkills in storage: {len(skills)}")
    for s in skills:
        print(f"  - {s.name} (v{s.version}): {s.metadata.description}")
        print(f"    Steps: {[step.instruction for step in s.steps]}")
    
    # Cleanup
    if os.path.exists("./test_feedback_skills.json"):
        os.remove("./test_feedback_skills.json")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    asyncio.run(test_feedback_flow())
