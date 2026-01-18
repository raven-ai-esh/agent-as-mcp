"""Tests for SkillAgent execution functionality."""

import pytest
from unittest.mock import AsyncMock, patch

from raven_skills.agent import SkillAgent
from raven_skills.models.skill import SkillStep

from tests.conftest import (
    MockAsyncOpenAI,
    InMemoryStorage,
    create_test_skill,
    create_test_task,
)


@pytest.fixture
def agent() -> SkillAgent:
    client = MockAsyncOpenAI()
    storage = InMemoryStorage()
    return SkillAgent(
        client=client,
        storage=storage,
        llm_model="gpt-test",
        embedding_model="text-embedding-test",
    )


class TestSkillExecution:
    """Tests for agent.execute()."""

    @pytest.mark.asyncio
    async def test_execute_all_steps_success(
        self,
        agent: SkillAgent,
    ) -> None:
        skill = create_test_skill(
            steps=[
                SkillStep(order=0, instruction="First step"),
                SkillStep(order=1, instruction="Second step"),
            ]
        )
        task = create_test_task()
        
        result = await agent.execute(skill, task)
        
        assert result.success is True
        assert result.error is None
        assert len(result.steps_completed) == 2
        
        # Check conversion log
        assert len(result.conversation_log) > 0

    @pytest.mark.asyncio
    async def test_execute_catches_exceptions(
        self,
        agent: SkillAgent,
    ) -> None:
        skill = create_test_skill(
            steps=[SkillStep(order=0, instruction="Step 1")]
        )
        task = create_test_task()
        
        # Mock _llm.execute_step to raise exception
        with patch.object(
            agent._llm, 
            "execute_step", 
            side_effect=Exception("LLM Error")
        ):
            result = await agent.execute(skill, task)
            
            assert result.success is False
            assert result.error == "LLM Error"
            assert len(result.steps_completed) == 0

    @pytest.mark.asyncio
    async def test_execute_with_initial_context(
        self,
        agent: SkillAgent,
    ) -> None:
        skill = create_test_skill(
            steps=[SkillStep(order=0, instruction="Step 1")]
        )
        task = create_test_task()
        
        # We can't easily verify context was passed without inspecting the mock call args
        # But we can at least ensure it runs without error
        result = await agent.execute(
            skill, 
            task, 
            initial_context={"custom_key": "custom_value"},
        )
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_logs_conversation(
        self,
        agent: SkillAgent,
    ) -> None:
        skill = create_test_skill(
            steps=[SkillStep(order=0, instruction="Do something")]
        )
        task = create_test_task()
        
        result = await agent.execute(skill, task)
        
        assert len(result.conversation_log) > 0
        roles = [log["role"] for log in result.conversation_log]
        assert "system" in roles  # Instruction
        assert "assistant" in roles  # Result
