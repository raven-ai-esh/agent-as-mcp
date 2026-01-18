"""Tests for SkillAgent matching functionality."""

import pytest
from unittest.mock import MagicMock

from raven_skills.agent import SkillAgent
from raven_skills.models.task import TaskContext

from tests.conftest import (
    MockAsyncOpenAI,
    InMemoryStorage,
    create_test_skill,
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
        similarity_threshold=0.75,
    )


class TestSkillMatching:
    """Tests for agent.match() and agent.prepare_task()."""

    @pytest.mark.asyncio
    async def test_prepare_task_extracts_aspects(
        self,
        agent: SkillAgent,
    ) -> None:
        task = await agent.prepare_task("How to deploy the application?")
        
        assert task.query == "How to deploy the application?"
        assert len(task.key_aspects) > 0
        assert len(task.embedding) > 0

    @pytest.mark.asyncio
    async def test_prepare_task_with_context(
        self,
        agent: SkillAgent,
    ) -> None:
        context = TaskContext(user_id="user-123", session_id="session-456")
        
        task = await agent.prepare_task("Test query", context)
        
        assert task.context.user_id == "user-123"
        assert task.context.session_id == "session-456"

    @pytest.mark.asyncio
    async def test_match_finds_similar_skill(
        self,
        agent: SkillAgent,
    ) -> None:
        # Create and store a skill with a known embedding
        # In our mock, "same text" -> "same embedding"
        # Skill embedding is generated from description + goal + keywords
        skill = create_test_skill(
            name="Deploy App",
            description="Deploy to prod",
            goal="Running in prod",
            keywords=["deploy"],
        )
        # Manually set embedding to match what we expect for the task
        # (This is a bit tricky with mocks, so we rely on the mock simply working
        # if we inject the embedding directly)
        embedding = [0.1] * 128
        skill.metadata.embedding = embedding
        await agent.storage.save(skill)
        
        # Prepare task and force same embedding
        task = await agent.prepare_task("How to deploy?")
        task.embedding = embedding
        
        # Match
        result = await agent.match_task(task)
        
        assert result.found is True
        assert result.skill is not None
        assert result.skill.id == skill.id

    @pytest.mark.asyncio
    async def test_match_returns_not_found_when_below_threshold(
        self,
        agent: SkillAgent,
    ) -> None:
        # Create a skill
        skill = create_test_skill(name="Cooking")
        skill.metadata.embedding = [0.1] * 128 # "Cooking" embedding
        await agent.storage.save(skill)
        
        # Create task with orthogonal embedding
        task = await agent.prepare_task("Coding")
        task.embedding = [-0.1] * 128 # "Coding", very different
        
        result = await agent.match_task(task)
        
        assert result.found is False
        assert result.skill is None

    @pytest.mark.asyncio
    async def test_match_convenience_method(
        self,
        agent: SkillAgent,
    ) -> None:
        task, result = await agent.match("Test query")
        
        assert task is not None
        assert result is not None
        assert task.query == "Test query"
