"""Tests for SkillAgent core functionality (Generate, Optimize, Refine)."""

import pytest
from unittest.mock import MagicMock

from raven_skills.agent import SkillAgent
from raven_skills.models.result import ExecutionResult, RefinementType
from raven_skills.models.skill import SkillMetadata

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


class TestSkillGeneration:
    """Tests for agent.generate_skill()."""

    @pytest.mark.asyncio
    async def test_generate_from_conversation(
        self,
        agent: SkillAgent,
    ) -> None:
        task = create_test_task(query="How to deploy the app?")
        conversation = [
            {"role": "user", "content": "How to deploy?"},
            {"role": "assistant", "content": "First, build the app..."},
        ]
        
        skill = await agent.generate_skill(
            task=task,
            conversation=conversation,
            final_result="Deployed successfully",
        )
        
        assert skill.id is not None
        assert skill.name == "Generated Skill"
        assert len(skill.steps) == 2
        
        # Should be saved
        saved = await agent.storage.get(skill.id)
        assert saved is not None

    @pytest.mark.asyncio
    async def test_generate_without_auto_save(
        self,
        agent: SkillAgent,
    ) -> None:
        task = create_test_task()
        
        skill = await agent.generate_skill(
            task=task,
            conversation=[],
            final_result="Result",
            auto_save=False,
        )
        
        saved = await agent.storage.get(skill.id)
        assert saved is None


class TestSkillOptimization:
    """Tests for agent.optimize()."""

    @pytest.mark.asyncio
    async def test_optimize_finds_and_merges(
        self,
        agent: SkillAgent,
    ) -> None:
        # Create two similar skills (embeddings mocked to return same hash for same text)
        skill1 = create_test_skill(skill_id="skill-1", name="Skill 1")
        skill2 = create_test_skill(skill_id="skill-2", name="Skill 2")
        
        # Force same embedding
        embedding = [0.1] * 128
        skill1.metadata.embedding = embedding
        skill2.metadata.embedding = embedding
        
        await agent.storage.save(skill1)
        await agent.storage.save(skill2)
        
        # Optimize with actual merge (dry_run=False)
        results = await agent.optimize(similarity_threshold=0.90, dry_run=False)
        
        assert len(results) == 1
        original_skills, merged_skill = results[0]
        
        assert len(original_skills) == 2
        assert merged_skill is not None
        assert merged_skill.name == "Merged Skill"
        
        # Originals should be deleted
        assert await agent.storage.get("skill-1") is None
        assert await agent.storage.get("skill-2") is None


class TestSkillRefinement:
    """Tests for agent.diagnose() and agent.refine()."""

    @pytest.mark.asyncio
    async def test_diagnose_returns_refinement_action(
        self,
        agent: SkillAgent,
    ) -> None:
        skill = create_test_skill()
        task = create_test_task()
        result = ExecutionResult(success=False, output=None)
        
        action = await agent.diagnose(
            skill=skill,
            task=task,
            result=result,
            user_feedback="The steps were wrong",
        )
        
        # Mock returns "wrong_steps" -> EDIT_SKILL
        assert action.skill_id == skill.id
        assert action.type == RefinementType.EDIT_SKILL

    @pytest.mark.asyncio
    async def test_refine_skill(
        self,
        agent: SkillAgent,
    ) -> None:
        skill = create_test_skill()
        # Ensure skill has embedding
        skill.metadata.embedding = [0.1] * 128
        await agent.storage.save(skill)
        
        diagnosis_action = await agent.diagnose(
            skill=skill,
            task=create_test_task(),
            result=ExecutionResult(success=False, output=None),
            user_feedback="Fix it",
        )
        
        refined = await agent.refine(skill, diagnosis_action)
        
        assert refined is not None
        assert refined.name == "Refined Skill"
        assert refined.version == skill.version + 1
        
        # Should be saved
        saved = await agent.storage.get(skill.id)
        assert saved is not None
        assert saved.version == skill.version + 1
