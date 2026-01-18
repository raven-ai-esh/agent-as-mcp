"""Test configuration and fixtures."""

import uuid
from datetime import datetime
from typing import Any, TypeVar
from unittest.mock import MagicMock, AsyncMock

from pydantic import BaseModel

from raven_skills.interfaces.storage import SkillStorage
from raven_skills.models.skill import Skill, SkillMetadata, SkillStep
from raven_skills.models.task import Task, TaskContext
from raven_skills.utils.similarity import cosine_similarity
from raven_skills.prompts import (
    KeyAspectsResponse,
    SkillFromConversation,
    FailureDiagnosis,
    RefinedSkill,
    MergedSkill,
    StepExecutionResult,
    SkillMatchValidation,
)

T = TypeVar("T", bound=BaseModel)


# ─────────────────────────────────────────────────────────────────
# Mock OpenAI Client
# ─────────────────────────────────────────────────────────────────

class MockResponse(BaseModel):
    output_parsed: Any
    
    # Mock other attributes if needed
    choices: list = []

class MockEmbeddings:
    async def create(self, model: str, input: str | list[str]) -> Any:
        # Create deterministic embeddings based on text hash
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = input
            
        data = []
        for i, text in enumerate(inputs):
            embedding = _generate_embedding(text)
            mock_item = MagicMock()
            mock_item.embedding = embedding
            mock_item.index = i
            data.append(mock_item)
            
        mock_resp = MagicMock()
        mock_resp.data = data
        return mock_resp

class MockResponses:
    async def parse(
        self,
        model: str,
        input: list[dict],
        text_format: type[T],
    ) -> MockResponse:
        """Mock behavior for structured outputs."""
        prompt_content = input[-1]["content"] if input else ""
        
        # Decide what to return based on the schema type
        if text_format == KeyAspectsResponse:
            return MockResponse(output_parsed=KeyAspectsResponse(
                query_understanding="Test query understanding",
                domain="Test Domain",
                key_aspects=["aspect1", "aspect2", "aspect3"],
            ))
            
        elif text_format == SkillFromConversation:
            return MockResponse(output_parsed=SkillFromConversation(
                conversation_analysis="Analyzed conversation",
                is_generalizable=True,
                name="Generated Skill",
                description="Description of generated skill",
                goal="Goal of generated skill",
                keywords=["gen", "skill"],
                steps=["Step 1", "Step 2"],
            ))
            
        elif text_format == FailureDiagnosis:
            return MockResponse(output_parsed=FailureDiagnosis(
                skill_goal_analysis="Goal analysis",
                execution_analysis="Execution analysis",
                user_expectation_analysis="Expectation analysis",
                root_cause="wrong_steps",
                diagnosis="Diagnosis result",
                suggested_changes="New step needed",
            ))
            
        elif text_format == RefinedSkill:
            return MockResponse(output_parsed=RefinedSkill(
                problem_understanding="Understanding",
                changes_rationale="Rationale",
                name="Refined Skill",
                description="Refined description",
                goal="Refined goal",
                keywords=["refined"],
                steps=["Refined Step 1"],
            ))
            
        elif text_format == MergedSkill:
            return MockResponse(output_parsed=MergedSkill(
                overlap_analysis="Overlap",
                differences_analysis="Differences",
                merge_strategy="Strategy",
                name="Merged Skill",
                description="Merged description",
                goal="Merged goal",
                keywords=["merged"],
                steps=["Merged Step 1"],
            ))
            
        elif text_format == StepExecutionResult:
            return MockResponse(output_parsed=StepExecutionResult(
                understanding="Step understanding",
                approach="Step approach",
                result="Step execution result",
            ))
            
        elif text_format == SkillMatchValidation:
            return MockResponse(output_parsed=SkillMatchValidation(
                task_analysis="Task analysis",
                skill_analysis="Skill analysis",
                alignment_analysis="Alignment",
                is_good_match=True,
                confidence=0.95,
                reason="Good match",
            ))
            
        raise ValueError(f"Unknown schema type: {text_format}")

class MockAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        self.embeddings = MockEmbeddings()
        self.responses = MockResponses()


def _generate_embedding(text: str, dim: int = 128) -> list[float]:
    """Helper to generate deterministic embeddings."""
    import hashlib
    hash_bytes = hashlib.sha256(text.encode()).digest()
    embedding = []
    for i in range(dim):
        byte_val = hash_bytes[i % len(hash_bytes)]
        embedding.append((byte_val / 127.5) - 1.0)
    norm = sum(x * x for x in embedding) ** 0.5
    return [x / norm for x in embedding]


# ─────────────────────────────────────────────────────────────────
# In-Memory Storage
# ─────────────────────────────────────────────────────────────────

class InMemoryStorage(SkillStorage):
    """In-memory skill storage for testing."""

    def __init__(self) -> None:
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
        self,
        embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Skill, float]]:
        """Search by cosine similarity."""
        results: list[tuple[Skill, float]] = []
        
        for skill in self._skills.values():
            if not skill.metadata.embedding:
                continue
            
            score = cosine_similarity(embedding, skill.metadata.embedding)
            if score >= min_score:
                results.append((skill, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def create_test_skill(
    skill_id: str | None = None,
    name: str = "Test Skill",
    description: str = "A test skill",
    goal: str = "Complete the test",
    keywords: list[str] | None = None,
    steps: list[SkillStep] | None = None,
    embedding: list[float] | None = None,
) -> Skill:
    """Create a skill for testing."""
    if steps is None:
        steps = [SkillStep(order=0, instruction="Do test")]
    
    return Skill(
        id=skill_id or str(uuid.uuid4()),
        name=name,
        metadata=SkillMetadata(
            description=description,
            goal=goal,
            keywords=keywords or ["test"],
            embedding=embedding or [],
        ),
        steps=steps,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )

def create_test_task(
    task_id: str | None = None,
    query: str = "Test query",
    key_aspects: list[str] | None = None,
    embedding: list[float] | None = None,
) -> Task:
    """Create a task for testing."""
    return Task(
        id=task_id or str(uuid.uuid4()),
        query=query,
        key_aspects=key_aspects or ["test"],
        embedding=embedding or [],
        context=TaskContext(),
        created_at=datetime.now(),
    )
