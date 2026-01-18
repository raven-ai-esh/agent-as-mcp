"""Tests for data models."""

import pytest
from datetime import datetime

from raven_skills.models.skill import Skill, SkillMetadata, SkillStep
from raven_skills.models.task import Task, TaskContext
from raven_skills.models.result import (
    MatchResult,
    ExecutionResult,
    RefinementAction,
    RefinementType,
)


class TestSkillMetadata:
    """Tests for SkillMetadata."""

    def test_to_embedding_text(self) -> None:
        metadata = SkillMetadata(
            description="Deploy application",
            goal="Application is deployed to production",
            keywords=["deploy", "production", "server"],
        )
        
        text = metadata.to_embedding_text()
        
        assert "Deploy application" in text
        assert "Application is deployed to production" in text
        assert "deploy, production, server" in text

    def test_empty_keywords(self) -> None:
        metadata = SkillMetadata(
            description="Test",
            goal="Test goal",
            keywords=[],
        )
        
        text = metadata.to_embedding_text()
        
        assert "Test" in text
        assert "Test goal" in text


class TestSkill:
    """Tests for Skill."""

    def test_fork_creates_copy(self) -> None:
        original = Skill(
            id="original-id",
            name="Original Skill",
            metadata=SkillMetadata(
                description="Original description",
                goal="Original goal",
                keywords=["keyword1", "keyword2"],
                embedding=[0.1, 0.2, 0.3],
            ),
            steps=[
                SkillStep(order=0, instruction="Step 1"),
                SkillStep(order=1, instruction="Step 2"),
            ],
            version=3,
        )
        
        forked = original.fork("forked-id", "New goal")
        
        assert forked.id == "forked-id"
        assert forked.parent_id == "original-id"
        assert forked.metadata.goal == "New goal"
        assert forked.metadata.description == original.metadata.description
        assert forked.metadata.embedding == []  # Should be recomputed
        assert forked.version == 1
        assert len(forked.steps) == len(original.steps)

    def test_fork_preserves_name_suffix(self) -> None:
        original = Skill(
            id="id",
            name="My Skill",
            metadata=SkillMetadata(description="", goal="", keywords=[]),
            steps=[],
        )
        
        forked = original.fork("new-id")
        
        assert "(fork)" in forked.name


class TestTask:
    """Tests for Task."""

    def test_to_embedding_text(self) -> None:
        task = Task(
            id="task-id",
            query="How to deploy the application?",
            key_aspects=["deployment", "production", "CI/CD"],
            context=TaskContext(user_id="user-1"),
        )
        
        text = task.to_embedding_text()
        
        assert "How to deploy the application?" in text
        assert "deployment, production, CI/CD" in text


class TestMatchResult:
    """Tests for MatchResult."""

    def test_found_when_threshold_passed(self) -> None:
        skill = Skill(
            id="id",
            name="Skill",
            metadata=SkillMetadata(description="", goal="", keywords=[]),
            steps=[],
        )
        
        result = MatchResult(
            skill=skill,
            score=0.85,
            threshold_passed=True,
        )
        
        assert result.found is True

    def test_not_found_when_threshold_not_passed(self) -> None:
        skill = Skill(
            id="id",
            name="Skill",
            metadata=SkillMetadata(description="", goal="", keywords=[]),
            steps=[],
        )
        
        result = MatchResult(
            skill=skill,
            score=0.5,
            threshold_passed=False,
        )
        
        assert result.found is False

    def test_not_found_when_no_skill(self) -> None:
        result = MatchResult(
            skill=None,
            score=0.0,
            threshold_passed=False,
        )
        
        assert result.found is False


class TestRefinementType:
    """Tests for RefinementType."""

    def test_enum_values(self) -> None:
        assert RefinementType.EDIT_SKILL.value == "edit_skill"
        assert RefinementType.EDIT_MATCHING.value == "edit_matching"
        assert RefinementType.FORK_SKILL.value == "fork_skill"
