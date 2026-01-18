"""Result and action data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from raven_skills.models.skill import Skill, SkillStep


@dataclass
class MatchResult:
    """Result of matching a task to available skills.
    
    Attributes:
        skill: The matched skill, or None if no match was found.
        score: Similarity score of the match (0.0 to 1.0).
        threshold_passed: Whether the match exceeded the similarity threshold.
        alternatives: Other candidate skills with their scores.
    """
    skill: Skill | None
    score: float
    threshold_passed: bool
    alternatives: list[tuple[Skill, float]] = field(default_factory=list)

    @property
    def found(self) -> bool:
        """Whether a suitable skill was found."""
        return self.skill is not None and self.threshold_passed


@dataclass
class ExecutionResult:
    """Result of executing a skill.
    
    Attributes:
        success: Whether execution completed successfully.
        output: Final output/result of the execution.
        steps_completed: List of steps that were completed.
        error: Error message if execution failed.
        conversation_log: Log of messages during execution.
    """
    success: bool
    output: Any
    steps_completed: list[SkillStep] = field(default_factory=list)
    error: str | None = None
    conversation_log: list[dict[str, Any]] = field(default_factory=list)


class RefinementType(Enum):
    """Types of skill refinement actions.
    
    EDIT_SKILL: The skill itself has issues (wrong steps, inaccuracies).
    EDIT_MATCHING: The skill was incorrectly selected for this task.
    FORK_SKILL: The skill is correct but expectations don't match.
    """
    EDIT_SKILL = "edit_skill"
    EDIT_MATCHING = "edit_matching"
    FORK_SKILL = "fork_skill"


@dataclass
class RefinementAction:
    """A diagnosed refinement action to improve skills.
    
    Attributes:
        type: The type of refinement needed.
        skill_id: ID of the skill to refine.
        diagnosis: Explanation of what went wrong.
        suggested_changes: Specific changes to make (text description).
    """
    type: RefinementType
    skill_id: str
    diagnosis: str
    suggested_changes: str = ""
