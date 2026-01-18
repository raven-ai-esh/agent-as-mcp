"""Skill data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SkillMetadata:
    """Metadata describing a skill for matching and embedding.
    
    Attributes:
        description: Human-readable description of what the skill does.
        goal: Expected outcome/result when the skill is executed successfully.
        keywords: Key terms describing the main stages/aspects of the skill.
        embedding: Computed embedding vector for similarity matching.
    """
    description: str
    goal: str
    keywords: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding computation."""
        keywords_str = ", ".join(self.keywords) if self.keywords else ""
        return f"{self.description}\n\nGoal: {self.goal}\n\nKeywords: {keywords_str}"


@dataclass
class SkillStep:
    """A single step within a skill's execution flow.
    
    Attributes:
        order: Execution order (0-indexed).
        instruction: What to do in this step.
        expected_output: Optional description of expected output.
        metadata: Additional step-specific data.
    """
    order: int
    instruction: str
    expected_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Skill:
    """A skill that can be matched to tasks and executed.
    
    Attributes:
        id: Unique identifier.
        name: Human-readable name.
        metadata: Skill metadata for matching and embedding.
        steps: Ordered list of execution steps.
        version: Version number, incremented on updates.
        parent_id: ID of parent skill if this is a fork.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
    """
    id: str
    name: str
    metadata: SkillMetadata
    steps: list[SkillStep] = field(default_factory=list)
    version: int = 1
    parent_id: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def fork(self, new_id: str, new_goal: str | None = None) -> "Skill":
        """Create a forked copy of this skill.
        
        Args:
            new_id: ID for the forked skill.
            new_goal: Optional new goal to replace the original.
            
        Returns:
            A new Skill instance with parent_id set to this skill's ID.
        """
        new_metadata = SkillMetadata(
            description=self.metadata.description,
            goal=new_goal if new_goal else self.metadata.goal,
            keywords=self.metadata.keywords.copy(),
            embedding=[],  # Will need to be recomputed
        )
        return Skill(
            id=new_id,
            name=f"{self.name} (fork)",
            metadata=new_metadata,
            steps=[
                SkillStep(
                    order=s.order,
                    instruction=s.instruction,
                    expected_output=s.expected_output,
                    metadata=s.metadata.copy(),
                )
                for s in self.steps
            ],
            version=1,
            parent_id=self.id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
