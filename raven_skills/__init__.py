"""raven-skills: Skill-based AI agent library.

A library for building adaptive AI agents with skill-based task solving.
The agent automatically selects the best skill for a task, or generates
new skills from successful conversations.

Example:
    ```python
    from openai import AsyncOpenAI
    from raven_skills import SkillAgent, InMemoryStorage
    
    agent = SkillAgent(
        client=AsyncOpenAI(),
        storage=InMemoryStorage(),
    )
    
    task, result = await agent.match("How to deploy to Kubernetes?")
    if result.found:
        execution = await agent.execute(result.skill, task)
    ```
"""

__version__ = "0.3.0"

# Main entry points
from raven_skills.agent import SkillAgent
from raven_skills.dialogue_agent import SkillDialogueAgent

# Models
from raven_skills.models.skill import Skill, SkillMetadata, SkillStep
from raven_skills.models.task import Task, TaskContext
from raven_skills.models.result import (
    MatchResult,
    ExecutionResult,
    RefinementAction,
    RefinementType,
)
from raven_skills.models.dialogue import Tool, DialogueResponse

# Interface (only storage remains abstract)
from raven_skills.interfaces.storage import SkillStorage

# Utilities
from raven_skills.utils.similarity import (
    cosine_similarity,
    normalize_embedding,
    euclidean_distance,
)

__all__ = [
    # Version
    "__version__",
    # Main classes
    "SkillAgent",
    "SkillDialogueAgent",
    # Models
    "Skill",
    "SkillMetadata",
    "SkillStep",
    "Task",
    "TaskContext",
    "MatchResult",
    "ExecutionResult",
    "RefinementAction",
    "RefinementType",
    "Tool",
    "DialogueResponse",
    # Interface
    "SkillStorage",
    # Utilities
    "cosine_similarity",
    "normalize_embedding",
    "euclidean_distance",
]
