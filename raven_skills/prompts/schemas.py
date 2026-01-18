"""Pydantic schemas for Schema-Guided Reasoning (SGR).

These schemas enforce structured reasoning through predefined steps,
improving accuracy and reproducibility of LLM outputs.
"""

from pydantic import BaseModel, Field
from typing import Literal


# ─────────────────────────────────────────────────────────────────
# Key Aspects Extraction
# ─────────────────────────────────────────────────────────────────

class KeyAspectsResponse(BaseModel):
    """SGR schema for extracting key aspects from a user query."""
    
    query_understanding: str = Field(
        description="Brief understanding of the query essence — what the user wants to do"
    )
    domain: str = Field(
        description="Subject area of the query (e.g., DevOps, ML, Web, Data)"
    )
    key_aspects: list[str] = Field(
        description="3-5 key aspects of the query that determine the approach to solving it"
    )


# ─────────────────────────────────────────────────────────────────
# Skill Generation from Conversation
# ─────────────────────────────────────────────────────────────────

class SkillFromConversation(BaseModel):
    """SGR schema for extracting a reusable skill from a conversation."""
    
    conversation_analysis: str = Field(
        description="Analysis: what problem was solved in the dialogue and what approach was used"
    )
    is_generalizable: bool = Field(
        description="Can this dialogue be generalized into a reusable skill"
    )
    name: str = Field(
        description="Short skill name (2-5 words)"
    )
    description: str = Field(
        description="Full description of what the skill does (1-2 sentences)"
    )
    goal: str = Field(
        description="Expected result of successful skill execution"
    )
    keywords: list[str] = Field(
        description="5-10 keywords for finding this skill"
    )
    steps: list[str] = Field(
        description="Sequential steps for skill execution"
    )


# ─────────────────────────────────────────────────────────────────
# Failure Diagnosis
# ─────────────────────────────────────────────────────────────────

class FailureDiagnosis(BaseModel):
    """SGR schema for diagnosing skill execution failures."""
    
    skill_goal_analysis: str = Field(
        description="Analysis: what is the goal of the skill and does it fit the user's task"
    )
    execution_analysis: str = Field(
        description="Analysis: what specifically went wrong during execution"
    )
    user_expectation_analysis: str = Field(
        description="Analysis: what the user expected vs what they got"
    )
    root_cause: Literal["wrong_steps", "wrong_selection", "wrong_expectations"] = Field(
        description=(
            "Root cause of the problem: "
            "wrong_steps — skill steps are incorrect, "
            "wrong_selection — skill was selected incorrectly (matching), "
            "wrong_expectations — skill is correct, but expectations don't match"
        )
    )
    diagnosis: str = Field(
        description="Final problem diagnosis (1-2 sentences)"
    )
    suggested_changes: str = Field(
        default="",
        description="Suggested changes in text format"
    )


# ─────────────────────────────────────────────────────────────────
# Skill Refinement
# ─────────────────────────────────────────────────────────────────

class RefinedSkill(BaseModel):
    """SGR schema for refining a skill based on feedback."""
    
    problem_understanding: str = Field(
        description="Understanding of the problem that needs to be fixed"
    )
    changes_rationale: str = Field(
        description="Rationale for the proposed changes"
    )
    name: str = Field(
        description="Updated skill name (may remain the same)"
    )
    description: str = Field(
        description="Updated skill description"
    )
    goal: str = Field(
        description="Updated skill goal"
    )
    keywords: list[str] = Field(
        description="Updated keywords"
    )
    steps: list[str] = Field(
        description="Updated execution steps"
    )


# ─────────────────────────────────────────────────────────────────
# Skill Merging
# ─────────────────────────────────────────────────────────────────

class MergedSkill(BaseModel):
    """SGR schema for merging similar skills."""
    
    overlap_analysis: str = Field(
        description="Analysis: what the skills have in common"
    )
    differences_analysis: str = Field(
        description="Analysis: what are the differences between the skills"
    )
    merge_strategy: str = Field(
        description="Merge strategy: how to combine both skills"
    )
    name: str = Field(
        description="Name of the merged skill"
    )
    description: str = Field(
        description="Description of the merged skill"
    )
    goal: str = Field(
        description="Goal of the merged skill"
    )
    keywords: list[str] = Field(
        description="Keywords of the merged skill (combined + new)"
    )
    steps: list[str] = Field(
        description="Steps of the merged skill"
    )


# ─────────────────────────────────────────────────────────────────
# Step Execution
# ─────────────────────────────────────────────────────────────────

class StepExecutionResult(BaseModel):
    """SGR schema for executing a single skill step."""
    
    understanding: str = Field(
        description="Understanding of the step task and context"
    )
    approach: str = Field(
        description="Approach to executing the step"
    )
    result: str = Field(
        description="Result of step execution"
    )
    needs_user_input: bool = Field(
        default=False,
        description="Is additional user input required"
    )
    user_input_request: str | None = Field(
        default=None,
        description="If input is required — what exactly"
    )


# ─────────────────────────────────────────────────────────────────
# Skill Match Validation
# ─────────────────────────────────────────────────────────────────

class SkillMatchValidation(BaseModel):
    """SGR schema for validating if a skill matches a task."""
    
    task_analysis: str = Field(
        description="Analysis: what the user wants to do"
    )
    skill_analysis: str = Field(
        description="Analysis: what the proposed skill does"
    )
    alignment_analysis: str = Field(
        description="Analysis: how well the skill fits the task"
    )
    is_good_match: bool = Field(
        description="Does the skill fit the task"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the match (0.0-1.0)"
    )
    reason: str = Field(
        description="Brief justification for the decision"
    )
