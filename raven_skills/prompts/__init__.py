"""Prompts module - SGR schemas and prompt templates."""

from raven_skills.prompts.schemas import (
    KeyAspectsResponse,
    SkillFromConversation,
    FailureDiagnosis,
    RefinedSkill,
    MergedSkill,
    StepExecutionResult,
    SkillMatchValidation,
)

from raven_skills.prompts.templates import (
    EXTRACT_KEY_ASPECTS_SYSTEM,
    EXTRACT_KEY_ASPECTS_USER,
    GENERATE_SKILL_SYSTEM,
    GENERATE_SKILL_USER,
    DIAGNOSE_FAILURE_SYSTEM,
    DIAGNOSE_FAILURE_USER,
    REFINE_SKILL_SYSTEM,
    REFINE_SKILL_USER,
    MERGE_SKILLS_SYSTEM,
    MERGE_SKILLS_USER,
    EXECUTE_STEP_SYSTEM,
    EXECUTE_STEP_USER,
    VALIDATE_MATCH_SYSTEM,
    VALIDATE_MATCH_USER,
)

__all__ = [
    # Schemas
    "KeyAspectsResponse",
    "SkillFromConversation",
    "FailureDiagnosis",
    "RefinedSkill",
    "MergedSkill",
    "StepExecutionResult",
    "SkillMatchValidation",
    # Templates
    "EXTRACT_KEY_ASPECTS_SYSTEM",
    "EXTRACT_KEY_ASPECTS_USER",
    "GENERATE_SKILL_SYSTEM",
    "GENERATE_SKILL_USER",
    "DIAGNOSE_FAILURE_SYSTEM",
    "DIAGNOSE_FAILURE_USER",
    "REFINE_SKILL_SYSTEM",
    "REFINE_SKILL_USER",
    "MERGE_SKILLS_SYSTEM",
    "MERGE_SKILLS_USER",
    "EXECUTE_STEP_SYSTEM",
    "EXECUTE_STEP_USER",
    "VALIDATE_MATCH_SYSTEM",
    "VALIDATE_MATCH_USER",
]
