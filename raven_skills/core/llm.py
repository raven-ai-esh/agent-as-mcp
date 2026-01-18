"""Internal LLM client wrapper.

Wraps OpenAI client and provides typed methods for all LLM operations.
All prompts and schemas are embedded in this module.

Supports both OpenAI (with structured outputs) and Ollama (with JSON parsing).
"""

import json
from typing import Any, TypeVar
from pydantic import BaseModel

from raven_skills.prompts import (
    # Schemas
    KeyAspectsResponse,
    SkillFromConversation,
    FailureDiagnosis,
    RefinedSkill,
    MergedSkill,
    StepExecutionResult,
    SkillMatchValidation,
    # Templates
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
from raven_skills.models.skill import Skill, SkillStep
from raven_skills.models.task import Task
from raven_skills.models.result import ExecutionResult, RefinementAction, RefinementType

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Internal LLM client that handles all model interactions.
    
    Wraps an OpenAI-compatible async client and provides typed methods
    for each operation using predefined prompts and SGR schemas.
    
    Supports:
    - OpenAI with structured outputs (responses.parse)
    - Ollama and other providers (chat.completions with JSON parsing)
    
    Args:
        client: An async OpenAI client instance (AsyncOpenAI or compatible).
        model: Model identifier to use for completions (default: gpt-4o-mini).
        use_structured_outputs: Use OpenAI structured outputs API. Set to False for Ollama.
    """
    
    def __init__(
        self,
        client: Any,
        model: str = "gpt-4o-mini",
        use_structured_outputs: bool = True,
    ):
        self.client = client
        self.model = model
        self.use_structured_outputs = use_structured_outputs
    
    def _schema_to_json_instruction(self, schema: type[T]) -> str:
        """Generate JSON instruction from Pydantic schema for non-structured output models."""
        fields = []
        for name, field in schema.model_fields.items():
            desc = field.description or ""
            fields.append(f'  "{name}": "{desc}"')
        return "Ответь строго в JSON формате:\n{\n" + ",\n".join(fields) + "\n}"
    
    async def _parse(
        self,
        system_prompt: str,
        user_prompt: str,
        response_schema: type[T],
    ) -> T:
        """Make a structured output request."""
        if self.use_structured_outputs:
            # OpenAI responses API (structured outputs)
            response = await self.client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=response_schema,
            )
            return response.output_parsed
        else:
            # Ollama / other OpenAI-compatible APIs via chat.completions.parse
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=response_schema,
            )
            return response.choices[0].message.parsed
    
    def _normalize_llm_output(self, data: dict, schema: type) -> dict:
        """Normalize LLM output to match expected schema types.
        
        Local LLMs sometimes return:
        - Lists as comma-separated strings
        - Booleans as "Да"/"Нет" or "true"/"false" strings
        This normalizes them to proper Python types.
        """
        # Normalize list fields
        list_fields = ["keywords", "steps", "key_aspects"]
        
        for field in list_fields:
            if field in data and isinstance(data[field], str):
                value = data[field]
                if "\n" in value and any(value.strip().startswith(f"{i}.") for i in range(1, 10)):
                    # Numbered list
                    lines = [line.strip() for line in value.split("\n") if line.strip()]
                    items = []
                    for line in lines:
                        for i in range(1, 20):
                            if line.startswith(f"{i}. "):
                                line = line[len(f"{i}. "):]
                                break
                        if line:
                            items.append(line)
                    data[field] = items
                else:
                    # Comma-separated
                    data[field] = [item.strip() for item in value.split(",") if item.strip()]
        
        # Normalize boolean fields
        bool_fields = ["is_generalizable", "is_good_match", "needs_user_input"]
        true_values = {"true", "да", "yes", "1", "верно", "правда"}
        false_values = {"false", "нет", "no", "0", "неверно", "ложь"}
        
        for field in bool_fields:
            if field in data and isinstance(data[field], str):
                value = data[field].lower().strip()
                # Check first word only
                first_word = value.split()[0] if value else ""
                if first_word in true_values or value.startswith("да"):
                    data[field] = True
                elif first_word in false_values or value.startswith("нет"):
                    data[field] = False
                else:
                    # Default to True if can't parse
                    data[field] = True
        
        return data
    
    # ─────────────────────────────────────────────────────────────
    # Key Aspects Extraction
    # ─────────────────────────────────────────────────────────────
    
    async def extract_key_aspects(self, query: str) -> KeyAspectsResponse:
        """Extract key aspects from a user query."""
        return await self._parse(
            system_prompt=EXTRACT_KEY_ASPECTS_SYSTEM,
            user_prompt=EXTRACT_KEY_ASPECTS_USER.format(query=query),
            response_schema=KeyAspectsResponse,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Skill Generation
    # ─────────────────────────────────────────────────────────────
    
    async def generate_skill_from_conversation(
        self,
        task: Task,
        conversation: list[dict[str, str]],
        final_result: str,
    ) -> SkillFromConversation:
        """Generate a skill definition from a conversation."""
        conv_text = "\n".join(
            f"{msg['role']}: {msg['content']}" 
            for msg in conversation
        )
        return await self._parse(
            system_prompt=GENERATE_SKILL_SYSTEM,
            user_prompt=GENERATE_SKILL_USER.format(
                query=task.query,
                conversation=conv_text,
                result=final_result,
            ),
            response_schema=SkillFromConversation,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Failure Diagnosis
    # ─────────────────────────────────────────────────────────────
    
    async def diagnose_failure(
        self,
        skill: Skill,
        task: Task,
        result: ExecutionResult,
        user_feedback: str,
    ) -> FailureDiagnosis:
        """Diagnose why a skill execution failed to meet expectations."""
        steps_text = "\n".join(
            f"{i+1}. {step.instruction}"
            for i, step in enumerate(skill.steps)
        )
        return await self._parse(
            system_prompt=DIAGNOSE_FAILURE_SYSTEM,
            user_prompt=DIAGNOSE_FAILURE_USER.format(
                skill_name=skill.name,
                skill_description=skill.metadata.description,
                skill_goal=skill.metadata.goal,
                skill_steps=steps_text,
                task_query=task.query,
                execution_output=result.output or "Нет вывода",
                execution_error=result.error or "Нет ошибки",
                user_feedback=user_feedback,
            ),
            response_schema=FailureDiagnosis,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Skill Refinement
    # ─────────────────────────────────────────────────────────────
    
    async def refine_skill(
        self,
        skill: Skill,
        action: RefinementAction,
    ) -> RefinedSkill:
        """Refine a skill based on diagnosis."""
        steps_text = "\n".join(
            f"{i+1}. {step.instruction}"
            for i, step in enumerate(skill.steps)
        )
        return await self._parse(
            system_prompt=REFINE_SKILL_SYSTEM,
            user_prompt=REFINE_SKILL_USER.format(
                skill_name=skill.name,
                skill_description=skill.metadata.description,
                skill_goal=skill.metadata.goal,
                skill_keywords=", ".join(skill.metadata.keywords),
                skill_steps=steps_text,
                refinement_type=action.type.value,
                diagnosis=action.diagnosis,
                suggested_changes=str(action.suggested_changes),
            ),
            response_schema=RefinedSkill,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Skill Merging
    # ─────────────────────────────────────────────────────────────
    
    async def merge_skills(self, skills: list[Skill]) -> MergedSkill:
        """Merge multiple similar skills into one."""
        descriptions = []
        for i, skill in enumerate(skills, 1):
            steps_text = "\n".join(
                f"  {j+1}. {step.instruction}"
                for j, step in enumerate(skill.steps)
            )
            descriptions.append(f"""## Навык {i}: {skill.name}
Описание: {skill.metadata.description}
Цель: {skill.metadata.goal}
Ключевые слова: {", ".join(skill.metadata.keywords)}
Шаги:
{steps_text}""")
        
        return await self._parse(
            system_prompt=MERGE_SKILLS_SYSTEM,
            user_prompt=MERGE_SKILLS_USER.format(
                skills_descriptions="\n\n".join(descriptions),
            ),
            response_schema=MergedSkill,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Step Execution
    # ─────────────────────────────────────────────────────────────
    
    async def execute_step(
        self,
        step: SkillStep,
        context: dict[str, Any],
    ) -> StepExecutionResult:
        """Execute a single skill step."""
        return await self._parse(
            system_prompt=EXECUTE_STEP_SYSTEM,
            user_prompt=EXECUTE_STEP_USER.format(
                step_instruction=step.instruction,
                expected_output=step.expected_output or "Не указан",
                context=json.dumps(context, ensure_ascii=False, default=str),
            ),
            response_schema=StepExecutionResult,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Match Validation
    # ─────────────────────────────────────────────────────────────
    
    async def validate_skill_match(
        self,
        task: Task,
        skill: Skill,
    ) -> SkillMatchValidation:
        """Validate if a skill is a good match for a task."""
        return await self._parse(
            system_prompt=VALIDATE_MATCH_SYSTEM,
            user_prompt=VALIDATE_MATCH_USER.format(
                task_query=task.query,
                task_aspects=", ".join(task.key_aspects),
                skill_name=skill.name,
                skill_description=skill.metadata.description,
                skill_goal=skill.metadata.goal,
                skill_keywords=", ".join(skill.metadata.keywords),
            ),
            response_schema=SkillMatchValidation,
        )
