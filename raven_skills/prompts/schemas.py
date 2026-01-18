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
        description="Краткое понимание сути запроса — что пользователь хочет сделать"
    )
    domain: str = Field(
        description="Предметная область запроса (например: DevOps, ML, Web, Data)"
    )
    key_aspects: list[str] = Field(
        description="3-5 ключевых аспектов запроса, определяющих подход к решению"
    )


# ─────────────────────────────────────────────────────────────────
# Skill Generation from Conversation
# ─────────────────────────────────────────────────────────────────

class SkillFromConversation(BaseModel):
    """SGR schema for extracting a reusable skill from a conversation."""
    
    conversation_analysis: str = Field(
        description="Анализ: какую проблему решали в диалоге и какой подход использовали"
    )
    is_generalizable: bool = Field(
        description="Можно ли обобщить этот диалог в переиспользуемый навык"
    )
    name: str = Field(
        description="Краткое название навыка (2-5 слов)"
    )
    description: str = Field(
        description="Полное описание того, что делает навык (1-2 предложения)"
    )
    goal: str = Field(
        description="Ожидаемый результат успешного выполнения навыка"
    )
    keywords: list[str] = Field(
        description="5-10 ключевых слов для поиска этого навыка"
    )
    steps: list[str] = Field(
        description="Последовательные шаги выполнения навыка"
    )


# ─────────────────────────────────────────────────────────────────
# Failure Diagnosis
# ─────────────────────────────────────────────────────────────────

class FailureDiagnosis(BaseModel):
    """SGR schema for diagnosing skill execution failures."""
    
    skill_goal_analysis: str = Field(
        description="Анализ: какова цель навыка и подходит ли она для задачи пользователя"
    )
    execution_analysis: str = Field(
        description="Анализ: что конкретно пошло не так при выполнении"
    )
    user_expectation_analysis: str = Field(
        description="Анализ: чего ожидал пользователь vs что он получил"
    )
    root_cause: Literal["wrong_steps", "wrong_selection", "wrong_expectations"] = Field(
        description=(
            "Корневая причина проблемы: "
            "wrong_steps — шаги навыка ошибочны, "
            "wrong_selection — навык выбран неправильно (матчинг), "
            "wrong_expectations — навык правильный, но ожидания не совпадают"
        )
    )
    diagnosis: str = Field(
        description="Итоговый диагноз проблемы (1-2 предложения)"
    )
    suggested_changes: str = Field(
        default="",
        description="Предлагаемые изменения в текстовом формате"
    )


# ─────────────────────────────────────────────────────────────────
# Skill Refinement
# ─────────────────────────────────────────────────────────────────

class RefinedSkill(BaseModel):
    """SGR schema for refining a skill based on feedback."""
    
    problem_understanding: str = Field(
        description="Понимание проблемы, которую нужно исправить"
    )
    changes_rationale: str = Field(
        description="Обоснование предлагаемых изменений"
    )
    name: str = Field(
        description="Обновлённое название навыка (может остаться прежним)"
    )
    description: str = Field(
        description="Обновлённое описание навыка"
    )
    goal: str = Field(
        description="Обновлённая цель навыка"
    )
    keywords: list[str] = Field(
        description="Обновлённые ключевые слова"
    )
    steps: list[str] = Field(
        description="Обновлённые шаги выполнения"
    )


# ─────────────────────────────────────────────────────────────────
# Skill Merging
# ─────────────────────────────────────────────────────────────────

class MergedSkill(BaseModel):
    """SGR schema for merging similar skills."""
    
    overlap_analysis: str = Field(
        description="Анализ: что общего между навыками"
    )
    differences_analysis: str = Field(
        description="Анализ: в чём различия между навыками"
    )
    merge_strategy: str = Field(
        description="Стратегия объединения: как совместить оба навыка"
    )
    name: str = Field(
        description="Название объединённого навыка"
    )
    description: str = Field(
        description="Описание объединённого навыка"
    )
    goal: str = Field(
        description="Цель объединённого навыка"
    )
    keywords: list[str] = Field(
        description="Ключевые слова объединённого навыка (объединение + новые)"
    )
    steps: list[str] = Field(
        description="Шаги объединённого навыка"
    )


# ─────────────────────────────────────────────────────────────────
# Step Execution
# ─────────────────────────────────────────────────────────────────

class StepExecutionResult(BaseModel):
    """SGR schema for executing a single skill step."""
    
    understanding: str = Field(
        description="Понимание задачи шага и контекста"
    )
    approach: str = Field(
        description="Подход к выполнению шага"
    )
    result: str = Field(
        description="Результат выполнения шага"
    )
    needs_user_input: bool = Field(
        default=False,
        description="Требуется ли дополнительный ввод от пользователя"
    )
    user_input_request: str | None = Field(
        default=None,
        description="Если требуется ввод — какой именно"
    )


# ─────────────────────────────────────────────────────────────────
# Skill Match Validation
# ─────────────────────────────────────────────────────────────────

class SkillMatchValidation(BaseModel):
    """SGR schema for validating if a skill matches a task."""
    
    task_analysis: str = Field(
        description="Анализ: что пользователь хочет сделать"
    )
    skill_analysis: str = Field(
        description="Анализ: что делает предложенный навык"
    )
    alignment_analysis: str = Field(
        description="Анализ: насколько навык подходит для задачи"
    )
    is_good_match: bool = Field(
        description="Подходит ли навык для задачи"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Уверенность в соответствии (0.0-1.0)"
    )
    reason: str = Field(
        description="Краткое обоснование решения"
    )
