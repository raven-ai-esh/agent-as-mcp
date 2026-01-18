"""SkillAgent - Main entry point for the raven-skills library.

Provides a unified interface for skill matching, execution, generation,
refinement, and optimization. User only needs to provide an OpenAI client.
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from raven_skills.core.llm import LLMClient
from raven_skills.core.embeddings import EmbeddingsClient
from raven_skills.interfaces.storage import SkillStorage
from raven_skills.models.skill import Skill, SkillMetadata, SkillStep
from raven_skills.models.task import Task, TaskContext
from raven_skills.models.result import (
    MatchResult,
    ExecutionResult,
    RefinementAction,
    RefinementType,
)
from raven_skills.utils.similarity import cosine_similarity


class SkillAgent:
    """Main agent class for skill-based task solving.
    
    This is the primary interface for the raven-skills library.
    It handles skill matching, execution, generation, refinement, and optimization
    using predefined prompts and SGR schemas.
    
    Args:
        client: An async OpenAI client instance (AsyncOpenAI) for LLM operations.
        storage: A SkillStorage implementation for persisting skills.
        embedding_client: Optional separate client for embeddings (e.g., Ollama).
                         If not provided, uses the main client.
        llm_model: Model to use for LLM operations (default: gpt-4o-mini).
        embedding_model: Model for embeddings (default: text-embedding-3-small).
        similarity_threshold: Minimum similarity score for skill matching (default: 0.75).
        validate_matches: Whether to use LLM validation for matches (default: True).
    
    Example:
        ```python
        from openai import AsyncOpenAI
        from raven_skills import SkillAgent
        
        # Simple: one client for everything
        agent = SkillAgent(
            client=AsyncOpenAI(),
            storage=my_storage,
        )
        
        # Advanced: separate clients for LLM and embeddings
        agent = SkillAgent(
            client=AsyncOpenAI(),  # For LLM (GPT)
            embedding_client=AsyncOpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),  # For embeddings (Ollama)
            storage=my_storage,
            embedding_model="bge-m3:latest",
        )
        
        task, result = await agent.match("How to deploy to Kubernetes?")
        if result.found:
            execution = await agent.execute(result.skill, task)
        ```
    """
    
    def __init__(
        self,
        client: Any,
        storage: SkillStorage,
        embedding_client: Any | None = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.75,
        validate_matches: bool = True,
        use_structured_outputs: bool = True,
    ):
        self.storage = storage
        self.similarity_threshold = similarity_threshold
        self.validate_matches = validate_matches
        
        # Internal clients
        self._llm = LLMClient(
            client, 
            model=llm_model,
            use_structured_outputs=use_structured_outputs,
        )
        # Use separate embedding client if provided, otherwise use main client
        self._embeddings = EmbeddingsClient(
            embedding_client or client, 
            model=embedding_model
        )
    
    # ─────────────────────────────────────────────────────────────
    # Skill Matching
    # ─────────────────────────────────────────────────────────────
    
    async def prepare_task(
        self,
        query: str,
        context: TaskContext | None = None,
    ) -> Task:
        """Prepare a task by extracting key aspects and generating embedding.
        
        Args:
            query: The user's query/request.
            context: Optional task context (user_id, session_id, etc.).
        
        Returns:
            A prepared Task object with extracted aspects and embedding.
        """
        # Extract key aspects using LLM
        aspects_response = await self._llm.extract_key_aspects(query)
        
        # Build embedding text
        embedding_text = f"{query}\n{' '.join(aspects_response.key_aspects)}"
        embedding = await self._embeddings.embed_text(embedding_text)
        
        return Task(
            id=str(uuid4()),
            query=query,
            key_aspects=aspects_response.key_aspects,
            embedding=embedding,
            context=context or TaskContext(),
            created_at=datetime.now(),
        )
    
    async def match(
        self,
        query: str,
        context: TaskContext | None = None,
        top_k: int = 5,
    ) -> tuple[Task, MatchResult]:
        """Find the best matching skill for a query.
        
        This is a convenience method that combines prepare_task and match_task.
        
        Args:
            query: The user's query/request.
            context: Optional task context.
            top_k: Number of candidate skills to consider.
        
        Returns:
            A tuple of (prepared Task, MatchResult).
        """
        task = await self.prepare_task(query, context)
        result = await self.match_task(task, top_k)
        return task, result
    
    async def match_task(
        self,
        task: Task,
        top_k: int = 5,
    ) -> MatchResult:
        """Find the best matching skill for a prepared task.
        
        Args:
            task: A prepared Task object with embedding.
            top_k: Number of candidate skills to consider.
        
        Returns:
            MatchResult with the best skill (if found) and alternatives.
        """
        # Search for similar skills
        candidates = await self.storage.search_by_embedding(
            embedding=task.embedding,
            top_k=top_k,
            min_score=0.0,
        )
        
        if not candidates:
            return MatchResult(
                skill=None,
                score=0.0,
                threshold_passed=False,
                alternatives=[],
            )
        
        best_skill, best_score = candidates[0]
        alternatives = candidates[1:] if len(candidates) > 1 else []
        
        # Check if above threshold
        threshold_passed = best_score >= self.similarity_threshold
        
        # Optional LLM validation
        if threshold_passed and self.validate_matches:
            validation = await self._llm.validate_skill_match(task, best_skill)
            if not validation.is_good_match:
                threshold_passed = False
                best_score = validation.confidence
        
        return MatchResult(
            skill=best_skill if threshold_passed else None,
            score=best_score,
            threshold_passed=threshold_passed,
            alternatives=alternatives,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Skill Execution
    # ─────────────────────────────────────────────────────────────
    
    async def execute(
        self,
        skill: Skill,
        task: Task,
        initial_context: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """Execute a skill for a given task.
        
        Runs each step of the skill sequentially, passing context between steps.
        
        Args:
            skill: The skill to execute.
            task: The task being solved.
            initial_context: Optional initial context for the first step.
        
        Returns:
            ExecutionResult with success status, output, and conversation log.
        """
        context = {
            "task_query": task.query,
            "task_aspects": task.key_aspects,
            "skill_goal": skill.metadata.goal,
            **(initial_context or {}),
        }
        
        conversation_log: list[dict] = []
        steps_completed: list[SkillStep] = []
        
        for step in sorted(skill.steps, key=lambda s: s.order):
            # Log the step
            conversation_log.append({
                "role": "system",
                "content": f"Executing step {step.order}: {step.instruction}",
                "step_order": step.order,
            })
            
            try:
                # Execute step
                result = await self._llm.execute_step(step, context)
                
                # Log result
                conversation_log.append({
                    "role": "assistant",
                    "content": result.result,
                    "step_order": step.order,
                    "needs_user_input": result.needs_user_input,
                })
                
                # Update context
                context[f"step_{step.order}_result"] = result.result
                context["last_step_result"] = result.result
                
                steps_completed.append(step)
                
                # Handle user input request
                if result.needs_user_input:
                    return ExecutionResult(
                        success=False,
                        output=result.user_input_request,
                        steps_completed=steps_completed,
                        error="User input required",
                        conversation_log=conversation_log,
                    )
                
            except Exception as e:
                conversation_log.append({
                    "role": "error",
                    "content": str(e),
                    "step_order": step.order,
                })
                return ExecutionResult(
                    success=False,
                    output=None,
                    steps_completed=steps_completed,
                    error=str(e),
                    conversation_log=conversation_log,
                )
        
        # All steps completed successfully
        final_output = context.get("last_step_result", "")
        
        return ExecutionResult(
            success=True,
            output=final_output,
            steps_completed=steps_completed,
            error=None,
            conversation_log=conversation_log,
        )
    
    # ─────────────────────────────────────────────────────────────
    # Skill Generation
    # ─────────────────────────────────────────────────────────────
    
    async def generate_skill(
        self,
        task: Task,
        conversation: list[dict[str, str]],
        final_result: str,
        auto_save: bool = True,
    ) -> Skill:
        """Generate a new skill from a successful conversation.
        
        Args:
            task: The original task that was solved.
            conversation: The conversation log.
            final_result: The final result of the conversation.
            auto_save: Whether to automatically save the skill to storage.
        
        Returns:
            The generated Skill.
        """
        # Generate skill using LLM
        skill_data = await self._llm.generate_skill_from_conversation(
            task=task,
            conversation=conversation,
            final_result=final_result,
        )
        
        if not skill_data.is_generalizable:
            # Still create the skill but mark it
            skill_data.keywords.append("specific-case")
        
        # Generate embedding for the skill
        embedding_text = f"{skill_data.description}\n{skill_data.goal}\n{' '.join(skill_data.keywords)}"
        embedding = await self._embeddings.embed_text(embedding_text)
        
        # Create skill object
        skill = Skill(
            id=str(uuid4()),
            name=skill_data.name,
            metadata=SkillMetadata(
                description=skill_data.description,
                goal=skill_data.goal,
                keywords=skill_data.keywords,
                embedding=embedding,
            ),
            steps=[
                SkillStep(order=i, instruction=step)
                for i, step in enumerate(skill_data.steps)
            ],
            version=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        if auto_save:
            await self.storage.save(skill)
        
        return skill
    
    # ─────────────────────────────────────────────────────────────
    # Skill Refinement
    # ─────────────────────────────────────────────────────────────
    
    async def diagnose(
        self,
        skill: Skill,
        task: Task,
        result: ExecutionResult,
        user_feedback: str,
    ) -> RefinementAction:
        """Diagnose why a skill execution didn't meet expectations.
        
        Args:
            skill: The skill that was executed.
            task: The task it was executed for.
            result: The execution result.
            user_feedback: Feedback from the user about what went wrong.
        
        Returns:
            RefinementAction describing the problem and suggested fix.
        """
        diagnosis = await self._llm.diagnose_failure(
            skill=skill,
            task=task,
            result=result,
            user_feedback=user_feedback,
        )
        
        # Map root cause to refinement type
        type_map = {
            "wrong_steps": RefinementType.EDIT_SKILL,
            "wrong_selection": RefinementType.EDIT_MATCHING,
            "wrong_expectations": RefinementType.FORK_SKILL,
        }
        
        return RefinementAction(
            type=type_map[diagnosis.root_cause],
            skill_id=skill.id,
            diagnosis=diagnosis.diagnosis,
            suggested_changes=diagnosis.suggested_changes,
        )
    
    async def refine(
        self,
        skill: Skill,
        action: RefinementAction,
        auto_save: bool = True,
    ) -> Skill:
        """Apply refinement to a skill based on diagnosis.
        
        Args:
            skill: The skill to refine.
            action: The refinement action from diagnose().
            auto_save: Whether to save the refined skill.
        
        Returns:
            The refined (or forked) skill.
        """
        # Get refined skill data from LLM
        refined_data = await self._llm.refine_skill(skill, action)
        
        # Generate new embedding
        embedding_text = f"{refined_data.description}\n{refined_data.goal}\n{' '.join(refined_data.keywords)}"
        embedding = await self._embeddings.embed_text(embedding_text)
        
        if action.type == RefinementType.FORK_SKILL:
            # Create a new skill
            refined_skill = Skill(
                id=str(uuid4()),
                name=refined_data.name,
                metadata=SkillMetadata(
                    description=refined_data.description,
                    goal=refined_data.goal,
                    keywords=refined_data.keywords,
                    embedding=embedding,
                ),
                steps=[
                    SkillStep(order=i, instruction=step)
                    for i, step in enumerate(refined_data.steps)
                ],
                version=1,
                parent_id=skill.id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        else:
            # Update existing skill
            refined_skill = Skill(
                id=skill.id,
                name=refined_data.name,
                metadata=SkillMetadata(
                    description=refined_data.description,
                    goal=refined_data.goal,
                    keywords=refined_data.keywords,
                    embedding=embedding,
                ),
                steps=[
                    SkillStep(order=i, instruction=step)
                    for i, step in enumerate(refined_data.steps)
                ],
                version=skill.version + 1,
                parent_id=skill.parent_id,
                created_at=skill.created_at,
                updated_at=datetime.now(),
            )
        
        if auto_save:
            await self.storage.save(refined_skill)
        
        return refined_skill
    
    # ─────────────────────────────────────────────────────────────
    # Skill Optimization
    # ─────────────────────────────────────────────────────────────
    
    async def find_similar_skills(
        self,
        threshold: float = 0.90,
    ) -> list[tuple[Skill, Skill, float]]:
        """Find pairs of similar skills that could be merged.
        
        Args:
            threshold: Minimum similarity for considering a merge.
        
        Returns:
            List of (skill_a, skill_b, similarity) tuples.
        """
        all_skills = await self.storage.get_all()
        if len(all_skills) < 2:
            return []
        
        pairs = []
        for i, skill_a in enumerate(all_skills):
            for skill_b in all_skills[i + 1:]:
                if skill_a.metadata.embedding and skill_b.metadata.embedding:
                    similarity = cosine_similarity(
                        skill_a.metadata.embedding,
                        skill_b.metadata.embedding,
                    )
                    if similarity >= threshold:
                        pairs.append((skill_a, skill_b, similarity))
        
        return sorted(pairs, key=lambda x: x[2], reverse=True)
    
    async def merge(
        self,
        skill_ids: list[str],
        delete_originals: bool = False,
    ) -> Skill:
        """Merge multiple skills into one.
        
        Args:
            skill_ids: IDs of skills to merge.
            delete_originals: Whether to delete the original skills after merge.
        
        Returns:
            The merged skill.
        """
        if len(skill_ids) < 2:
            raise ValueError("At least 2 skills required for merging")
        
        skills = []
        for skill_id in skill_ids:
            skill = await self.storage.get(skill_id)
            if skill:
                skills.append(skill)
        
        if len(skills) < 2:
            raise ValueError("Could not find enough skills to merge")
        
        # Get merged skill data from LLM
        merged_data = await self._llm.merge_skills(skills)
        
        # Generate embedding
        embedding_text = f"{merged_data.description}\n{merged_data.goal}\n{' '.join(merged_data.keywords)}"
        embedding = await self._embeddings.embed_text(embedding_text)
        
        # Create merged skill
        merged_skill = Skill(
            id=str(uuid4()),
            name=merged_data.name,
            metadata=SkillMetadata(
                description=merged_data.description,
                goal=merged_data.goal,
                keywords=merged_data.keywords,
                embedding=embedding,
            ),
            steps=[
                SkillStep(order=i, instruction=step)
                for i, step in enumerate(merged_data.steps)
            ],
            version=1,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        # Save merged skill
        await self.storage.save(merged_skill)
        
        # Optionally delete originals
        if delete_originals:
            for skill_id in skill_ids:
                await self.storage.delete(skill_id)
        
        return merged_skill
    
    async def optimize(
        self,
        similarity_threshold: float = 0.90,
        dry_run: bool = True,
    ) -> list[tuple[list[Skill], Skill | None]]:
        """Automatically find and merge similar skills.
        
        Args:
            similarity_threshold: Minimum similarity for merging.
            dry_run: If True, only return suggestions without merging.
        
        Returns:
            List of (original_skills, merged_skill) tuples.
            If dry_run=True, merged_skill will be None.
        """
        pairs = await self.find_similar_skills(threshold=similarity_threshold)
        
        results = []
        merged_ids = set()
        
        for skill_a, skill_b, similarity in pairs:
            # Skip if already merged
            if skill_a.id in merged_ids or skill_b.id in merged_ids:
                continue
            
            if dry_run:
                results.append(([skill_a, skill_b], None))
            else:
                merged = await self.merge(
                    [skill_a.id, skill_b.id],
                    delete_originals=True,
                )
                results.append(([skill_a, skill_b], merged))
                merged_ids.add(skill_a.id)
                merged_ids.add(skill_b.id)
        
        return results
