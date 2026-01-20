"""SkillDialogueAgent - Conversational agent that always operates through skills.

This agent:
1. Receives user message
2. Matches to existing skill (or generates new one)
3. Executes skill steps with tool support
4. Returns response and maintains conversation state
"""

import asyncio
import inspect
import json
from typing import Any
from uuid import uuid4
from datetime import datetime

from raven_skills.models.skill import Skill
from raven_skills.models.task import Task, TaskContext
from raven_skills.models.dialogue import (
    Tool,
    ToolCall,
    DialogueResponse,
    ConversationState,
)
from raven_skills.interfaces.storage import SkillStorage
from raven_skills.core.llm import LLMClient
from raven_skills.core.embeddings import EmbeddingsClient


class SkillDialogueAgent:
    """Conversational agent that always operates through skills.
    
    This agent matches user queries to skills, executes them using available
    tools, and maintains conversation context across multiple turns.
    
    Example:
        ```python
        agent = SkillDialogueAgent(
            client=openai_client,
            storage=skill_storage,
            tools=[search_tool, calculator_tool],
        )
        
        response = await agent.chat("Deploy my app to Kubernetes")
        print(response.message)  # Skill-based response
        
        response = await agent.chat("Now add monitoring")
        # Continues conversation, uses different skill
        ```
    """
    
    def __init__(
        self,
        client: Any,
        storage: SkillStorage,
        tools: list[Tool] | None = None,
        embedding_client: Any | None = None,
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        similarity_threshold: float = 0.55,
        auto_generate_skills: bool = True,
        use_structured_outputs: bool = True,
        logger: Any | None = None,
    ):
        """Initialize the dialogue agent.
        
        Args:
            client: OpenAI-compatible client for LLM calls.
            storage: Storage for skills.
            tools: List of tools available for execution.
            embedding_client: Optional separate client for embeddings.
            llm_model: Model to use for LLM calls.
            embedding_model: Model to use for embeddings.
            similarity_threshold: Minimum score for skill matching.
            auto_generate_skills: Generate new skill if no match found.
            use_structured_outputs: Use structured outputs API.
            logger: Optional logger for debug output. If None, no logging.
        """
        self.storage = storage
        self.tools = tools or []
        self.similarity_threshold = similarity_threshold
        self.auto_generate_skills = auto_generate_skills
        self._logger = logger
        
        # Internal clients
        self._client = client
        self._llm = LLMClient(
            client,
            model=llm_model,
            use_structured_outputs=use_structured_outputs,
        )
        self._embeddings = EmbeddingsClient(
            embedding_client or client,
            model=embedding_model,
        )
        
        # Conversation state
        self._conversation = ConversationState()
    
    def _log(self, message: str, level: str = "debug") -> None:
        """Log message if logger is configured."""
        if self._logger is None:
            return
        
        log_func = getattr(self._logger, level, self._logger.debug)
        log_func(message)
        
        # Pending skill execution (for multi-turn clarifications)
        self._pending_skill: Skill | None = None
        self._pending_step_index: int = 0
        self._pending_task: Task | None = None
        self._collected_inputs: dict[str, Any] = {}
        
        # Execution tracking (for feedback)
        self._last_skill: Skill | None = None
        self._last_task: Task | None = None
        self._last_step_outputs: list[str] = []
        self._last_tools_called: list = []
        
        # Tool lookup
        self._tool_map = {tool.name: tool for tool in self.tools}
    
    @property
    def conversation_id(self) -> str:
        """Current conversation ID."""
        return self._conversation.id
    
    @property
    def conversation_history(self) -> list[dict]:
        """Get conversation history in OpenAI format."""
        return self._conversation.to_openai_messages()
    
    @property
    def is_awaiting_input(self) -> bool:
        """True if agent is waiting for user clarification."""
        return self._pending_skill is not None
    
    def reset(self) -> None:
        """Reset conversation state for new dialogue."""
        self._conversation = ConversationState()
        self._pending_skill = None
        self._pending_step_index = 0
        self._pending_task = None
        self._collected_inputs = {}
        # Clear execution tracking
        self._last_skill = None
        self._last_task = None
        self._last_step_outputs = []
        self._last_tools_called = []
    
    async def chat(
        self,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> DialogueResponse:
        """Process a user message and return a skill-based response.
        
        Supports multi-turn clarifications: if a skill step asks for user input,
        the agent returns with needs_user_input=True and waits for next message.
        
        Args:
            message: User's message.
            context: Optional additional context.
        
        Returns:
            DialogueResponse with message, skill used, and tools called.
        """
        # Add user message to conversation
        self._conversation.add_message("user", message)
        
        # Update context
        if context:
            self._conversation.context.update(context)
        
        # If we have a pending skill awaiting user input, continue execution
        if self._pending_skill:
            return await self._continue_skill_execution(message)
        
        # Check if this is negative feedback on last execution
        if self._last_skill and await self._detect_negative_feedback(message):
            return await self._retry_with_feedback(message)
        
        # Prepare task for matching
        task = await self._prepare_task(message)
        
        # Try to match a skill
        skill, score = await self._match_skill(task)
        skill_generated = False
        
        if skill is None and self.auto_generate_skills:
            # Generate a new skill from the request
            skill = await self._generate_skill_from_request(task, message)
            skill_generated = True
        
        if skill is None:
            # Fallback: respond directly without skill
            response_text = await self._respond_without_skill(message)
            self._conversation.add_message("assistant", response_text)
            return DialogueResponse(
                message=response_text,
                skill_used=None,
                conversation_id=self._conversation.id,
            )
        
        # Execute skill step-by-step with possible clarifications
        return await self._execute_skill_stepwise(skill, task, skill_generated)
    
    async def _continue_skill_execution(self, user_input: str) -> DialogueResponse:
        """Continue executing a pending skill after user provided input."""
        # Store user's answer in collected inputs
        current_step_idx = self._pending_step_index - 1  # Previous step asked
        step_key = f"step_{current_step_idx}_input"
        self._collected_inputs[step_key] = user_input
        
        # Resume execution from current step
        return await self._execute_skill_stepwise(
            self._pending_skill,
            self._pending_task,
            skill_generated=False,
        )
    
    async def _execute_skill_stepwise(
        self,
        skill: Skill,
        task: Task,
        skill_generated: bool,
    ) -> DialogueResponse:
        """Execute skill steps one by one, pausing for clarifications if needed."""
        tools_called: list[ToolCall] = []
        context = {
            "task_query": task.query,
            "task_aspects": task.key_aspects,
            "skill_goal": skill.metadata.goal,
            "conversation_history": self._conversation.to_openai_messages()[-5:],
            **self._collected_inputs,
        }
        
        openai_tools = [tool.to_openai_format() for tool in self.tools]
        sorted_steps = sorted(skill.steps, key=lambda s: s.order)
        
        # Determine entry point - which step to start from based on context
        if self._pending_step_index == 0:
            # Only analyze entry point on fresh execution (not continuation)
            start_step, satisfied_data = await self._determine_entry_point(
                skill, sorted_steps, context
            )
            # Store data from satisfied steps
            context.update(satisfied_data)
        else:
            start_step = self._pending_step_index
        
        # Execute steps starting from entry point
        for i in range(start_step, len(sorted_steps)):
            step = sorted_steps[i]
            
            # Check if step requires clarification (contains question markers)
            needs_clarification = self._step_needs_clarification(step.instruction)
            
            if needs_clarification and f"step_{i}_input" not in self._collected_inputs:
                # First, check if info is already in context (user's query or conversation)
                extracted_info = await self._extract_info_from_context(step.instruction, context)
                
                if extracted_info:
                    # Info found in context - use it, don't ask
                    self._collected_inputs[f"step_{i}_input"] = extracted_info
                    self._log(f"[SKILL] Step {i+1}: Found info in context: '{extracted_info[:50]}...'")
                else:
                    # Info not found - ask for clarification
                    clarification_q = await self._generate_clarification(step.instruction, context)
                    
                    # Save state for continuation
                    self._pending_skill = skill
                    self._pending_step_index = i + 1
                    self._pending_task = task
                    
                    self._conversation.add_message("assistant", clarification_q, skill_id=skill.id)
                    
                    return DialogueResponse(
                        message=clarification_q,
                        skill_used=skill,
                        skill_generated=skill_generated,
                        needs_user_input=True,
                        conversation_id=self._conversation.id,
                    )
            
            # Execute the step
            step_result, step_tools = await self._execute_step_with_tools(
                step.instruction, context, openai_tools
            )
            tools_called.extend(step_tools)
            context[f"step_{step.order}_result"] = step_result
        
        # All steps completed - synthesize final response
        self._pending_skill = None
        self._pending_step_index = 0
        self._pending_task = None
        
        final_response = await self._synthesize_response(
            skill, task, list(context.values()), tools_called
        )
        
        self._conversation.add_message(
            "assistant", final_response, tool_calls=tools_called, skill_id=skill.id
        )
        self._conversation.current_skill = skill
        self._collected_inputs = {}
        
        # Save execution state for potential feedback
        self._last_skill = skill
        self._last_task = task
        self._last_step_outputs = [context.get(f"step_{s.order}_result", "") for s in sorted_steps]
        self._last_tools_called = tools_called
        
        return DialogueResponse(
            message=final_response,
            skill_used=skill,
            skill_generated=skill_generated,
            tools_called=tools_called,
            conversation_id=self._conversation.id,
            can_retry=True,  # Allow retry with feedback
        )
    
    def _step_needs_clarification(self, instruction: str) -> bool:
        """Check if step instruction indicates need for user input."""
        clarification_markers = [
            # English markers
            "ask", "clarify", "request", "confirm", "inquire",
            "which", "what", "when", "where", "how many",
            "do you want", "would you like", "do you need",
            # Russian markers (for backward compatibility)
            "уточни", "спроси", "узнай", "запроси",
            "какой", "какая", "какое", "на какую",
            "нужно ли", "требуется ли", "хотите ли",
        ]
        instruction_lower = instruction.lower()
        return any(marker in instruction_lower for marker in clarification_markers)
    
    async def _determine_entry_point(
        self,
        skill: Skill,
        steps: list,
        context: dict,
    ) -> tuple[int, dict]:
        """Determine which step to start from based on context.
        
        Returns:
            (start_step_index, satisfied_data): Step index to start from and
            data extracted from context for satisfied steps.
        """
        steps_text = "\n".join(
            f"Step {i+1}: {s.instruction}" 
            for i, s in enumerate(steps)
        )
        
        task_query = context.get("task_query", "")
        conversation = context.get("conversation_history", [])
        conversation_text = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')}" 
            for m in conversation[-5:]
        )
        
        response = await self._client.chat.completions.create(
            model=self._llm.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyze skill steps and conversation context to determine "
                        "which steps are already satisfied.\n\n"
                        "For each step, determine if the required information/action is "
                        "already present in the user's query or conversation.\n\n"
                        "Respond in this exact format:\n"
                        "STEP 1: SATISFIED|value extracted\n"
                        "STEP 2: PENDING\n"
                        "STEP 3: PENDING\n"
                        "START: 2\n\n"
                        "The START line indicates which step to begin execution from (1-indexed)."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Skill: {skill.name}\n"
                        f"Goal: {skill.metadata.goal}\n\n"
                        f"Steps:\n{steps_text}\n\n"
                        f"User query: {task_query}\n\n"
                        f"Conversation:\n{conversation_text}\n\n"
                        "Analyze which steps are already satisfied:"
                    )
                }
            ],
            temperature=0,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Parse response
        satisfied_data = {}
        start_step = 0  # Default: start from beginning
        
        for line in answer.split("\n"):
            line = line.strip()
            
            # Parse START line
            if line.upper().startswith("START:"):
                try:
                    start_step = int(line.split(":")[1].strip()) - 1  # Convert to 0-indexed
                    start_step = max(0, min(start_step, len(steps) - 1))
                except:
                    pass
            
            # Parse STEP lines with SATISFIED
            elif line.upper().startswith("STEP") and "SATISFIED" in line.upper():
                try:
                    # Extract step number and value
                    parts = line.split(":", 1)
                    step_num = int(parts[0].replace("STEP", "").strip()) - 1
                    if "|" in parts[1]:
                        value = parts[1].split("|", 1)[1].strip()
                        satisfied_data[f"step_{step_num}_result"] = value
                except:
                    pass
        
        # Log entry point decision
        self._log(f"[SKILL] Entry point analysis for '{skill.name}':")
        for i, step in enumerate(steps):
            status = "SATISFIED" if f"step_{i}_result" in satisfied_data else "PENDING"
            marker = "✓" if status == "SATISFIED" else "○"
            self._log(f"[SKILL]   {marker} Step {i+1}: {status}")
        self._log(f"[SKILL] → Starting from step {start_step + 1}")
        
        return start_step, satisfied_data
    
    async def _generate_clarification(self, instruction: str, context: dict) -> str:
        """Generate a clarification question from step instruction."""
        response = await self._client.chat.completions.create(
            model=self._llm.model,
            messages=[
                {"role": "system", "content": "Generate a short, natural clarification question based on the instruction. Respond only with the question, nothing else."},
                {"role": "user", "content": f"Instruction: {instruction}\nContext: {json.dumps(context, ensure_ascii=False, default=str)[:500]}"},
            ],
        )
        return response.choices[0].message.content
    
    async def _extract_info_from_context(self, instruction: str, context: dict) -> str | None:
        """Try to extract required info from existing context.
        
        Returns extracted info if found, None if user needs to be asked.
        """
        # Include full conversation history for context
        conversation_text = ""
        history = context.get("conversation_history", [])
        if history:
            conversation_text = "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in history[-10:])
        
        task_query = context.get("task_query", "")
        
        response = await self._client.chat.completions.create(
            model=self._llm.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an information extractor. Given a step instruction and conversation context, "
                        "determine if the required information is already provided.\n\n"
                        "If the info IS in the context, respond with: FOUND: <the extracted info>\n"
                        "If the info is NOT in the context, respond with: NOT_FOUND"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Step instruction: {instruction}\n\n"
                        f"User's original query: {task_query}\n\n"
                        f"Conversation:\n{conversation_text}\n\n"
                        "Is the required information already in the context?"
                    )
                }
            ],
            max_tokens=100,
            temperature=0,
        )
        
        answer = response.choices[0].message.content.strip()
        
        if answer.startswith("FOUND:"):
            return answer[6:].strip()
        return None
    
    # ─────────────────────────────────────────────────────────────
    # Feedback Mechanism
    # ─────────────────────────────────────────────────────────────
    
    async def _detect_negative_feedback(self, message: str) -> bool:
        """Detect if message is negative feedback on last execution."""
        negative_patterns = [
            # English
            "no", "wrong", "incorrect", "not right", "mistake", "error",
            "missing", "forgot", "didn't", "failed", "broken", "bad",
            "that's not", "you missed", "you forgot", "try again",
            # Russian
            "нет", "неправильно", "не так", "ошибка", "пропустил",
            "забыл", "не то", "плохо", "неверно",
        ]
        message_lower = message.lower()
        return any(pattern in message_lower for pattern in negative_patterns)
    
    async def _retry_with_feedback(self, correction: str) -> DialogueResponse:
        """Retry last execution with user's correction."""
        from raven_skills.models.result import ExecutionResult
        
        # Create a mock execution result for diagnosis
        last_output = "\n".join(str(s) for s in self._last_step_outputs) if self._last_step_outputs else "No output"
        mock_result = ExecutionResult(
            success=False,  # It failed, that's why we're retrying
            output=last_output,
        )
        
        # Diagnose what went wrong
        try:
            diagnosis = await self._llm.diagnose_failure(
                self._last_skill,
                self._last_task,
                mock_result,
                correction,
            )
            
            # Map root_cause to RefinementType
            from raven_skills.models.result import RefinementAction, RefinementType
            
            root_cause_map = {
                "wrong_steps": RefinementType.EDIT_SKILL,
                "wrong_selection": RefinementType.EDIT_MATCHING,
                "wrong_expectations": RefinementType.FORK_SKILL,
            }
            refinement_type = root_cause_map.get(diagnosis.root_cause, RefinementType.EDIT_SKILL)
            
            action = RefinementAction(
                type=refinement_type,
                skill_id=self._last_skill.id,
                diagnosis=diagnosis.diagnosis,
                suggested_changes=diagnosis.suggested_changes,
            )
            
            # Refine the skill
            refined = await self._llm.refine_skill(
                self._last_skill,
                action,
            )
            
            # Build refined skill - UPDATE existing skill (same ID, increment version)
            from raven_skills.models.skill import SkillMetadata, SkillStep
            
            # Re-generate embedding for refined skill (better matching in future)
            new_embedding_text = f"{refined.name}\n{refined.description}\n{refined.goal}"
            new_embedding = await self._embeddings.embed_text(new_embedding_text)
            
            refined_skill = Skill(
                id=self._last_skill.id,  # SAME ID - update in place
                name=refined.name,
                version=self._last_skill.version + 1,
                parent_id=self._last_skill.parent_id,  # Keep original parent
                metadata=SkillMetadata(
                    description=refined.description,
                    goal=refined.goal,
                    keywords=refined.keywords,
                    embedding=new_embedding,  # Updated embedding
                ),
                steps=[
                    SkillStep(order=i + 1, instruction=step)
                    for i, step in enumerate(refined.steps)
                ],
            )
            
            # Save updated skill immediately (replaces old version in storage)
            await self.storage.save(refined_skill)
            self._log(f"[FEEDBACK] Updated skill '{refined_skill.name}' to v{refined_skill.version}")
            
            # Execute refined skill
            result = await self._execute_skill_stepwise(
                refined_skill, self._last_task, skill_generated=False
            )
            
            # Mark as refined
            result.skill_refined = True
            result.message += "\n\n✅ Skill updated and saved."
            
            # Clear pending (already saved)
            self._refined_skill_pending = None
            
            return result
            
        except Exception as e:
            # Log the error for debugging
            import traceback
            self._log(f"[FEEDBACK DEBUG] Refinement failed: {e}")
            traceback.print_exc()
            
            # If refinement fails, just acknowledge and retry with original approach
            self._conversation.add_message(
                "assistant", 
                f"I understand. Let me try again with your feedback: {correction}"
            )
            return await self._execute_skill_stepwise(
                self._last_skill, self._last_task, skill_generated=False
            )
    
    async def feedback(
        self,
        rating: str,  # "positive" or "negative"
        correction: str | None = None,
    ) -> DialogueResponse:
        """Provide explicit feedback on last execution.
        
        Args:
            rating: "positive" or "negative"
            correction: What was wrong (required for negative feedback)
        
        Returns:
            DialogueResponse with retry result or acknowledgment
        
        Example:
            # When execution was wrong
            await agent.feedback("negative", "You missed the second sheet")
            
            # When execution was correct
            await agent.feedback("positive")
        """
        if not self._last_skill:
            return DialogueResponse(
                message="No recent execution to provide feedback on.",
                conversation_id=self._conversation.id,
            )
        
        if rating == "negative" and correction:
            self._conversation.add_message("user", f"Feedback: {correction}")
            return await self._retry_with_feedback(correction)
        elif rating == "positive":
            # Could track success metrics here
            return DialogueResponse(
                message="Thanks for the feedback! Skill execution was successful.",
                skill_used=self._last_skill,
                conversation_id=self._conversation.id,
            )
        else:
            return DialogueResponse(
                message="Please provide a correction with negative feedback.",
                conversation_id=self._conversation.id,
            )
    
    async def save_refined_skill(self) -> bool:
        """Save the pending refined skill after user confirmation.
        
        Returns:
            True if skill was saved, False if nothing to save.
        """
        if hasattr(self, '_refined_skill_pending') and self._refined_skill_pending:
            await self.storage.save(self._refined_skill_pending)
            self._refined_skill_pending = None
            return True
        return False
    
    async def _prepare_task(self, query: str) -> Task:
        """Prepare a task for skill matching with HyDE transformation."""
        aspects = await self._llm.extract_key_aspects(query)
        
        # HyDE: Generate hypothetical skill description for the query
        # This creates a "skill-like" text that matches better with actual skills
        hypothetical = await self._generate_hypothetical_skill(query, aspects.key_aspects)
        
        # Embed the hypothetical skill description (HyDE)
        embedding = await self._embeddings.embed_text(hypothetical)
        
        return Task(
            id=str(uuid4()),
            query=query,
            key_aspects=aspects.key_aspects,
            embedding=embedding,
            context=TaskContext(**self._conversation.context),
            created_at=datetime.now(),
        )
    
    async def _generate_hypothetical_skill(self, query: str, aspects: list[str]) -> str:
        """Generate a hypothetical skill description for HyDE matching.
        
        Instead of embedding the query directly, we generate what an ideal
        matching skill would look like, then embed that. This improves
        semantic matching with actual skills.
        """
        response = await self._client.chat.completions.create(
            model=self._llm.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a brief skill description that would solve the user's query. "
                        "Format:\n"
                        "Name: [skill name]\n"
                        "Description: [what the skill does]\n"
                        "Goal: [expected outcome]\n\n"
                        "Keep it concise (2-3 sentences max)."
                    )
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\nKey aspects: {', '.join(aspects)}"
                }
            ],
            max_tokens=150,
            temperature=0,
        )
        
        hypothetical = response.choices[0].message.content.strip()
        self._log(f"[HyDE] Generated hypothetical skill:\n{hypothetical[:100]}...")
        return hypothetical
    
    async def _match_skill(self, task: Task) -> tuple[Skill | None, float]:
        """Find best matching skill for task."""
        from raven_skills.utils.similarity import cosine_similarity
        
        all_skills = await self.storage.get_all()
        if not all_skills:
            self._log(f"[MATCH] No skills in storage")
            return None, 0.0
        
        best_skill = None
        best_score = 0.0
        candidates = []
        
        for skill in all_skills:
            # Multi-vector matching: take max score across all skill embeddings
            skill_scores = []
            
            # Check multi-vector embeddings first
            if skill.metadata.embeddings:
                for emb in skill.metadata.embeddings:
                    if emb:
                        skill_scores.append(cosine_similarity(task.embedding, emb))
            
            # Fallback to primary embedding
            if not skill_scores and skill.metadata.embedding:
                skill_scores.append(cosine_similarity(task.embedding, skill.metadata.embedding))
            
            if skill_scores:
                # Max pooling: best match across all representations
                score = max(skill_scores)
                candidates.append((skill, score))
                if score > best_score:
                    best_score = score
                    best_skill = skill
        
        # Debug logging
        self._log(f"[MATCH] Query: '{task.query[:50]}...' | Threshold: {self.similarity_threshold}")
        for skill, score in sorted(candidates, key=lambda x: -x[1])[:5]:
            marker = "✓" if score >= self.similarity_threshold else ("?" if score >= 0.35 else "✗")
            self._log(f"[MATCH]   {marker} {skill.name}: {score:.3f}")
        
        # Direct match - high confidence
        if best_score >= self.similarity_threshold:
            self._log(f"[MATCH] → Selected: '{best_skill.name}' ({best_score:.3f})")
            return best_skill, best_score
        
        # Gray zone (0.35 - threshold) - ask LLM to verify
        gray_zone_min = 0.35
        if best_score >= gray_zone_min and best_skill:
            self._log(f"[MATCH] Gray zone ({best_score:.3f}) - asking LLM to verify...")
            is_match = await self._llm_verify_match(task, best_skill)
            if is_match:
                self._log(f"[MATCH] → LLM confirmed: '{best_skill.name}'")
                return best_skill, best_score
            else:
                self._log(f"[MATCH] → LLM rejected match")
        
        self._log(f"[MATCH] → No match (best: {best_score:.3f})")
        return None, best_score
    
    async def _llm_verify_match(self, task: Task, skill: Skill) -> bool:
        """Ask LLM to verify if skill matches the task."""
        steps_text = "\n".join(f"- {s.instruction}" for s in skill.steps[:5])
        
        response = await self._client.chat.completions.create(
            model=self._llm.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a skill matching validator. "
                        "Determine if the given skill can solve the user's task. "
                        "Respond with only 'YES' or 'NO'."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"User task: {task.query}\n\n"
                        f"Skill: {skill.name}\n"
                        f"Description: {skill.metadata.description}\n"
                        f"Goal: {skill.metadata.goal}\n"
                        f"Steps:\n{steps_text}\n\n"
                        "Can this skill solve the user's task? YES or NO:"
                    )
                }
            ],
            max_tokens=10,
            temperature=0,
        )
        
        answer = response.choices[0].message.content.strip().upper()
        self._log(f"[MATCH] LLM verification: '{answer}'")
        return "YES" in answer
    
    async def _generate_skill_from_request(
        self, task: Task, message: str
    ) -> Skill | None:
        """Generate a new skill from user request."""
        # Create a synthetic conversation for skill generation
        conversation = [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"I'll help you with: {message}"},
        ]
        
        try:
            skill_data = await self._llm.generate_skill_from_conversation(
                task=task,
                conversation=conversation,
                final_result=f"Task completed: {message}",
            )
            
            # Build skill from schema response
            from raven_skills.models.skill import SkillMetadata, SkillStep
            
            # Generate multi-vector embeddings for improved matching
            embedding_texts = [
                skill_data.name,                           # Embed name separately
                skill_data.description,                    # Embed description  
                skill_data.goal,                           # Embed goal
                " ".join(skill_data.keywords),             # Embed keywords
            ]
            multi_embeddings = await self._embeddings.embed_texts(embedding_texts)
            
            # Primary embedding (combined) for backward compatibility
            primary_text = f"{skill_data.name}\n{skill_data.description}\n{skill_data.goal}"
            primary_embedding = await self._embeddings.embed_text(primary_text)
            
            metadata = SkillMetadata(
                description=skill_data.description,
                goal=skill_data.goal,
                keywords=skill_data.keywords,
                embedding=primary_embedding,      # Primary (legacy)
                embeddings=multi_embeddings,      # Multi-vector
            )
            
            steps = [
                SkillStep(order=i + 1, instruction=step_text)
                for i, step_text in enumerate(skill_data.steps)
            ]
            
            skill = Skill(
                id=str(uuid4()),
                name=skill_data.name,
                version=1,
                metadata=metadata,
                steps=steps,
                created_at=datetime.now(),
            )
            
            await self.storage.save(skill)
            self._log(f"[SKILL] Created new skill: '{skill.name}' ({len(skill.steps)} steps)")
            return skill
        except Exception as e:
            import traceback
            self._log(f"⚠️ Skill generation failed: {e}")
            traceback.print_exc()
            return None
    
    async def _execute_skill_with_tools(
        self, skill: Skill, task: Task
    ) -> tuple[str, list[ToolCall]]:
        """Execute skill steps, calling tools as needed."""
        tools_called: list[ToolCall] = []
        results: list[str] = []
        context = {
            "task_query": task.query,
            "task_aspects": task.key_aspects,
            "skill_goal": skill.metadata.goal,
            "conversation_history": self._conversation.to_openai_messages()[-5:],
        }
        
        # Prepare tools for OpenAI format
        openai_tools = [tool.to_openai_format() for tool in self.tools]
        
        for step in sorted(skill.steps, key=lambda s: s.order):
            # Execute step with possible tool calls
            step_result, step_tools = await self._execute_step_with_tools(
                step.instruction, context, openai_tools
            )
            results.append(step_result)
            tools_called.extend(step_tools)
            context[f"step_{step.order}_result"] = step_result
        
        # Synthesize final response
        final_response = await self._synthesize_response(
            skill, task, results, tools_called
        )
        
        return final_response, tools_called
    
    async def _execute_step_with_tools(
        self,
        instruction: str,
        context: dict,
        openai_tools: list[dict],
    ) -> tuple[str, list[ToolCall]]:
        """Execute a single step, handling tool calls."""
        tools_called: list[ToolCall] = []
        
        messages = [
            {
                "role": "system",
                "content": f"Execute this step: {instruction}\nContext: {json.dumps(context, ensure_ascii=False, default=str)}"
            },
            {"role": "user", "content": "Execute the step and use tools if needed."},
        ]
        
        if openai_tools:
            response = await self._client.chat.completions.create(
                model=self._llm.model,
                messages=messages,
                tools=openai_tools,
            )
        else:
            response = await self._client.chat.completions.create(
                model=self._llm.model,
                messages=messages,
            )
        
        message = response.choices[0].message
        
        # Handle tool calls
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                
                tool_result = await self._call_tool(tool_name, arguments)
                tools_called.append(ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=tool_result,
                ))
        
        content = message.content or ""
        if tools_called:
            content += f"\n[Tools used: {', '.join(t.tool_name for t in tools_called)}]"
        
        return content, tools_called
    
    async def _call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool by name with arguments."""
        tool = self._tool_map.get(name)
        if not tool:
            return f"Error: Tool '{name}' not found"
        
        try:
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**arguments)
            else:
                result = tool.function(**arguments)
            return str(result)
        except Exception as e:
            return f"Error calling {name}: {str(e)}"
    
    async def _synthesize_response(
        self,
        skill: Skill,
        task: Task,
        step_results: list[str],
        tools_called: list[ToolCall],
    ) -> str:
        """Synthesize a natural response from skill execution results."""
        prompt = f"""Based on executing skill "{skill.name}" for the query "{task.query}":

Step results:
{chr(10).join(f"- {r}" for r in step_results)}

Tools called:
{chr(10).join(f"- {t.tool_name}: {t.result[:100]}" for t in tools_called) if tools_called else "None"}

Provide a helpful, conversational response summarizing what was done."""

        response = await self._client.chat.completions.create(
            model=self._llm.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide concise, natural responses."},
                {"role": "user", "content": prompt},
            ],
        )
        
        return response.choices[0].message.content
    
    async def _respond_without_skill(self, message: str) -> str:
        """Direct response when no skill matches and generation is disabled."""
        response = await self._client.chat.completions.create(
            model=self._llm.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. The user's request doesn't match any existing skills."},
                {"role": "user", "content": message},
            ],
        )
        return response.choices[0].message.content
