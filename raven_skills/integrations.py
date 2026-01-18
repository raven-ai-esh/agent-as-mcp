"""LangChain and LangGraph integration adapters.

Provides compatibility layers for using raven-skills agents
within LangChain chains and LangGraph state machines.
"""

import asyncio
from typing import Any, Optional, Type, Callable
from pydantic import BaseModel, Field

from raven_skills.dialogue_agent import SkillDialogueAgent
from raven_skills.agent import SkillAgent
from raven_skills.models.dialogue import DialogueResponse


# ═══════════════════════════════════════════════════════════════════
# LangChain Integration
# ═══════════════════════════════════════════════════════════════════

try:
    from langchain_core.tools import BaseTool
    from langchain_core.callbacks import CallbackManagerForToolRun
    
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    BaseTool = object
    CallbackManagerForToolRun = Any


class SkillAgentInput(BaseModel):
    """Input schema for SkillAgent tool."""
    query: str = Field(description="The user query to match against skills")


class SkillDialogueInput(BaseModel):
    """Input schema for SkillDialogueAgent tool."""
    message: str = Field(description="The user message to process")


class SkillMatcherTool(BaseTool if LANGCHAIN_AVAILABLE else object):
    """LangChain tool wrapper for SkillAgent.match().
    
    Use this to find and execute skills within a LangChain pipeline.
    
    Example:
        ```python
        from langchain.agents import AgentExecutor
        from raven_skills.integrations import SkillMatcherTool
        
        skill_tool = SkillMatcherTool(agent=skill_agent)
        tools = [skill_tool, ...]
        
        agent = AgentExecutor(agent=..., tools=tools)
        ```
    """
    
    name: str = "skill_matcher"
    description: str = "Find and execute a skill matching the user's query. Returns skill name and execution result."
    args_schema: Type[BaseModel] = SkillAgentInput
    
    agent: Any = None  # SkillAgent instance
    
    def __init__(self, agent: SkillAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
    
    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Sync execution."""
        return asyncio.run(self._arun(query, run_manager))
    
    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async execution."""
        task, result = await self.agent.match(query)
        
        if not result.found:
            return f"No skill found for: {query}"
        
        execution = await self.agent.execute(result.skill, task)
        return f"Skill '{result.skill.name}': {execution.output}"


class SkillDialogueTool(BaseTool if LANGCHAIN_AVAILABLE else object):
    """LangChain tool wrapper for SkillDialogueAgent.chat().
    
    Use this for conversational skill-based responses.
    
    Example:
        ```python
        from raven_skills.integrations import SkillDialogueTool
        
        dialogue_tool = SkillDialogueTool(agent=dialogue_agent)
        result = dialogue_tool.invoke({"message": "What's the weather?"})
        ```
    """
    
    name: str = "skill_dialogue"
    description: str = "Have a skill-based conversation. Supports multi-turn dialogues with clarifications."
    args_schema: Type[BaseModel] = SkillDialogueInput
    
    agent: Any = None  # SkillDialogueAgent instance
    
    def __init__(self, agent: SkillDialogueAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
    
    def _run(
        self,
        message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Sync execution."""
        return asyncio.run(self._arun(message, run_manager))
    
    async def _arun(
        self,
        message: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async execution."""
        response = await self.agent.chat(message)
        
        result = response.message
        if response.needs_user_input:
            result += " [AWAITING_USER_INPUT]"
        if response.skill_used:
            result += f" [SKILL: {response.skill_used.name}]"
        
        return result


# ═══════════════════════════════════════════════════════════════════
# LangGraph Integration
# ═══════════════════════════════════════════════════════════════════

try:
    from typing import TypedDict
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    TypedDict = dict


class SkillGraphState(TypedDict):
    """State schema for LangGraph integration.
    
    Use this as the state type in your LangGraph StateGraph.
    """
    messages: list[dict]  # Conversation history
    current_message: str  # Latest user message
    response: str  # Agent response
    skill_name: str | None  # Used skill name
    skill_generated: bool  # Whether skill was newly generated
    needs_user_input: bool  # Waiting for clarification
    tools_called: list[str]  # Tools invoked


def create_skill_node(agent: SkillDialogueAgent) -> Callable:
    """Create a LangGraph node function from SkillDialogueAgent.
    
    Example:
        ```python
        from langgraph.graph import StateGraph
        from raven_skills.integrations import create_skill_node, SkillGraphState
        
        graph = StateGraph(SkillGraphState)
        skill_node = create_skill_node(dialogue_agent)
        
        graph.add_node("skill_agent", skill_node)
        graph.add_edge("__start__", "skill_agent")
        
        # Conditional routing based on needs_user_input
        def route(state):
            if state["needs_user_input"]:
                return "wait_for_input"
            return "__end__"
        
        graph.add_conditional_edges("skill_agent", route)
        ```
    """
    
    async def skill_node(state: SkillGraphState) -> SkillGraphState:
        """Process state through SkillDialogueAgent."""
        message = state.get("current_message", "")
        
        if not message:
            return state
        
        response = await agent.chat(message)
        
        return {
            **state,
            "messages": state.get("messages", []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response.message},
            ],
            "response": response.message,
            "skill_name": response.skill_used.name if response.skill_used else None,
            "skill_generated": response.skill_generated,
            "needs_user_input": response.needs_user_input,
            "tools_called": [t.tool_name for t in response.tools_called],
        }
    
    return skill_node


def create_skill_router(
    needs_input_node: str = "wait_for_input",
    complete_node: str = "__end__",
) -> Callable:
    """Create a router function for LangGraph conditional edges.
    
    Routes to 'needs_input_node' if agent is waiting for clarification,
    otherwise routes to 'complete_node'.
    
    Example:
        ```python
        router = create_skill_router(
            needs_input_node="get_user_input",
            complete_node="process_result",
        )
        graph.add_conditional_edges("skill_agent", router)
        ```
    """
    
    def router(state: SkillGraphState) -> str:
        if state.get("needs_user_input", False):
            return needs_input_node
        return complete_node
    
    return router


# ═══════════════════════════════════════════════════════════════════
# Runnable Interface (LangChain Expression Language)
# ═══════════════════════════════════════════════════════════════════

try:
    from langchain_core.runnables import RunnableSerializable
    RUNNABLE_AVAILABLE = True
except ImportError:
    RUNNABLE_AVAILABLE = False
    RunnableSerializable = object


class SkillAgentRunnable(RunnableSerializable if RUNNABLE_AVAILABLE else object):
    """LangChain Runnable wrapper for SkillDialogueAgent.
    
    Enables use in LCEL (LangChain Expression Language) chains.
    
    Example:
        ```python
        from langchain_core.prompts import ChatPromptTemplate
        from raven_skills.integrations import SkillAgentRunnable
        
        skill_runnable = SkillAgentRunnable(agent=dialogue_agent)
        
        # Use in a chain
        chain = prompt | skill_runnable | output_parser
        result = await chain.ainvoke({"query": "Book a restaurant"})
        ```
    """
    
    agent: Any = None
    
    def __init__(self, agent: SkillDialogueAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
    
    def invoke(self, input: str | dict, config: Optional[dict] = None) -> DialogueResponse:
        """Sync invoke."""
        return asyncio.run(self.ainvoke(input, config))
    
    async def ainvoke(self, input: str | dict, config: Optional[dict] = None) -> DialogueResponse:
        """Async invoke."""
        message = input if isinstance(input, str) else input.get("message", input.get("query", ""))
        return await self.agent.chat(message)


# ═══════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    # LangChain
    "SkillMatcherTool",
    "SkillDialogueTool",
    "SkillAgentRunnable",
    # LangGraph
    "SkillGraphState",
    "create_skill_node",
    "create_skill_router",
    # Availability flags
    "LANGCHAIN_AVAILABLE",
    "LANGGRAPH_AVAILABLE",
]
