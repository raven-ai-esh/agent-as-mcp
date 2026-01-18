"""Dialogue models for skill-based conversational agent."""

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from datetime import datetime
from uuid import uuid4

from raven_skills.models.skill import Skill


@dataclass
class Tool:
    """A tool that can be called during skill execution.
    
    Compatible with OpenAI function calling format.
    """
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema format
    function: Callable[..., str | Awaitable[str]]  # Sync or async
    
    def to_openai_format(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class ToolCall:
    """Record of a tool call during execution."""
    tool_name: str
    arguments: dict[str, Any]
    result: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DialogueMessage:
    """A message in the conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    tool_calls: list[ToolCall] = field(default_factory=list)
    skill_id: str | None = None  # If this message was generated using a skill


@dataclass
class DialogueResponse:
    """Response from the dialogue agent."""
    message: str
    skill_used: Skill | None = None
    skill_generated: bool = False
    needs_user_input: bool = False  # True when waiting for clarification
    tools_called: list[ToolCall] = field(default_factory=list)
    conversation_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ConversationState:
    """State of the ongoing conversation."""
    id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[DialogueMessage] = field(default_factory=list)
    current_skill: Skill | None = None
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_message(
        self, 
        role: str, 
        content: str, 
        tool_calls: list[ToolCall] | None = None,
        skill_id: str | None = None,
    ) -> DialogueMessage:
        """Add a message to the conversation."""
        msg = DialogueMessage(
            role=role,
            content=content,
            tool_calls=tool_calls or [],
            skill_id=skill_id,
        )
        self.messages.append(msg)
        return msg
    
    def to_openai_messages(self) -> list[dict]:
        """Convert to OpenAI messages format."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
