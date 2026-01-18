"""Task data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TaskContext:
    """Contextual information about a task.
    
    Attributes:
        user_id: Optional identifier of the user making the request.
        session_id: Optional session identifier for tracking conversations.
        extra: Additional context data that may be useful for skill execution.
    """
    user_id: str | None = None
    session_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A user task/request to be matched against skills.
    
    Attributes:
        id: Unique identifier.
        query: Original user query/request text.
        key_aspects: Extracted key aspects that define the approach to the task.
        embedding: Computed embedding vector for similarity matching.
        context: Additional context about the task.
        created_at: Creation timestamp.
    """
    id: str
    query: str
    key_aspects: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    context: TaskContext = field(default_factory=TaskContext)
    created_at: datetime = field(default_factory=datetime.now)

    def to_embedding_text(self) -> str:
        """Generate text representation for embedding computation."""
        aspects_str = ", ".join(self.key_aspects) if self.key_aspects else ""
        return f"{self.query}\n\nKey aspects: {aspects_str}"
