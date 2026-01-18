"""Abstract skill storage interface."""

from abc import ABC, abstractmethod

from raven_skills.models.skill import Skill


class SkillStorage(ABC):
    """Abstract interface for skill persistence.
    
    Implementations should provide methods to store, retrieve, and search
    skills. The storage can be backed by various systems:
    
    - Vector databases (Pinecone, Weaviate, Qdrant, Milvus)
    - Traditional databases with vector extensions (PostgreSQL + pgvector)
    - In-memory storage for testing
    - File-based storage
    """

    @abstractmethod
    async def save(self, skill: Skill) -> None:
        """Save or update a skill in storage.
        
        If a skill with the same ID exists, it should be updated.
        
        Args:
            skill: The skill to save.
        """
        ...

    @abstractmethod
    async def get(self, skill_id: str) -> Skill | None:
        """Retrieve a skill by its ID.
        
        Args:
            skill_id: The unique identifier of the skill.
            
        Returns:
            The skill if found, None otherwise.
        """
        ...

    @abstractmethod
    async def get_all(self) -> list[Skill]:
        """Retrieve all skills from storage.
        
        Returns:
            List of all stored skills.
        """
        ...

    @abstractmethod
    async def delete(self, skill_id: str) -> None:
        """Delete a skill from storage.
        
        Args:
            skill_id: The ID of the skill to delete.
        """
        ...

    @abstractmethod
    async def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Skill, float]]:
        """Search for skills by embedding similarity.
        
        Args:
            embedding: The query embedding vector.
            top_k: Maximum number of results to return.
            min_score: Minimum similarity score (0.0 to 1.0).
            
        Returns:
            List of (skill, score) tuples, ordered by descending similarity.
        """
        ...

    async def exists(self, skill_id: str) -> bool:
        """Check if a skill exists in storage.
        
        Default implementation uses get(). Implementations may override
        for better performance.
        
        Args:
            skill_id: The ID to check.
            
        Returns:
            True if the skill exists, False otherwise.
        """
        return await self.get(skill_id) is not None

    async def count(self) -> int:
        """Get the total number of skills in storage.
        
        Default implementation uses get_all(). Implementations may override
        for better performance.
        
        Returns:
            The number of skills in storage.
        """
        return len(await self.get_all())
