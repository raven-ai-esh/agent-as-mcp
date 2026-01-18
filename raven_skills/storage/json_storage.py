"""JSON file-based storage for skills.

Provides persistent storage using a JSON file.
Suitable for development and small deployments.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any

from raven_skills.interfaces.storage import SkillStorage
from raven_skills.models.skill import Skill, SkillMetadata, SkillStep
from raven_skills.utils.similarity import cosine_similarity


class JSONStorage(SkillStorage):
    """JSON file storage for skills.
    
    Stores skills in a JSON file with embeddings for vector search.
    Thread-safe for single-process use.
    
    Example:
        storage = JSONStorage("./skills.json")
        await storage.save(skill)
        skills = await storage.get_all()
    """
    
    def __init__(self, path: str | Path = "./skills.json"):
        """Initialize storage.
        
        Args:
            path: Path to JSON file. Created if doesn't exist.
        """
        self.path = Path(path)
        self._ensure_file()
    
    def _ensure_file(self) -> None:
        """Create file if it doesn't exist."""
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("[]")
    
    def _load(self) -> list[dict]:
        """Load all skills from file."""
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_all(self, skills: list[dict]) -> None:
        """Save all skills to file."""
        self.path.write_text(json.dumps(skills, ensure_ascii=False, indent=2, default=str))
    
    def _skill_to_dict(self, skill: Skill) -> dict:
        """Convert Skill to JSON-serializable dict."""
        return {
            "id": skill.id,
            "name": skill.name,
            "version": skill.version,
            "parent_id": skill.parent_id,
            "created_at": skill.created_at.isoformat() if skill.created_at else None,
            "metadata": {
                "description": skill.metadata.description,
                "goal": skill.metadata.goal,
                "keywords": skill.metadata.keywords,
                "embedding": skill.metadata.embedding,
            },
            "steps": [
                {
                    "order": step.order,
                    "instruction": step.instruction,
                    "expected_output": step.expected_output,
                }
                for step in skill.steps
            ],
        }
    
    def _dict_to_skill(self, data: dict) -> Skill:
        """Convert dict to Skill object."""
        return Skill(
            id=data["id"],
            name=data["name"],
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            metadata=SkillMetadata(
                description=data["metadata"]["description"],
                goal=data["metadata"]["goal"],
                keywords=data["metadata"].get("keywords", []),
                embedding=data["metadata"].get("embedding"),
            ),
            steps=[
                SkillStep(
                    order=s["order"],
                    instruction=s["instruction"],
                    expected_output=s.get("expected_output"),
                )
                for s in data.get("steps", [])
            ],
        )
    
    async def save(self, skill: Skill) -> None:
        """Save or update a skill."""
        skills = self._load()
        
        # Remove existing skill with same ID
        skills = [s for s in skills if s["id"] != skill.id]
        
        # Add new/updated skill
        skills.append(self._skill_to_dict(skill))
        
        self._save_all(skills)
    
    async def get(self, skill_id: str) -> Skill | None:
        """Get a skill by ID."""
        skills = self._load()
        for data in skills:
            if data["id"] == skill_id:
                return self._dict_to_skill(data)
        return None
    
    async def get_all(self) -> list[Skill]:
        """Get all skills."""
        return [self._dict_to_skill(data) for data in self._load()]
    
    async def delete(self, skill_id: str) -> None:
        """Delete a skill by ID."""
        skills = self._load()
        skills = [s for s in skills if s["id"] != skill_id]
        self._save_all(skills)
    
    async def search_by_embedding(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[Skill, float]]:
        """Search skills by embedding similarity."""
        results = []
        
        for data in self._load():
            skill_embedding = data.get("metadata", {}).get("embedding")
            if skill_embedding:
                score = cosine_similarity(embedding, skill_embedding)
                if score >= min_score:
                    results.append((self._dict_to_skill(data), score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    async def get_by_name(self, name: str) -> Skill | None:
        """Get a skill by name (case-insensitive)."""
        name_lower = name.lower()
        for data in self._load():
            if data["name"].lower() == name_lower:
                return self._dict_to_skill(data)
        return None
