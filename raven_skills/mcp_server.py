"""MCP Server for raven-skills.

Exposes skill-based agents as an MCP server that can be used by
Claude, Cursor, and other MCP-compatible clients.

Usage:
    python -m raven_skills serve --port 8000
    python -m raven_skills serve --transport stdio
"""

import json
import os
from typing import Any
from datetime import datetime

# MCP imports (optional dependency)
try:
    from mcp.server.fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    FastMCP = None

from raven_skills.storage.json_storage import JSONStorage
from raven_skills.agent import SkillAgent
from raven_skills.models.skill import Skill, SkillMetadata, SkillStep


def create_mcp_server(
    storage_path: str = "./skills.json",
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "text-embedding-3-small",
    base_url: str | None = None,
    api_key: str | None = None,
    embedding_base_url: str | None = None,
) -> "FastMCP":
    """Create and configure the MCP server.
    
    Args:
        storage_path: Path to JSON file for skill storage
        llm_model: LLM model for skill execution
        embedding_model: Model for embeddings
        base_url: Base URL for OpenAI-compatible API (e.g., Ollama)
        api_key: API key (overrides OPENAI_API_KEY env var)
        embedding_base_url: Separate base URL for embeddings
    
    Returns:
        Configured FastMCP server instance
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP SDK not installed. Install with: pip install mcp"
        )
    
    from openai import AsyncOpenAI
    
    # Create server
    mcp = FastMCP(name="raven-skills")
    
    # Create OpenAI clients
    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key
    
    client = AsyncOpenAI(**client_kwargs)
    
    # Embedding client (may be different from LLM client)
    embedding_client = None
    if embedding_base_url:
        emb_kwargs = {"base_url": embedding_base_url}
        if api_key:
            emb_kwargs["api_key"] = api_key
        embedding_client = AsyncOpenAI(**emb_kwargs)
    
    # Shared state
    storage = JSONStorage(storage_path)
    agent = SkillAgent(
        client=client,
        storage=storage,
        llm_model=llm_model,
        embedding_model=embedding_model,
        embedding_client=embedding_client,
    )
    
    # ─────────────────────────────────────────────────────────────
    # Tools
    # ─────────────────────────────────────────────────────────────
    
    @mcp.tool()
    async def execute_skill(query: str, skill_name: str | None = None) -> dict:
        """Execute a skill for the given query.
        
        If skill_name is provided, uses that specific skill.
        Otherwise, automatically matches the best skill for the query.
        
        Args:
            query: The task or question to solve
            skill_name: Optional specific skill name to use
        
        Returns:
            Execution result with status and output
        """
        import asyncio
        
        # If specific skill requested
        if skill_name:
            skill = await storage.get_by_name(skill_name)
            if not skill:
                return {
                    "status": "error",
                    "message": f"Skill '{skill_name}' not found",
                }
            task, _ = await agent.match(query)
        else:
            # Auto-match skill
            task, match_result = await agent.match(query)
            if not match_result.found:
                return {
                    "status": "no_skill",
                    "message": "No matching skill found for this query",
                    "suggestion": "Try listing available skills with list_skills()",
                }
            skill = match_result.skill
        
        # Execute skill
        result = await agent.execute(skill, task)
        
        return {
            "status": "success" if not result.error else "error",
            "skill_name": skill.name,
            "skill_id": skill.id,
            "output": result.output,
            "error": result.error,
            "steps_completed": len(result.step_outputs),
        }
    
    @mcp.tool()
    async def list_skills() -> list[dict]:
        """List all available skills.
        
        Returns a list of skills with their names, descriptions, and metadata.
        """
        skills = await storage.get_all()
        return [
            {
                "name": skill.name,
                "id": skill.id,
                "description": skill.metadata.description,
                "goal": skill.metadata.goal,
                "keywords": skill.metadata.keywords,
                "steps_count": len(skill.steps),
                "version": skill.version,
            }
            for skill in skills
        ]
    
    @mcp.tool()
    async def create_skill(
        name: str,
        description: str,
        goal: str,
        steps: list[str],
        keywords: list[str] | None = None,
    ) -> dict:
        """Create a new skill manually.
        
        Args:
            name: Short skill name (2-5 words)
            description: What the skill does
            goal: Expected outcome of successful execution
            steps: List of step instructions
            keywords: Optional search keywords
        
        Returns:
            Created skill info
        """
        from uuid import uuid4
        
        # Generate embedding for the skill
        skill_text = f"{name} {description} {goal}"
        embedding = await agent._embeddings.embed_text(skill_text)
        
        skill = Skill(
            id=f"skill-{uuid4().hex[:8]}",
            name=name,
            version=1,
            metadata=SkillMetadata(
                description=description,
                goal=goal,
                keywords=keywords or name.lower().split(),
                embedding=embedding,
            ),
            steps=[
                SkillStep(order=i + 1, instruction=step)
                for i, step in enumerate(steps)
            ],
            created_at=datetime.now(),
        )
        
        await storage.save(skill)
        
        return {
            "status": "created",
            "skill_id": skill.id,
            "skill_name": skill.name,
            "steps_count": len(skill.steps),
        }
    
    @mcp.tool()
    async def search_skills(query: str, top_k: int = 5) -> list[dict]:
        """Search for skills by semantic similarity.
        
        Args:
            query: Search query
            top_k: Maximum number of results
        
        Returns:
            List of matching skills with similarity scores
        """
        embedding = await agent._embeddings.embed_text(query)
        results = await storage.search_by_embedding(embedding, top_k=top_k, min_score=0.3)
        
        return [
            {
                "name": skill.name,
                "id": skill.id,
                "description": skill.metadata.description,
                "similarity": round(score, 3),
            }
            for skill, score in results
        ]
    
    # NOTE: delete_skill intentionally not exposed via MCP for security reasons.
    # Skills can only be deleted programmatically via the storage API.
    
    # ─────────────────────────────────────────────────────────────
    # Resources
    # ─────────────────────────────────────────────────────────────
    
    @mcp.resource("skills://library")
    async def skills_library() -> str:
        """Get all skills as JSON."""
        skills = await storage.get_all()
        return json.dumps([
            {
                "id": s.id,
                "name": s.name,
                "description": s.metadata.description,
                "goal": s.metadata.goal,
                "keywords": s.metadata.keywords,
                "steps": [step.instruction for step in s.steps],
            }
            for s in skills
        ], ensure_ascii=False, indent=2)
    
    @mcp.resource("skills://skill/{skill_id}")
    async def skill_detail(skill_id: str) -> str:
        """Get a specific skill as JSON."""
        skill = await storage.get(skill_id)
        if not skill:
            return json.dumps({"error": f"Skill '{skill_id}' not found"})
        
        return json.dumps({
            "id": skill.id,
            "name": skill.name,
            "version": skill.version,
            "description": skill.metadata.description,
            "goal": skill.metadata.goal,
            "keywords": skill.metadata.keywords,
            "steps": [
                {"order": s.order, "instruction": s.instruction}
                for s in skill.steps
            ],
            "created_at": skill.created_at.isoformat() if skill.created_at else None,
        }, ensure_ascii=False, indent=2)
    
    return mcp


# Convenience function for direct import
def get_server(**kwargs) -> "FastMCP":
    """Get the configured MCP server instance."""
    return create_mcp_server(**kwargs)
