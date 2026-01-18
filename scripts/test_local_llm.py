"""Business tests with local Ollama LLM (qwen3-vl:30b).

Tests the library using entirely local models:
- LLM: qwen3-vl:30b via Ollama
- Embeddings: bge-m3 via Ollama
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import AsyncOpenAI
from raven_skills import SkillAgent, SkillStorage, Skill
from raven_skills.utils.similarity import cosine_similarity

# Import test suite from main business tests
from business_tests import BusinessTestSuite, InMemoryStorage


async def main():
    print("\n" + "🦙"*30)
    print("  raven-skills Local LLM Test")
    print("  LLM: qwen3-vl:30b (Ollama)")
    print("  Embeddings: bge-m3 (Ollama)")
    print("🦙"*30)
    
    # Create Ollama client for both LLM and embeddings
    ollama_client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    
    # Create agent with local LLM
    storage = InMemoryStorage()
    agent = SkillAgent(
        client=ollama_client,
        embedding_client=ollama_client,  # Same client for embeddings
        storage=storage,
        llm_model="qwen3-vl:30b",
        embedding_model="bge-m3:latest",
        similarity_threshold=0.55,
        validate_matches=False,
        use_structured_outputs=False,  # Ollama uses JSON parsing fallback
    )
    
    print("\n✅ SkillAgent initialized with local LLM")
    print("   LLM: qwen3-vl:30b")
    print("   Embeddings: bge-m3")
    print("   use_structured_outputs: False (JSON parsing mode)")
    
    # Run tests
    suite = BusinessTestSuite(agent)
    report = await suite.run_all()
    
    print("\n" + "="*70)
    print("✅ Local LLM Test Complete!")
    print("="*70 + "\n")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())
