"""Internal embeddings client wrapper.

Wraps OpenAI client and provides methods for generating embeddings.
"""

from typing import Any


class EmbeddingsClient:
    """Internal embeddings client that handles all embedding operations.
    
    Wraps an OpenAI-compatible async client and provides methods
    for generating text embeddings.
    
    Args:
        client: An async OpenAI client instance (AsyncOpenAI or compatible).
        model: Model identifier for embeddings (default: text-embedding-3-small).
    """
    
    def __init__(
        self,
        client: Any,
        model: str = "text-embedding-3-small",
    ):
        self.client = client
        self.model = model
    
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single request."""
        if not texts:
            return []
        
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        # Sort by index to ensure correct order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
