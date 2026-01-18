"""Similarity and embedding utilities."""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        a: First vector.
        b: Second vector.
        
    Returns:
        Cosine similarity score between -1.0 and 1.0.
        
    Raises:
        ValueError: If vectors have different dimensions or are empty.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have same dimensions: {len(a)} != {len(b)}")
    
    if len(a) == 0:
        raise ValueError("Vectors cannot be empty")
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def normalize_embedding(embedding: list[float]) -> list[float]:
    """Normalize an embedding vector to unit length.
    
    Args:
        embedding: The vector to normalize.
        
    Returns:
        Normalized vector with magnitude 1.0.
        
    Raises:
        ValueError: If the vector is empty.
    """
    if len(embedding) == 0:
        raise ValueError("Embedding cannot be empty")
    
    norm = math.sqrt(sum(x * x for x in embedding))
    
    if norm == 0:
        return embedding
    
    return [x / norm for x in embedding]


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute Euclidean distance between two vectors.
    
    Args:
        a: First vector.
        b: Second vector.
        
    Returns:
        Euclidean distance (>= 0).
        
    Raises:
        ValueError: If vectors have different dimensions or are empty.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors must have same dimensions: {len(a)} != {len(b)}")
    
    if len(a) == 0:
        raise ValueError("Vectors cannot be empty")
    
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
