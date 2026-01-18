"""Tests for similarity utilities."""

import pytest
import math

from raven_skills.utils.similarity import (
    cosine_similarity,
    normalize_embedding,
    euclidean_distance,
)


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        
        result = cosine_similarity(a, b)
        
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        
        result = cosine_similarity(a, b)
        
        assert result == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        
        result = cosine_similarity(a, b)
        
        assert result == pytest.approx(-1.0)

    def test_similar_vectors(self) -> None:
        a = [1.0, 1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        
        result = cosine_similarity(a, b)
        
        # cos(45°) ≈ 0.707
        assert result == pytest.approx(0.707, abs=0.01)

    def test_different_dimensions_raises(self) -> None:
        a = [1.0, 0.0]
        b = [1.0, 0.0, 0.0]
        
        with pytest.raises(ValueError, match="same dimensions"):
            cosine_similarity(a, b)

    def test_empty_vectors_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            cosine_similarity([], [])

    def test_zero_vector_returns_zero(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 1.0, 1.0]
        
        result = cosine_similarity(a, b)
        
        assert result == 0.0


class TestNormalizeEmbedding:
    """Tests for normalize_embedding function."""

    def test_normalize_unit_vector(self) -> None:
        vec = [1.0, 0.0, 0.0]
        
        result = normalize_embedding(vec)
        
        assert result == [1.0, 0.0, 0.0]

    def test_normalize_scales_to_unit(self) -> None:
        vec = [3.0, 4.0, 0.0]
        
        result = normalize_embedding(vec)
        
        # Magnitude should be 1.0
        magnitude = math.sqrt(sum(x * x for x in result))
        assert magnitude == pytest.approx(1.0)

    def test_normalize_preserves_direction(self) -> None:
        vec = [2.0, 2.0, 2.0]
        
        result = normalize_embedding(vec)
        
        # All components should be equal
        assert result[0] == pytest.approx(result[1])
        assert result[1] == pytest.approx(result[2])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            normalize_embedding([])

    def test_zero_vector_unchanged(self) -> None:
        vec = [0.0, 0.0, 0.0]
        
        result = normalize_embedding(vec)
        
        assert result == [0.0, 0.0, 0.0]


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_same_point_zero_distance(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        
        result = euclidean_distance(a, b)
        
        assert result == 0.0

    def test_distance_along_axis(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [3.0, 0.0, 0.0]
        
        result = euclidean_distance(a, b)
        
        assert result == 3.0

    def test_3d_distance(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 2.0]
        
        result = euclidean_distance(a, b)
        
        assert result == 3.0  # sqrt(1 + 4 + 4) = 3

    def test_different_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="same dimensions"):
            euclidean_distance([1.0], [1.0, 2.0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            euclidean_distance([], [])
