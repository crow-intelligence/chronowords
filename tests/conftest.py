"""Shared fixtures for chronowords tests."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from chronowords.utils.probabilistic_counter import CountMinSketch


@pytest.fixture
def small_corpus():
    """Create a test corpus with richer patterns and relationships."""
    return [
        "king queen crown palace royal throne",
        "king rules kingdom palace throne",
        "queen rules kingdom crown royal",
        "prince princess royal palace crown",
        "man woman child family home",
        "boy girl child plays home",
        "father mother parent child family",
        "king leads army battle victory",
        "queen commands navy battle victory",
        "king man queen woman",
        "king queen crown palace royal throne",
        "king rules kingdom palace throne",
        "queen rules kingdom crown royal",
        "man woman child family home",
    ]


@pytest.fixture
def small_sketch():
    """Create a small Count-Min Sketch for testing."""
    return CountMinSketch(width=100, depth=3, seed=42)


@pytest.fixture
def filled_sketch(small_sketch, small_corpus):
    """Create a Count-Min Sketch filled with test data."""
    for line in small_corpus:
        for word in line.split():
            small_sketch.update(word)
    return small_sketch


@pytest.fixture
def simple_embeddings():
    """Create simple embeddings for testing alignment."""
    source_vocab = ["king", "queen", "man", "woman"]
    source_embeddings = np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]
    )

    angle = np.pi / 4
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    target_embeddings = source_embeddings @ rotation
    target_vocab = source_vocab.copy()

    return source_vocab, source_embeddings, target_vocab, target_embeddings


@pytest.fixture
def simple_ppmi():
    """Create a simple PPMI matrix for testing."""
    matrix = np.array(
        [
            [0.5, 0.4, 0.0, 0.0, 0.0],
            [0.4, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.4, 0.0],
            [0.0, 0.0, 0.4, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    vocabulary = ["king", "queen", "dog", "cat", "computer"]
    return csr_matrix(matrix), vocabulary
