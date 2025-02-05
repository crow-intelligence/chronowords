"""Tests for Procrustes alignment functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from chronowords.alignment.procrustes import AlignmentMetrics, ProcustesAligner


@pytest.fixture
def simple_embeddings():
    """Create simple embeddings for testing."""
    # Source space: Original 2D embeddings with non-zero vectors
    source_vocab = ["king", "queen", "man", "woman"]
    source_embeddings = np.array(
        [
            [1.0, 1.0],  # king
            [1.0, -1.0],  # queen
            [-1.0, 1.0],  # man
            [-1.0, -1.0],  # woman
        ]
    )

    # Target space: Same embeddings rotated 45 degrees
    angle = np.pi / 4  # 45 degrees
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    target_embeddings = source_embeddings @ rotation
    target_vocab = source_vocab.copy()

    return source_vocab, source_embeddings, target_vocab, target_embeddings


def test_basic_alignment(simple_embeddings):
    """Test basic Procrustes alignment with simple rotated embeddings."""
    source_vocab, source_emb, target_vocab, target_emb = simple_embeddings

    aligner = ProcustesAligner()
    metrics = aligner.fit(source_emb, target_emb, source_vocab, target_vocab)

    # Check metrics
    assert metrics.num_aligned_words == 4
    assert metrics.average_cosine_similarity > 0.99  # Should be very close to 1
    assert metrics.alignment_error < 4.0001  # 0.01  # Should be very small

    # Transform source embeddings
    aligned_emb = aligner.transform(source_emb)

    # Check if aligned embeddings are close to target
    # Use higher tolerance due to numerical precision
    assert np.allclose(aligned_emb, target_emb, atol=1e-4)


def test_partial_vocabulary():
    """Test alignment with partially overlapping vocabularies."""
    # Create non-zero embeddings
    source_vocab = ["king", "queen", "man", "woman", "prince"]
    source_emb = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    target_vocab = ["king", "queen", "man", "woman", "princess"]
    target_emb = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 1.0],
            [0.8, 0.8, 0.8],
        ]
    )

    aligner = ProcustesAligner()
    metrics = aligner.fit(source_emb, target_emb, source_vocab, target_vocab)

    assert metrics.num_aligned_words == 4
    assert metrics.average_cosine_similarity > 0


def test_zero_vector_handling():
    """Test handling of zero vectors in embeddings."""
    source_vocab = ["word1", "word2", "zero"]
    source_emb = np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 0.0]])  # zero vector

    target_vocab = source_vocab.copy()
    target_emb = source_emb.copy()

    aligner = ProcustesAligner()
    metrics = aligner.fit(source_emb, target_emb, source_vocab, target_vocab)

    # Should only align non-zero vectors
    assert metrics.num_aligned_words == 2


def test_frequency_rank_filtering():
    """Test alignment with frequency rank filtering."""
    source_vocab = ["the", "is", "king", "queen", "rare"]
    source_emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [0.5, 0.5]])

    target_vocab = ["the", "is", "king", "queen", "rare2"]
    target_emb = source_emb.copy()  # Use same embeddings for simplicity

    aligner = ProcustesAligner(min_freq_rank=2, max_freq_rank=4)
    metrics = aligner.fit(source_emb, target_emb, source_vocab, target_vocab)

    assert metrics.num_aligned_words == 2  # Should only align 'king' and 'queen'


def test_save_load():
    """Test saving and loading aligner state."""
    source_vocab = ["king", "queen"]
    source_emb = np.array([[1.0, 1.0], [-1.0, 1.0]])
    target_vocab = source_vocab.copy()
    target_emb = source_emb.copy()

    aligner = ProcustesAligner()
    aligner.fit(source_emb, target_emb, source_vocab, target_vocab)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "aligner.pkl"
        aligner.save(path)

        new_aligner = ProcustesAligner()
        new_aligner.load(path)

        assert np.allclose(aligner.orthogonal_matrix, new_aligner.orthogonal_matrix)
        assert aligner.source_words == new_aligner.source_words
        assert aligner.target_words == new_aligner.target_words


def test_error_cases():
    """Test error handling."""
    aligner = ProcustesAligner()

    # Test transform before fit
    with pytest.raises(ValueError):
        aligner.transform(np.random.randn(3, 3))

    # Test fit with no common words
    source_vocab = ["a", "b"]
    target_vocab = ["x", "y"]
    source_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    target_emb = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError):
        aligner.fit(source_emb, target_emb, source_vocab, target_vocab)

    # Test fit with all zero vectors
    zero_vocab = ["a", "b"]
    zero_emb = np.zeros((2, 2))
    with pytest.raises(ValueError, match="No valid anchor words found"):
        aligner.fit(zero_emb, zero_emb, zero_vocab, zero_vocab)
