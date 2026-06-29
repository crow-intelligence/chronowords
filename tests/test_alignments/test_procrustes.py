"""Tests for Procrustes alignment functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hynp

from chronowords.alignment.procrustes import ProcrustesAligner
from tests.strategies import embedding_pair_with_orthogonal_target


def test_basic_alignment(simple_embeddings):
    """Test basic Procrustes alignment with simple rotated embeddings."""
    source_vocab, source_emb, target_vocab, target_emb = simple_embeddings

    aligner = ProcrustesAligner()
    metrics = aligner.fit(source_emb, target_emb, source_vocab, target_vocab)

    assert metrics.num_aligned_words == 4
    assert metrics.average_cosine_similarity > 0.99
    assert metrics.alignment_error < 4.0001

    aligned_emb = aligner.transform(source_emb)
    assert np.allclose(aligned_emb, target_emb, atol=1e-4)


def test_partial_vocabulary():
    """Test alignment with partially overlapping vocabularies."""
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

    aligner = ProcrustesAligner()
    metrics = aligner.fit(source_emb, target_emb, source_vocab, target_vocab)

    assert metrics.num_aligned_words == 4
    assert metrics.average_cosine_similarity > 0


def test_zero_vector_handling():
    """Test handling of zero vectors in embeddings."""
    source_vocab = ["word1", "word2", "zero"]
    source_emb = np.array([[1.0, 1.0], [1.0, -1.0], [0.0, 0.0]])

    aligner = ProcrustesAligner()
    metrics = aligner.fit(
        source_emb, source_emb.copy(), source_vocab, source_vocab.copy()
    )

    assert metrics.num_aligned_words == 2


def test_frequency_rank_filtering():
    """Test alignment with frequency rank filtering."""
    source_vocab = ["the", "is", "king", "queen", "rare"]
    source_emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 1.0], [0.5, 0.5]])

    target_vocab = ["the", "is", "king", "queen", "rare2"]
    target_emb = source_emb.copy()

    aligner = ProcrustesAligner(min_freq_rank=2, max_freq_rank=4)
    metrics = aligner.fit(source_emb, target_emb, source_vocab, target_vocab)

    assert metrics.num_aligned_words == 2


def test_save_load():
    """Test saving and loading aligner state."""
    source_vocab = ["king", "queen"]
    source_emb = np.array([[1.0, 1.0], [-1.0, 1.0]])

    aligner = ProcrustesAligner()
    aligner.fit(source_emb, source_emb.copy(), source_vocab, source_vocab.copy())

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "aligner.pkl"
        aligner.save(path)

        new_aligner = ProcrustesAligner()
        new_aligner.load(path)

        assert aligner.orthogonal_matrix is not None
        assert new_aligner.orthogonal_matrix is not None
        assert np.allclose(aligner.orthogonal_matrix, new_aligner.orthogonal_matrix)
        assert aligner.source_words == new_aligner.source_words
        assert aligner.target_words == new_aligner.target_words


def test_error_cases():
    """Test error handling."""
    aligner = ProcrustesAligner()

    with pytest.raises(ValueError):
        aligner.transform(np.random.randn(3, 3))

    source_vocab = ["a", "b"]
    target_vocab = ["x", "y"]
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError):
        aligner.fit(emb, emb, source_vocab, target_vocab)

    zero_vocab = ["a", "b"]
    zero_emb = np.zeros((2, 2))
    with pytest.raises(ValueError, match="No valid anchor words found"):
        aligner.fit(zero_emb, zero_emb, zero_vocab, zero_vocab)


def test_get_word_similarity():
    """Test word similarity between source and target spaces."""
    aligner = ProcrustesAligner()
    aligner.source_words = ["cat", "dog"]
    aligner.target_words = ["cat", "dog"]
    aligner.orthogonal_matrix = np.eye(2)

    source_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    target_emb = np.array([[1.0, 0.0], [0.0, 1.0]])

    sim = aligner.get_word_similarity("cat", source_emb, target_emb)
    assert sim is not None
    assert abs(sim - 1.0) < 0.01


def test_get_word_similarity_unknown():
    """Test word similarity for unknown word returns None."""
    aligner = ProcrustesAligner()
    aligner.source_words = ["cat"]
    aligner.target_words = ["cat"]
    aligner.orthogonal_matrix = np.eye(2)

    emb = np.array([[1.0, 0.0]])
    assert aligner.get_word_similarity("unknown", emb, emb) is None


@st.composite
def _embedding_pair_with_shared_vocab(draw):
    """Draw (source_emb, target_emb, vocab) where target = source @ random_orthogonal."""
    n = draw(st.integers(min_value=5, max_value=20))
    d = draw(st.integers(min_value=2, max_value=10))
    source = draw(
        hynp.arrays(
            dtype=np.float64,
            shape=(n, d),
            elements=st.floats(
                min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    # Require non-trivial rows so normalization doesn't collapse anchors.
    row_norms = np.linalg.norm(source, axis=1)
    assume_nonzero = np.all(row_norms > 1e-3)
    if not assume_nonzero:
        source = source + 0.5
    # Random orthogonal target via QR decomposition of a random matrix.
    q, _ = np.linalg.qr(
        draw(
            hynp.arrays(
                dtype=np.float64,
                shape=(d, d),
                elements=st.floats(
                    min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
                ),
            )
        )
        + np.eye(d)
    )
    target = source @ q
    vocab = [f"w{i}" for i in range(n)]
    return source, target, vocab


@given(data=_embedding_pair_with_shared_vocab())
@settings(deadline=None, max_examples=30)
def test_learned_matrix_is_orthogonal(data):
    """Procrustes' defining guarantee: the learned transform is orthogonal.

    For any paired embeddings with shared vocabulary, `fit` must produce a
    matrix `R` such that `R @ R.T ≈ I`. Without this, `.transform` does not
    preserve distances and the whole alignment theory breaks down.
    """
    source, target, vocab = data
    aligner = ProcrustesAligner(min_freq_rank=0, max_freq_rank=len(vocab))
    aligner.fit(source, target, vocab, list(vocab))

    assert aligner.orthogonal_matrix is not None
    r = aligner.orthogonal_matrix
    d = r.shape[0]
    assert np.allclose(r @ r.T, np.eye(d), atol=1e-6)


@given(data=embedding_pair_with_orthogonal_target())
@settings(deadline=None, max_examples=30)
def test_transform_is_an_isometry(data):
    """Transform applies an orthogonal map, so it preserves vector norms.

    Distance preservation is the entire point of Procrustes alignment: an
    orthogonal R means `||x @ R|| == ||x||`. If transform changed norms, the
    aligned cosine similarities used for semantic-shift detection would be
    meaningless.
    """
    source, target, vocab, _ = data
    aligner = ProcrustesAligner(min_freq_rank=0, max_freq_rank=len(vocab))
    aligner.fit(source, target, vocab, list(vocab))

    transformed = aligner.transform(source)
    assert np.allclose(
        np.linalg.norm(transformed, axis=1),
        np.linalg.norm(source, axis=1),
        atol=1e-6,
    )


@given(data=embedding_pair_with_orthogonal_target())
@settings(deadline=None, max_examples=30)
def test_save_load_round_trip(data):
    """A fitted aligner survives a save/load cycle unchanged.

    Persistence is only useful if a reloaded aligner is indistinguishable from
    the original — same rotation, same anchor bookkeeping.
    """
    source, target, vocab, _ = data
    aligner = ProcrustesAligner(min_freq_rank=0, max_freq_rank=len(vocab))
    aligner.fit(source, target, vocab, list(vocab))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aligner.pkl"
        aligner.save(path)
        reloaded = ProcrustesAligner()
        reloaded.load(path)

    assert reloaded.orthogonal_matrix is not None
    assert aligner.orthogonal_matrix is not None
    assert np.allclose(reloaded.orthogonal_matrix, aligner.orthogonal_matrix)
    assert reloaded.source_words == aligner.source_words
    assert reloaded.target_words == aligner.target_words
    assert reloaded.anchors == aligner.anchors


def test_find_common_words_returns_intersection():
    """find_common_words returns only words present in BOTH vocabularies.

    Mutation-testing gap: intersection->union survived (fit silently skips the
    extra words) until this direct test.
    """
    aligner = ProcrustesAligner(min_freq_rank=0, max_freq_rank=10)
    source = ["king", "queen", "man"]
    target = ["queen", "man", "woman"]
    assert aligner.find_common_words(source, target) == ["man", "queen"]
