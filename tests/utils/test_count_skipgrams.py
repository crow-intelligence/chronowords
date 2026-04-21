import numpy as np
import pytest

from chronowords.utils.count_skipgrams import PPMIComputer  # ty: ignore


@pytest.fixture
def ppmi_computer():
    """Create a PPMIComputer instance with toy corpus data."""
    skipgram_counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    word_counts = np.array([1, 1, 1], dtype=np.int32)
    seeds = [12345, 67890, 11111]

    return PPMIComputer(
        skipgram_counts=skipgram_counts,
        word_counts=word_counts,
        vocabulary=["word1", "word2", "word3"],
        seeds=seeds,
        width=10,
        skip_total=6.0,
        word_total=3.0,
        shift=1.0,
        alpha=0.75,
    )


def test_ppmi_matrix_positive_values(ppmi_computer):
    """Test that PPMI matrix values are non-negative."""
    ppmi_matrix = ppmi_computer.compute_ppmi_matrix_with_sketch(batch_size=2)
    assert np.all(ppmi_matrix.data >= 0)


def test_ppmi_empty_vocabulary():
    """Test PPMI computation with empty vocabulary."""
    counts = np.zeros((3, 10), dtype=np.int32)
    computer = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=[],
        seeds=[1, 2, 3],
        width=10,
        skip_total=0.0,
        word_total=0.0,
    )
    matrix = computer.compute_ppmi_matrix_with_sketch()
    assert matrix.shape == (0, 0)


def test_ppmi_single_word():
    """Test PPMI computation with single word vocabulary."""
    counts = np.array([[10], [10], [10]], dtype=np.int32)
    computer = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=["only"],
        seeds=[1, 2, 3],
        width=1,
        skip_total=10.0,
        word_total=10.0,
    )
    matrix = computer.compute_ppmi_matrix_with_sketch()
    assert matrix.shape == (1, 1)


def test_ppmi_all_zero_counts():
    """Test PPMI with all-zero count arrays."""
    counts = np.zeros((3, 10), dtype=np.int32)
    computer = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=["a", "b", "c"],
        seeds=[1, 2, 3],
        width=10,
        skip_total=1.0,
        word_total=1.0,
    )
    matrix = computer.compute_ppmi_matrix_with_sketch()
    assert matrix.nnz == 0


def test_ppmi_batch_boundaries():
    """Test PPMI batch processing at exact boundaries."""
    vocab_size = 4
    counts = np.zeros((3, 100), dtype=np.int32)
    counts[0, :vocab_size] = 10
    vocabulary = [f"w{i}" for i in range(vocab_size)]

    computer = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=vocabulary,
        seeds=[1, 2, 3],
        width=100,
        skip_total=40.0,
        word_total=40.0,
    )

    m1 = computer.compute_ppmi_matrix_with_sketch(batch_size=4)
    assert m1.shape == (vocab_size, vocab_size)

    computer2 = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=vocabulary,
        seeds=[1, 2, 3],
        width=100,
        skip_total=40.0,
        word_total=40.0,
    )
    m2 = computer2.compute_ppmi_matrix_with_sketch(batch_size=100)
    assert m2.shape == (vocab_size, vocab_size)


def test_ppmi_invalid_seeds():
    """Test that non-integer seeds raise ValueError."""
    counts = np.zeros((3, 10), dtype=np.int32)
    with pytest.raises(ValueError, match="All seeds must be integers"):
        PPMIComputer(
            skipgram_counts=counts,
            word_counts=counts,
            vocabulary=["a"],
            seeds=[1.5, 2.0, 3.0],
            width=10,
            skip_total=1.0,
            word_total=1.0,
        )
