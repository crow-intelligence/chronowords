import numpy as np
import pytest

from chronowords.utils.count_skipgrams import PPMIComputer


@pytest.fixture
def toy_corpus_data():
    """Create a simple toy corpus and corresponding word count data."""
    skipgram_counts = np.array(
        [
            [5, 2, 1, 0],  # Dog - barks, cat, chases, runs
            [2, 5, 2, 1],  # Cat - dog, meows, chases, runs
            [1, 2, 5, 0],  # Dog - cat, meows, chases, runs
            [0, 1, 0, 5],  # Cat - dog, meows, chases, runs
        ],
        dtype=np.int32,
    )

    word_counts = np.array(
        [5, 7, 5, 5], dtype=np.int32
    )  # Word frequency counts for dog, cat, barks, meows
    vocabulary = ["dog", "cat", "barks", "meows"]
    seeds = [42, 42, 42]  # Example seeds

    return skipgram_counts, word_counts, vocabulary, seeds


@pytest.fixture
def ppmi_computer():
    """Create a PPMIComputer instance with toy corpus data."""
    # Ensure arrays are np.int32
    skipgram_counts = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    word_counts = np.array([1, 1, 1], dtype=np.int32)
    seeds = [12345, 67890, 11111]  # Integer seeds are fine

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
    """Ompute the PPMI matrix using the PPMIComputer."""
    ppmi_matrix = ppmi_computer.compute_ppmi_matrix_with_sketch(batch_size=2)

    # Check if all values in the PPMI matrix are positive or zero
    assert np.all(ppmi_matrix.data >= 0), "PPMI matrix contains negative values!"

    # Optionally, print the matrix for inspection (useful for debugging)
    print("PPMI Matrix:")
    print(ppmi_matrix.toarray())
