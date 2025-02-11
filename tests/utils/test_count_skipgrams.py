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
def ppmi_computer(toy_corpus_data):
    """Create a PPMIComputer instance with toy corpus data."""
    # Unpack the toy corpus data
    skipgram_counts, word_counts, vocabulary, seeds = toy_corpus_data

    # Initialize the PPMIComputer
    return PPMIComputer(skipgram_counts, word_counts, vocabulary, seeds, 5, 20, 20)


def test_ppmi_matrix_positive_values(ppmi_computer):
    """Ompute the PPMI matrix using the PPMIComputer."""
    ppmi_matrix = ppmi_computer.compute_ppmi_matrix_with_sketch(batch_size=2)

    # Check if all values in the PPMI matrix are positive or zero
    assert np.all(ppmi_matrix.data >= 0), "PPMI matrix contains negative values!"

    # Optionally, print the matrix for inspection (useful for debugging)
    print("PPMI Matrix:")
    print(ppmi_matrix.toarray())
