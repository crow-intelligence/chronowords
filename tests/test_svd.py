import numpy as np
import pytest

from chronowords.algebra.svd import SVDAlgebra


@pytest.fixture
def tiny_corpus():
    """Create a tiny test corpus."""
    return [
        "the cat sat on the mat",
        "the dog ran in the park",
        "a cat and a dog played",
        "the mat was on the floor",
        "dogs and cats are animals",
    ]


@pytest.fixture
def test_model():
    """Create a test model."""
    return SVDAlgebra(
        n_components=10,  # Small number for testing
        window_size=2,  # Small window for testing
        min_word_length=2,  # Allow shorter words for testing
    )


def test_model_initialization(test_model):
    """Test if model initializes with correct parameters."""
    assert test_model.n_components == 10
    assert test_model.window_size == 2
    assert test_model.min_word_length == 2
    assert test_model.vocabulary == []
    assert test_model.embeddings is None


def test_model_training(test_model, tiny_corpus):
    """Test if model trains correctly on tiny corpus."""

    # Create generator to mimic file reading
    def corpus_generator():
        for line in tiny_corpus:
            yield line

    # Train model
    test_model.train(corpus_generator())

    # Basic checks
    assert len(test_model.vocabulary) > 0
    assert test_model.embeddings is not None
    assert test_model.embeddings.shape[1] == test_model.n_components

    # Check if common words are in vocabulary
    common_words = ["cat", "dog", "the", "mat"]
    for word in common_words:
        assert word in test_model.vocabulary


def test_word_similarity(test_model, tiny_corpus):
    """Test word similarity calculations."""
    # Train model first
    test_model.train((line for line in tiny_corpus))

    # Test most similar words
    similar_to_cat = test_model.most_similar("cat", n=2)
    assert len(similar_to_cat) > 0
    assert all(isinstance(result.word, str) for result in similar_to_cat)
    assert all(isinstance(result.similarity, float) for result in similar_to_cat)

    # Test word distance
    distance = test_model.distance("cat", "dog")
    assert distance is not None
    assert 0 <= distance <= 1


def test_edge_cases(test_model, tiny_corpus):
    """Test edge cases and error handling."""
    test_model.train((line for line in tiny_corpus))

    # Test with unknown word
    assert test_model.get_vector("nonexistentword") is None

    # Test with empty string
    assert test_model.get_vector("") is None

    # Test similarity with unknown word
    assert test_model.most_similar("nonexistentword") == []

    # Test distance with unknown word
    assert test_model.distance("cat", "nonexistentword") is None


def test_vector_properties(test_model, tiny_corpus):
    """Test properties of the word vectors."""
    test_model.train((line for line in tiny_corpus))

    # Get vectors for some words
    cat_vector = test_model.get_vector("cat")
    dog_vector = test_model.get_vector("dog")

    # Check vector dimensions
    assert cat_vector.shape == (test_model.n_components,)
    assert dog_vector.shape == (test_model.n_components,)

    # Check if vectors are normalized
    assert np.abs(np.linalg.norm(cat_vector) - 1.0) < 1e-6
    assert np.abs(np.linalg.norm(dog_vector) - 1.0) < 1e-6
