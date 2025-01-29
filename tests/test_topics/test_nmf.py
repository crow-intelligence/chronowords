"""Tests for NMF topic modeling."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from chronowords.topics.nmf import TopicModel


@pytest.fixture
def simple_ppmi():
    """Create a simple PPMI matrix for testing."""
    # Create a small PPMI matrix with clear topic structure
    matrix = np.array(
        [
            [0.5, 0.4, 0.0, 0.0, 0.0],  # Topic 1: word1, word2
            [0.4, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.4, 0.0],  # Topic 2: word3, word4
            [0.0, 0.0, 0.4, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],  # Topic 3: word5
        ]
    )
    vocabulary = ["king", "queen", "dog", "cat", "computer"]
    return csr_matrix(matrix), vocabulary


@pytest.fixture
def simple_doc_vector():
    """Create a simple document vector for testing."""
    return np.array([0.5, 0.5, 0.0, 0.0, 0.0])  # Document about royalty


def test_topic_model_basic(simple_ppmi):
    """Test basic topic model functionality."""
    ppmi_matrix, vocabulary = simple_ppmi

    model = TopicModel(n_topics=3)
    model.fit(ppmi_matrix, vocabulary)

    # Check if we got the right number of topics
    assert len(model.topics) == 3

    # Each topic should have words with weights
    for topic in model.topics:
        assert len(topic.words) > 0
        assert all(
            isinstance(word, str) and isinstance(weight, float)
            for word, weight in topic.words
        )

    # Topic distributions should sum to approximately 1
    for topic in model.topics:
        assert abs(np.sum(topic.distribution) - 1.0) < 0.1


def test_document_topics(simple_ppmi, simple_doc_vector):
    """Test document topic assignment."""
    ppmi_matrix, vocabulary = simple_ppmi

    model = TopicModel(n_topics=3)
    model.fit(ppmi_matrix, vocabulary)

    # Get topics for a document about royalty
    doc_topics = model.get_document_topics(simple_doc_vector)

    # Should get at least one topic
    assert len(doc_topics) > 0

    # Topic weights should be between 0 and 1
    assert all(0 <= weight <= 1 for _, weight in doc_topics)

    # Weights should sum to approximately 1
    total_weight = sum(weight for _, weight in doc_topics)
    assert abs(total_weight - 1.0) < 0.1


def test_topic_alignment(simple_ppmi):
    """Test topic alignment between two models."""
    ppmi_matrix, vocabulary = simple_ppmi

    # Create two models
    model1 = TopicModel(n_topics=3)
    model2 = TopicModel(n_topics=3)

    # Fit both models
    model1.fit(ppmi_matrix, vocabulary)
    model2.fit(ppmi_matrix, vocabulary)  # Same data for testing

    # Align topics
    aligned_topics = model1.align_with(model2)

    # Should get some aligned topics
    assert len(aligned_topics) > 0

    # Check aligned topic properties
    for aligned in aligned_topics:
        # Should have both source and target topics
        assert aligned.source_topic is not None
        assert aligned.target_topic is not None

        # Similarity should be between 0 and 1
        assert 0 <= aligned.similarity <= 1

        # Topics should have same structure
        assert len(aligned.source_topic.words) == len(aligned.target_topic.words)


def test_topic_printing(simple_ppmi, capsys):
    """Test topic printing functionality."""
    ppmi_matrix, vocabulary = simple_ppmi

    model = TopicModel(n_topics=3)
    model.fit(ppmi_matrix, vocabulary)

    # Print topics
    model.print_topics()

    # Check if anything was printed
    captured = capsys.readouterr()
    assert captured.out != ""
    assert "Topic" in captured.out


def test_edge_cases():
    """Test edge cases and error conditions."""
    model = TopicModel(n_topics=3)

    # Should raise error if not fit
    with pytest.raises(ValueError):
        model.get_document_topics(np.array([1, 2, 3]))

    # Should raise error if trying to align unfit models
    model2 = TopicModel(n_topics=3)
    with pytest.raises(ValueError):
        model.align_with(model2)
