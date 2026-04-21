"""Tests for NMF topic modeling."""

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from scipy.sparse import csr_matrix

from chronowords.topics.nmf import TopicModel


def test_topic_model_basic(simple_ppmi):
    """Test basic topic model functionality."""
    ppmi_matrix, vocabulary = simple_ppmi

    model = TopicModel(n_topics=3)
    model.fit(ppmi_matrix, vocabulary)

    assert len(model.topics) == 3

    for topic in model.topics:
        assert len(topic.words) > 0
        assert all(
            isinstance(word, str) and isinstance(weight, float)
            for word, weight in topic.words
        )

    for topic in model.topics:
        assert abs(np.sum(topic.distribution) - 1.0) < 0.1


def test_document_topics(simple_ppmi):
    """Test document topic assignment."""
    ppmi_matrix, vocabulary = simple_ppmi

    model = TopicModel(n_topics=3)
    model.fit(ppmi_matrix, vocabulary)

    doc_vector = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    doc_topics = model.get_document_topics(doc_vector)

    assert len(doc_topics) > 0
    assert all(0 <= weight <= 1 for _, weight in doc_topics)

    total_weight = sum(weight for _, weight in doc_topics)
    assert abs(total_weight - 1.0) < 0.1


def test_topic_alignment(simple_ppmi):
    """Test topic alignment between two models."""
    ppmi_matrix, vocabulary = simple_ppmi

    model1 = TopicModel(n_topics=3)
    model2 = TopicModel(n_topics=3)

    model1.fit(ppmi_matrix, vocabulary)
    model2.fit(ppmi_matrix, vocabulary)

    aligned_topics = model1.align_with(model2)

    assert len(aligned_topics) > 0

    for aligned in aligned_topics:
        assert aligned.source_topic is not None
        assert aligned.target_topic is not None
        assert 0 <= aligned.similarity <= 1
        assert len(aligned.source_topic.words) == len(aligned.target_topic.words)


def test_topic_printing(simple_ppmi, capsys):
    """Test topic printing functionality."""
    ppmi_matrix, vocabulary = simple_ppmi

    model = TopicModel(n_topics=3)
    model.fit(ppmi_matrix, vocabulary)

    model.print_topics()

    captured = capsys.readouterr()
    assert captured.out != ""
    assert "Topic" in captured.out


def test_edge_cases():
    """Test edge cases and error conditions."""
    model = TopicModel(n_topics=3)

    with pytest.raises(ValueError):
        model.get_document_topics(np.array([1, 2, 3]))

    model2 = TopicModel(n_topics=3)
    with pytest.raises(ValueError):
        model.align_with(model2)


def test_single_topic():
    """Test model with a single topic."""
    matrix = csr_matrix(np.array([[1.0, 0.5], [0.5, 1.0]]))
    model = TopicModel(n_topics=1)
    model.fit(matrix, ["word1", "word2"])

    assert len(model.topics) == 1
    assert len(model.topics[0].words) > 0


def test_alignment_different_vocabularies():
    """Test topic alignment with different vocabularies."""
    ppmi1 = csr_matrix(np.array([[1.0, 0.5], [0.5, 1.0]]))
    ppmi2 = csr_matrix(np.array([[1.0, 0.3], [0.3, 1.0]]))

    model1 = TopicModel(n_topics=2)
    model2 = TopicModel(n_topics=2)

    model1.fit(ppmi1, ["cat", "dog"])
    model2.fit(ppmi2, ["cat", "bird"])

    aligned = model1.align_with(model2)
    assert len(aligned) > 0


@given(n_topics=st.integers(min_value=2, max_value=5))
@settings(
    deadline=None,
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_topic_distributions_are_probability_distributions(simple_ppmi, n_topics):
    """After fit, each topic's `distribution` must be a valid probability vector.

    All entries must be non-negative, and the vector must sum to ~1. Downstream
    code (alignment, document-topic assignment) treats these distributions as
    probabilities; a violation breaks every consumer of the `Topic` object.
    """
    ppmi_matrix, vocabulary = simple_ppmi

    model = TopicModel(n_topics=n_topics)
    model.fit(ppmi_matrix, vocabulary)

    for topic in model.topics:
        assert np.all(topic.distribution >= 0)
        assert abs(float(np.sum(topic.distribution)) - 1.0) < 1e-6
