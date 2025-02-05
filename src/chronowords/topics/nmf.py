"""
Topic modeling using NMF on PPMI matrices with support for temporal alignment.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.decomposition import NMF


@dataclass
class Topic:
    """Container for topic information.

    Fields:
        id: Unique topic identifier
        words: List of (word, weight) pairs for top words
        distribution: Full probability distribution over vocabulary
    """

    id: int
    words: List[Tuple[str, float]]  # (word, weight) pairs
    distribution: np.ndarray  # Full word distribution


@dataclass
class AlignedTopic:
    """Container for aligned topic pairs.

    Fields:
        source_topic: Topic from source time period
        target_topic: Topic from target time period
        similarity: Cosine similarity between topics
    """

    source_topic: Topic
    target_topic: Topic
    similarity: float


class TopicModel:
    """
    Topic model using NMF on PPMI matrices.
    Supports temporal alignment of topics between different time periods.
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_iter: int = 500,
        min_similarity: float = 0.1,
    ):
        """
        Initialize topic model.

        Args:
            n_topics: Number of topics to extract
            max_iter: Maximum number of iterations for NMF
            min_similarity: Minimum similarity for topic alignment
        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.min_similarity = min_similarity

        self.nmf = NMF(
            n_components=n_topics,
            max_iter=max_iter,
            init="nndsvd",  # Better initialization for sparse data
        )

        self.vocabulary: List[str] = []
        self.topics: List[Topic] = []
        self.topic_word_matrix: Optional[np.ndarray] = None

    def fit(
        self, ppmi_matrix: csr_matrix, vocabulary: List[str], top_n_words: int = 10
    ) -> None:
        """Fit topic model to PPMI matrix.

        Args:
            ppmi_matrix: Sparse PPMI matrix from word embeddings
            vocabulary: List of words corresponding to matrix columns
            top_n_words: Number of top words to store per topic

        The method performs NMF factorization and normalizes topic distributions.
        """
        self.vocabulary = vocabulary

        # Run NMF
        self.topic_word_matrix = self.nmf.fit_transform(ppmi_matrix)
        word_topic_matrix = self.nmf.components_

        # Create topic objects
        self.topics = []
        for topic_idx in range(self.n_topics):
            # Get word weights for this topic
            word_weights = word_topic_matrix[topic_idx]

            # Normalize weights
            word_weights = (
                word_weights / word_weights.sum()
                if word_weights.sum() > 0
                else word_weights
            )

            # Get top words
            top_indices = np.argsort(word_weights)[-top_n_words:][::-1]
            top_words = [
                (vocabulary[idx], float(word_weights[idx])) for idx in top_indices
            ]

            # Create topic object with normalized distribution
            topic = Topic(
                id=topic_idx,
                words=top_words,
                distribution=word_weights,  # This is now normalized
            )
            self.topics.append(topic)

    def get_document_topics(
        self, doc_vector: np.ndarray, threshold: float = 0.1
    ) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a document vector.

        Args:
            doc_vector: Document vector in vocabulary space
            threshold: Minimum topic proportion to include

        Returns:
            List of (topic_id, weight) pairs
        """
        if self.topic_word_matrix is None:
            raise ValueError("Model must be fit before getting document topics")

        # Project document into topic space
        doc_topics = self.nmf.transform(doc_vector.reshape(1, -1))[0]

        # Normalize
        doc_topics = doc_topics / np.sum(doc_topics)

        # Get topics above threshold
        topic_weights = [
            (idx, float(weight))
            for idx, weight in enumerate(doc_topics)
            if weight > threshold
        ]

        return sorted(topic_weights, key=lambda x: x[1], reverse=True)

    def _compute_topic_similarity(self, topic1: Topic, topic2: Topic) -> float:
        """Compute cosine similarity between topic distributions."""
        return 1 - cosine(topic1.distribution, topic2.distribution)

    def align_with(self, other: "TopicModel") -> List[AlignedTopic]:
        """Align topics with another model using Hungarian algorithm.

        Finds optimal matching between topic sets by maximizing total similarity.
        Only returns pairs above min_similarity threshold.

        Args:
            other: Another fitted TopicModel

        Returns:
            List of aligned topic pairs sorted by similarity
        """
        if not self.topics or not other.topics:
            raise ValueError("Both models must be fit before alignment")

        # Compute cost matrix
        cost_matrix = np.zeros((self.n_topics, other.n_topics))
        for i, topic1 in enumerate(self.topics):
            for j, topic2 in enumerate(other.topics):
                # Convert similarity to cost (higher similarity = lower cost)
                similarity = self._compute_topic_similarity(topic1, topic2)
                cost_matrix[i, j] = 1 - similarity

        # Find optimal matching
        source_indices, target_indices = linear_sum_assignment(cost_matrix)

        # Create aligned topic pairs
        aligned_topics = []
        for source_idx, target_idx in zip(source_indices, target_indices):
            similarity = 1 - cost_matrix[source_idx, target_idx]

            # Only include if similarity is above threshold
            if similarity >= self.min_similarity:
                aligned_topics.append(
                    AlignedTopic(
                        source_topic=self.topics[source_idx],
                        target_topic=other.topics[target_idx],
                        similarity=float(similarity),
                    )
                )

        return aligned_topics

    def print_topics(self, top_n: int = 10) -> None:
        """Print top words for each topic."""
        for topic in self.topics:
            print(f"\nTopic {topic.id}:")
            for word, weight in topic.words[:top_n]:
                print(f"  {word}: {weight:.4f}")
