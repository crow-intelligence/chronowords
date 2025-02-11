"""Topic modeling using NMF on PPMI matrices with support for temporal alignment."""

from dataclasses import dataclass

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

    Examples
    --------
        >>> import numpy as np
        >>> dist = np.array([0.5, 0.3, 0.2])
        >>> topic = Topic(1, [('cat', 0.5), ('dog', 0.3)], dist)
        >>> topic.id
        1
        >>> topic.words
        [('cat', 0.5), ('dog', 0.3)]
        >>> np.allclose(topic.distribution, [0.5, 0.3, 0.2])
        True

    """

    id: int
    words: list[tuple[str, float]]  # (word, weight) pairs
    distribution: np.ndarray  # Full word distribution


@dataclass
class AlignedTopic:
    """Container for aligned topic pairs.

    Fields:
        source_topic: Topic from source time period
        target_topic: Topic from target time period
        similarity: Cosine similarity between topics

    Examples
    --------
        >>> import numpy as np
        >>> dist = np.array([0.5, 0.3, 0.2])
        >>> topic1 = Topic(1, [('cat', 0.5)], dist)
        >>> topic2 = Topic(2, [('dog', 0.4)], dist)
        >>> aligned = AlignedTopic(topic1, topic2, 0.8)
        >>> aligned.source_topic.id
        1
        >>> aligned.target_topic.id
        2
        >>> aligned.similarity
        0.8

    """

    source_topic: Topic
    target_topic: Topic
    similarity: float


class TopicModel:
    """Topic model using NMF on PPMI matrices.

    Supports temporal alignment of topics between different time periods.
    """

    def __init__(
        self,
        n_topics: int = 10,
        max_iter: int = 500,
        min_similarity: float = 0.1,
    ) -> None:
        """Initialize topic model.

        Args:
        ----
            n_topics: Number of topics to extract
            max_iter: Maximum number of iterations for NMF
            min_similarity: Minimum similarity for topic alignment

        Examples:
        --------
            >>> model = TopicModel(n_topics=5, max_iter=100)
            >>> model.n_topics
            5
            >>> model.max_iter
            100

        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.min_similarity = min_similarity

        self.nmf = NMF(
            n_components=n_topics,
            max_iter=max_iter,
            init="nndsvd",  # Better initialization for sparse data
        )

        self.vocabulary: list[str] = []
        self.topics: list[Topic] = []
        self.topic_word_matrix: np.ndarray | None = None

    def fit(
        self, ppmi_matrix: csr_matrix, vocabulary: list[str], top_n_words: int = 10
    ) -> None:
        """Fit topic model to PPMI matrix.

        Args:
        ----
            ppmi_matrix: Sparse PPMI matrix from word embeddings
            vocabulary: List of words corresponding to matrix columns
            top_n_words: Number of top words to store per topic

        Examples:
        --------
            >>> import numpy as np
            >>> from scipy.sparse import csr_matrix
            >>> model = TopicModel(n_topics=2)
            >>> ppmi = csr_matrix([[1, 0], [0, 1]])
            >>> model.fit(ppmi, ['word1', 'word2'])
            >>> len(model.topics)
            2
            >>> isinstance(model.topics[0], Topic)
            True
            >>> len(model.vocabulary)
            2

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
                distribution=word_weights,
            )
            self.topics.append(topic)

    def get_document_topics(
        self, doc_vector: np.ndarray, threshold: float = 0.1
    ) -> list[tuple[int, float]]:
        """Get topic distribution for a document vector.

        Args:
        ----
            doc_vector: Document vector in vocabulary space
            threshold: Minimum topic proportion to include

        Returns:
        -------
            List of (topic_id, weight) pairs above threshold, sorted by weight

        Examples:
        --------
            >>> import numpy as np
            >>> from scipy.sparse import csr_matrix
            >>> model = TopicModel(n_topics=2)
            >>> ppmi = csr_matrix([[1, 0], [0, 1]])
            >>> model.fit(ppmi, ['word1', 'word2'])
            >>> doc = np.array([0.8, 0.2])
            >>> topics = model.get_document_topics(doc, threshold=0.1)
            >>> len(topics) > 0
            True
            >>> all(w >= 0.1 for _, w in topics)
            True

        """
        if self.topic_word_matrix is None:
            raise ValueError("Model must be fit before getting document topics")

        # Project document into topic space
        doc_topics = self.nmf.transform(doc_vector.reshape(1, -1))[0]

        # Normalize
        doc_topics = (
            doc_topics / np.sum(doc_topics) if np.sum(doc_topics) > 0 else doc_topics
        )

        # Get topics above threshold
        topic_weights = [
            (idx, float(weight))
            for idx, weight in enumerate(doc_topics)
            if weight > threshold
        ]

        return sorted(topic_weights, key=lambda x: x[1], reverse=True)

    def _align_distributions(
        self, topic1: Topic, topic2: Topic, vocab1: list[str], vocab2: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Align two topic distributions to use the same vocabulary space.

        Args:
        ----
            topic1: First topic
            topic2: Second topic
            vocab1: Vocabulary for first topic
            vocab2: Vocabulary for second topic

        Returns:
        -------
            Tuple of aligned distributions

        Examples:
        --------
            >>> import numpy as np
            >>> model = TopicModel()
            >>> dist1 = np.array([0.6, 0.4])
            >>> dist2 = np.array([0.3, 0.7])
            >>> t1 = Topic(1, [('cat', 0.6), ('dog', 0.4)], dist1)
            >>> t2 = Topic(2, [('dog', 0.3), ('bird', 0.7)], dist2)
            >>> aligned1, aligned2 = model._align_distributions(
            ...     t1, t2, ['cat', 'dog'], ['dog', 'bird']
            ... )
            >>> len(aligned1) == len(aligned2)  # Same length after alignment
            True
            >>> np.allclose(aligned1.sum(), 1.0)  # Still normalized
            True
            >>> np.allclose(aligned2.sum(), 1.0)
            True

        """
        # Create unified vocabulary and mapping
        unified_vocab: set[str] = set(vocab1) | set(vocab2)
        vocab1_idx = {word: idx for idx, word in enumerate(vocab1)}
        vocab2_idx = {word: idx for idx, word in enumerate(vocab2)}

        # Create aligned distributions
        dist1_aligned = np.zeros(len(unified_vocab))
        dist2_aligned = np.zeros(len(unified_vocab))

        # Fill in values using original distributions
        for idx, word in enumerate(sorted(unified_vocab)):
            if word in vocab1_idx:
                dist1_aligned[idx] = topic1.distribution[vocab1_idx[word]]
            if word in vocab2_idx:
                dist2_aligned[idx] = topic2.distribution[vocab2_idx[word]]

        # Renormalize if necessary
        if dist1_aligned.sum() > 0:
            dist1_aligned /= dist1_aligned.sum()
        if dist2_aligned.sum() > 0:
            dist2_aligned /= dist2_aligned.sum()

        return dist1_aligned, dist2_aligned

    def _compute_topic_similarity(self, topic1: Topic, topic2: Topic) -> float:
        """Compute cosine similarity between topic distributions.

        Args:
        ----
            topic1: First topic
            topic2: Second topic

        Returns:
        -------
            Cosine similarity between the topics

        Examples:
        --------
            >>> import numpy as np
            >>> model = TopicModel()
            >>> dist1 = np.array([1, 0])
            >>> dist2 = np.array([0, 1])
            >>> t1 = Topic(1, [('cat', 1.0)], dist1)
            >>> t2 = Topic(2, [('dog', 1.0)], dist2)
            >>> sim = model._compute_topic_similarity(t1, t2)
            >>> round(sim, 1)
            0.0

        """
        # Align distributions before computing similarity
        dist1_aligned, dist2_aligned = self._align_distributions(
            topic1, topic2, self.vocabulary, self.vocabulary
        )

        # Handle edge cases: if either vector is all zeros, similarity is 0
        if np.all(dist1_aligned == 0) or np.all(dist2_aligned == 0):
            return 0.0

        # Calculate cosine similarity, handling potential numerical issues
        try:
            similarity = 1 - cosine(dist1_aligned, dist2_aligned)
            # Handle edge case where cosine similarity is slightly outside [-1, 1]
            if np.isnan(similarity):
                return 0.0
            return float(np.clip(similarity, -1.0, 1.0))
        except Exception:
            return 0.0

    def align_with(self, other: "TopicModel") -> list[AlignedTopic]:
        """Align topics with another model using Hungarian algorithm.

        Args:
        ----
            other: Another fitted TopicModel

        Returns:
        -------
            List of aligned topic pairs sorted by similarity

        Raises:
        ------
            ValueError: If either model is not fitted

        Examples:
        --------
            >>> import numpy as np
            >>> from scipy.sparse import csr_matrix
            >>> model1 = TopicModel(n_topics=2)
            >>> model2 = TopicModel(n_topics=2)
            >>> ppmi = csr_matrix([[1, 0], [0, 1]])
            >>> model1.fit(ppmi, ['word1', 'word2'])
            >>> model2.fit(ppmi, ['word1', 'word2'])
            >>> aligned = model1.align_with(model2)
            >>> len(aligned) > 0
            True
            >>> isinstance(aligned[0], AlignedTopic)
            True

        """
        if not self.topics or not other.topics:
            raise ValueError("Both models must be fit before alignment")

        # Compute cost matrix using aligned vocabularies
        cost_matrix = np.zeros((self.n_topics, other.n_topics))
        for i, topic1 in enumerate(self.topics):
            for j, topic2 in enumerate(other.topics):
                dist1_aligned, dist2_aligned = self._align_distributions(
                    topic1, topic2, self.vocabulary, other.vocabulary
                )
                similarity = 1 - cosine(dist1_aligned, dist2_aligned)
                cost_matrix[i, j] = 1 - similarity

        # Find optimal matching
        source_indices, target_indices = linear_sum_assignment(cost_matrix)

        # Create aligned topic pairs
        aligned_topics = []
        for source_idx, target_idx in zip(source_indices, target_indices, strict=False):
            similarity = 1 - cost_matrix[source_idx, target_idx]

            if similarity >= self.min_similarity:
                aligned_topics.append(
                    AlignedTopic(
                        source_topic=self.topics[source_idx],
                        target_topic=other.topics[target_idx],
                        similarity=float(similarity),
                    )
                )

        return sorted(aligned_topics, key=lambda x: x.similarity, reverse=True)

    def print_topics(self, top_n: int = 10) -> None:
        """Print top words for each topic.

        Args:
        ----
            top_n: Number of top words to print per topic

        Examples:
        --------
            >>> from scipy.sparse import csr_matrix
            >>> model = TopicModel(n_topics=1)
            >>> ppmi = csr_matrix([[1, 0], [0, 1]])
            >>> model.fit(ppmi, ['word1', 'word2'])
            >>> model.print_topics(top_n=2)  # doctest: +ELLIPSIS
            <BLANKLINE>
            Topic 0:
              word...: 1.0000
              word...: 0.0000

        """
        if not self.topics:
            print("No topics available. Model needs to be fit first.")
            return

        for topic in self.topics:
            print(f"\nTopic {topic.id}:")
            n_words = min(top_n, len(topic.words))
            for word, weight in topic.words[:n_words]:
                print(f"  {word}: {weight:.4f}")
