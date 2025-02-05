"""
Procrustes alignment for comparing word embeddings from different time periods.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import orthogonal_procrustes


@dataclass
class AlignmentMetrics:
    """Container for alignment quality metrics.

    Fields:
        average_cosine_similarity: Mean cosine similarity between aligned word pairs
        num_aligned_words: Number of words successfully aligned
        alignment_error: Frobenius norm of the difference between aligned matrices

    Examples:
        >>> metrics = AlignmentMetrics(0.85, 1000, 0.15)
        >>> metrics.average_cosine_similarity
        0.85
        >>> metrics.num_aligned_words
        1000
        >>> metrics.alignment_error
        0.15
    """

    average_cosine_similarity: float
    num_aligned_words: int
    alignment_error: float


class ProcustesAligner:
    """Aligns word embeddings from different time periods using Procrustes analysis.

    Finds optimal orthogonal transformation to align embeddings while preserving distances.

    Example:
        aligner = ProcustesAligner()
        metrics = aligner.fit(
            embeddings_1800, embeddings_1850,
            vocab_1800, vocab_1850
        )
        aligned_embeddings = aligner.transform(embeddings_1800)
    """

    def __init__(
        self, min_freq_rank: Optional[int] = None, max_freq_rank: Optional[int] = 1000
    ):
        """
        Initialize the aligner.

        Args:
            min_freq_rank: Minimum frequency rank for anchor words
            max_freq_rank: Maximum frequency rank for anchor words

        Examples:
            >>> aligner = ProcustesAligner(min_freq_rank=0, max_freq_rank=10)
            >>> aligner.min_freq_rank
            0
            >>> aligner.max_freq_rank
            10
        """
        self.min_freq_rank = min_freq_rank
        self.max_freq_rank = max_freq_rank
        self.orthogonal_matrix: Optional[np.ndarray] = None
        self.source_words: List[str] = []
        self.target_words: List[str] = []
        self.anchors: Dict[str, Tuple[int, int]] = {}

    def find_common_words(
        self, source_vocab: List[str], target_vocab: List[str]
    ) -> List[str]:
        """Find common words between source and target vocabularies.

        Uses frequency rank filtering (min_freq_rank to max_freq_rank) to select
        stable anchor words for alignment.

        Returns:
            List of common words sorted alphabetically

        Examples:
            >>> aligner = ProcustesAligner(min_freq_rank=0, max_freq_rank=2)
            >>> source = ['the', 'in', 'a', 'rare']
            >>> target = ['in', 'the', 'new', 'a']
            >>> aligner.find_common_words(source, target)
            ['in', 'the']
        """
        source_set = set(source_vocab[self.min_freq_rank : self.max_freq_rank])
        target_set = set(target_vocab[self.min_freq_rank : self.max_freq_rank])
        return sorted(source_set.intersection(target_set))

    def fit(
        self,
        source_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
        source_vocab: List[str],
        target_vocab: List[str],
        anchor_words: Optional[List[str]] = None,
    ) -> AlignmentMetrics:
        """Learn the orthogonal transformation matrix using Procrustes analysis.

        Args:
            source_embeddings: Source space word embeddings
            target_embeddings: Target space word embeddings
            source_vocab: Vocabulary list for source embeddings
            target_vocab: Vocabulary list for target embeddings
            anchor_words: Optional list of specific words to use for alignment
                         If None, uses common words filtered by frequency rank

        Returns:
            AlignmentMetrics containing quality measures of the alignment

        Examples:
            >>> import numpy as np
            >>> aligner = ProcustesAligner()
            >>> source_emb = np.array([[1., 0.], [0., 1.]])
            >>> target_emb = np.array([[0., 1.], [-1., 0.]])  # 90 degree rotation
            >>> vocab = ['word1', 'word2']
            >>> metrics = aligner.fit(source_emb, target_emb, vocab, vocab, ['word1', 'word2'])
            >>> metrics.num_aligned_words
            2
            >>> round(metrics.average_cosine_similarity, 2)
            1.0
        """
        self.source_words = source_vocab
        self.target_words = target_vocab

        # Get words to use for alignment
        if anchor_words is None:
            anchor_words = self.find_common_words(source_vocab, target_vocab)

        if not anchor_words:
            raise ValueError("No common words found for alignment")

        # Build anchor word indices and matrices
        self.anchors.clear()
        source_vectors = []
        target_vectors = []

        for word in anchor_words:
            try:
                source_idx = source_vocab.index(word)
                target_idx = target_vocab.index(word)

                # Get vectors
                source_vec = source_embeddings[source_idx]
                target_vec = target_embeddings[target_idx]

                # Check for zero vectors
                if np.all(np.abs(source_vec) < 1e-10) or np.all(
                    np.abs(target_vec) < 1e-10
                ):
                    continue

                self.anchors[word] = (source_idx, target_idx)
                source_vectors.append(source_vec)
                target_vectors.append(target_vec)
            except ValueError:
                continue

        if not self.anchors:
            raise ValueError("No valid anchor words found")

        # Convert to arrays
        source_matrix = np.vstack(source_vectors)
        target_matrix = np.vstack(target_vectors)

        # Normalize matrices safely
        source_norms = np.linalg.norm(source_matrix, axis=1)
        target_norms = np.linalg.norm(target_matrix, axis=1)

        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        source_norms = np.maximum(source_norms, epsilon)
        target_norms = np.maximum(target_norms, epsilon)

        source_matrix = source_matrix / source_norms[:, np.newaxis]
        target_matrix = target_matrix / target_norms[:, np.newaxis]

        # Compute orthogonal transformation matrix
        self.orthogonal_matrix, error = orthogonal_procrustes(
            source_matrix, target_matrix
        )

        # Calculate alignment quality metrics
        aligned_source = source_matrix @ self.orthogonal_matrix
        cosine_sims = np.sum(aligned_source * target_matrix, axis=1)
        avg_similarity = float(np.mean(cosine_sims))

        return AlignmentMetrics(
            average_cosine_similarity=avg_similarity,
            num_aligned_words=len(self.anchors),
            alignment_error=float(error),
        )

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply the learned transformation to align embeddings.

        Args:
            embeddings: Embeddings to transform

        Returns:
            Transformed embeddings in the target space

        Examples:
            >>> import numpy as np
            >>> aligner = ProcustesAligner()
            >>> # No need to set source_words/target_words since we're just testing transform
            >>> aligner.orthogonal_matrix = np.array([[0, 1], [-1, 0]])  # 90 degree rotation
            >>> embeddings = np.array([[1, 0], [0, 1]])
            >>> aligned = aligner.transform(embeddings)
            >>> np.allclose(aligned, np.array([[0, 1], [-1, 0]]))
            True
        """
        if self.orthogonal_matrix is None:
            raise ValueError("Aligner must be fit before transform")
        return embeddings @ self.orthogonal_matrix

    def get_word_similarity(
        self, word: str, source_emb: np.ndarray, target_emb: np.ndarray
    ):
        """Get similarity between word representations in source and target spaces.

        Args:
            word: Word to compare
            source_emb: Source embeddings
            target_emb: Target embeddings

        Returns:
            Cosine similarity [-1,1] between aligned vectors,
            higher values indicate more similar usage between periods.
            Returns None if word not found in either vocabulary.

        Examples:
            >>> import numpy as np
            >>> aligner = ProcustesAligner()
            >>> aligner.source_words = ['cat', 'dog']  # Set after initialization
            >>> aligner.target_words = ['cat', 'dog']  # Set after initialization
            >>> aligner.orthogonal_matrix = np.eye(2)
            >>> source_emb = np.array([[1., 0.], [0., 1.]])
            >>> target_emb = np.array([[1., 0.], [0., 1.]])
            >>> round(aligner.get_word_similarity('cat', source_emb, target_emb), 2)
            1.0
        """
        try:
            source_idx = self.source_words.index(word)
            target_idx = self.target_words.index(word)
        except ValueError:
            return None

        source_vec = source_emb[source_idx]
        target_vec = target_emb[target_idx]

        # Normalize vectors
        source_vec = source_vec / np.linalg.norm(source_vec)
        target_vec = target_vec / np.linalg.norm(target_vec)

        # Align source vector and compute similarity
        aligned_vec = source_vec @ self.orthogonal_matrix
        return float(np.dot(aligned_vec, target_vec))

    def save(self, path: Path) -> None:
        """Save the aligner state."""
        data = {
            "orthogonal_matrix": self.orthogonal_matrix,
            "source_words": self.source_words,
            "target_words": self.target_words,
            "anchors": self.anchors,
            "min_freq_rank": self.min_freq_rank,
            "max_freq_rank": self.max_freq_rank,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Path) -> None:
        """Load the aligner state."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.orthogonal_matrix = data["orthogonal_matrix"]
        self.source_words = data["source_words"]
        self.target_words = data["target_words"]
        self.anchors = data["anchors"]
        self.min_freq_rank = data["min_freq_rank"]
        self.max_freq_rank = data["max_freq_rank"]
