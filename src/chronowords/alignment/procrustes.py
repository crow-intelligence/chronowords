"""Procrustes alignment for comparing word embeddings from different time periods."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes


@dataclass
class AlignmentMetrics:
    """Container for alignment quality metrics.

    Attributes:
        average_cosine_similarity: Mean cosine similarity between aligned word
            pairs, in the range [-1, 1] (near 1.0 for a good alignment).
        num_aligned_words: Number of anchor words successfully aligned.
        alignment_error: Frobenius norm of the residual between the rotated
            source and target anchor matrices (>= 0).

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


class ProcrustesAligner:
    """Aligns word embeddings from different time periods using Procrustes analysis.

    Finds the optimal orthogonal transformation that maps a source embedding
    space onto a target space while preserving distances, using shared
    vocabulary words as anchors. Must be :meth:`fit` before :meth:`transform`
    or :meth:`get_word_similarity` can be used.

    Example:
        >>> import numpy as np
        >>> aligner = ProcrustesAligner()
        >>> vocab = ["word1", "word2"]
        >>> emb_1800 = np.array([[1.0, 0.0], [0.0, 1.0]])
        >>> emb_1850 = np.array([[0.0, 1.0], [-1.0, 0.0]])
        >>> _ = aligner.fit(emb_1800, emb_1850, vocab, vocab)
        >>> aligned = aligner.transform(emb_1800)

    """

    def __init__(
        self, min_freq_rank: int | None = None, max_freq_rank: int | None = 1000
    ):
        """Initialize the aligner.

        Args:
            min_freq_rank: Lower bound (inclusive) of the frequency-rank slice
                used to select anchor words. ``None`` means "from the start".
            max_freq_rank: Upper bound (exclusive) of the frequency-rank slice.
                ``None`` means "to the end".

        Note:
            Both arguments are used directly as list-slice bounds on the
            vocabularies in :meth:`find_common_words`, which are assumed to be
            ordered by descending frequency. They are not validated; a
            ``min_freq_rank`` greater than ``max_freq_rank`` yields an empty
            anchor set and causes :meth:`fit` to raise ``ValueError``.

        Examples:
            >>> aligner = ProcrustesAligner(min_freq_rank=0, max_freq_rank=10)
            >>> aligner.min_freq_rank
            0
            >>> aligner.max_freq_rank
            10

        """
        self.min_freq_rank = min_freq_rank
        self.max_freq_rank = max_freq_rank
        self.orthogonal_matrix: np.ndarray | None = None
        self.source_words: list[str] = []
        self.target_words: list[str] = []
        self.anchors: dict[str, tuple[int, int]] = {}

    def find_common_words(
        self, source_vocab: list[str], target_vocab: list[str]
    ) -> list[str]:
        """Find common words between source and target vocabularies.

        Slices each vocabulary to the ``[min_freq_rank:max_freq_rank]`` rank
        window and returns the intersection, providing stable anchor words for
        alignment.

        Args:
            source_vocab: Source vocabulary, assumed ordered by descending
                frequency.
            target_vocab: Target vocabulary, assumed ordered by descending
                frequency.

        Returns:
            The common words within the rank window, sorted alphabetically.
            Empty if the windows do not overlap.

        Examples:
            >>> aligner = ProcrustesAligner(min_freq_rank=0, max_freq_rank=2)
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
        source_vocab: list[str],
        target_vocab: list[str],
        anchor_words: list[str] | None = None,
    ) -> AlignmentMetrics:
        """Learn the orthogonal transformation matrix using Procrustes analysis.

        Selects anchor words, L2-normalises their source and target vectors,
        and solves for the orthogonal matrix that best maps source anchors onto
        target anchors. Sets ``orthogonal_matrix``, ``source_words``,
        ``target_words`` and ``anchors``.

        Args:
            source_embeddings: Source-space embeddings, row-indexed by
                ``source_vocab``.
            target_embeddings: Target-space embeddings, row-indexed by
                ``target_vocab``.
            source_vocab: Vocabulary list for ``source_embeddings``.
            target_vocab: Vocabulary list for ``target_embeddings``.
            anchor_words: Specific words to align on. If ``None``, common words
                filtered by frequency rank (:meth:`find_common_words`) are used.

        Returns:
            :class:`AlignmentMetrics` describing alignment quality.

        Raises:
            ValueError: If no anchor words are available (no common words, or
                an empty ``anchor_words``), or if every candidate anchor is
                dropped because it is missing from one vocabulary or has a
                near-zero vector in either space.

        Note:
            Preconditions:
                - Each embedding matrix must have a row for every entry in its
                  vocabulary; a vocab/embedding length mismatch surfaces as an
                  ``IndexError`` while gathering anchor vectors (not checked).
                - The two embedding spaces must share dimensionality, otherwise
                  :func:`scipy.linalg.orthogonal_procrustes` raises (not
                  caught).
                - Anchor words whose source or target vector is effectively
                  zero are silently skipped.

        Examples:
            >>> import numpy as np
            >>> aligner = ProcrustesAligner()
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

        source_idx_map = {w: i for i, w in enumerate(source_vocab)}
        target_idx_map = {w: i for i, w in enumerate(target_vocab)}

        for word in anchor_words:
            source_idx = source_idx_map.get(word)
            target_idx = target_idx_map.get(word)

            if source_idx is None or target_idx is None:
                continue

            source_vec = source_embeddings[source_idx]
            target_vec = target_embeddings[target_idx]

            # Check for zero vectors
            if np.all(np.abs(source_vec) < 1e-10) or np.all(np.abs(target_vec) < 1e-10):
                continue

            self.anchors[word] = (source_idx, target_idx)
            source_vectors.append(source_vec)
            target_vectors.append(target_vec)

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
            embeddings: Embeddings to transform, with the same dimensionality
                as the space the aligner was fit on.

        Returns:
            The embeddings rotated into the target space (``embeddings @
            orthogonal_matrix``).

        Raises:
            ValueError: If the aligner has not been fit yet
                (``orthogonal_matrix is None``).

        Note:
            The column count of ``embeddings`` must match ``orthogonal_matrix``;
            a mismatch raises ``ValueError`` from the matrix multiply (not
            checked explicitly).

        Examples:
            >>> import numpy as np
            >>> aligner = ProcrustesAligner()
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

        Looks up ``word`` in both vocabularies, normalises its source and
        target vectors, rotates the source vector into the target space, and
        returns the cosine similarity.

        Args:
            word: Word to compare. Must be present in both ``source_words`` and
                ``target_words`` (populated by :meth:`fit`).
            source_emb: Source embeddings, row-indexed by ``source_words``.
            target_emb: Target embeddings, row-indexed by ``target_words``.

        Returns:
            Cosine similarity in [-1, 1] between the aligned source vector and
            the target vector; higher means more similar usage across periods.
            ``None`` if ``word`` is absent from either vocabulary.

        Raises:
            AttributeError: If the aligner has not been fit
                (``orthogonal_matrix is None``) — surfaces from the matrix
                multiply, which is *not* guarded here unlike :meth:`transform`.

        Note:
            The source/target vectors are divided by their L2 norm with no
            zero-norm guard. A zero vector produces ``nan``/``inf`` entries and
            a silent ``RuntimeWarning`` rather than ``None`` or an exception —
            see the project pre-mortem.

        Examples:
            >>> import numpy as np
            >>> aligner = ProcrustesAligner()
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
        """Save the aligner state to disk via pickle.

        Persists ``orthogonal_matrix``, ``source_words``, ``target_words``,
        ``anchors`` and the frequency-rank bounds so a later :meth:`load` can
        restore a fitted aligner.

        Args:
            path: File path to write the pickled state to.

        Raises:
            OSError: If ``path`` cannot be opened for writing (propagated from
                ``open``).

        Note:
            Saving an unfitted aligner is allowed and writes
            ``orthogonal_matrix=None``; reloading it yields an aligner that
            still needs :meth:`fit`.

        """
        data = {
            "orthogonal_matrix": self.orthogonal_matrix,
            "source_words": self.source_words,
            "target_words": self.target_words,
            "anchors": self.anchors,
            "min_freq_rank": self.min_freq_rank,
            "max_freq_rank": self.max_freq_rank,
        }
        with Path.open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Path) -> None:
        """Load aligner state from a pickle written by :meth:`save`.

        Overwrites ``orthogonal_matrix``, ``source_words``, ``target_words``,
        ``anchors`` and the frequency-rank bounds with the saved values.

        Args:
            path: File path to read the pickled state from.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            KeyError: If the pickle is missing an expected key (e.g. it was not
                written by :meth:`save`).

        Warning:
            This method unpickles ``path``. Unpickling executes arbitrary code
            embedded in the file, so only load aligner files you trust.

        """
        with Path.open(path, "rb") as f:
            data = pickle.load(f)
        self.orthogonal_matrix = data["orthogonal_matrix"]
        self.source_words = data["source_words"]
        self.target_words = data["target_words"]
        self.anchors = data["anchors"]
        self.min_freq_rank = data["min_freq_rank"]
        self.max_freq_rank = data["max_freq_rank"]
