"""SVD-based word embedding implementation with memory-efficient counting."""

import logging
import pickle
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nltk.util import skipgrams
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine

from ..utils.count_skipgrams import PPMIComputer  # ty: ignore
from ..utils.probabilistic_counter import CountMinSketch


logger = logging.getLogger(__name__)


@dataclass
class WordSimilarity:
    """Container for word similarity results.

    Attributes:
        word: The similar word found.
        similarity: Cosine similarity score, in the range [-1, 1].

    Examples:
        >>> sim = WordSimilarity("cat", 0.85)
        >>> sim.word
        'cat'
        >>> sim.similarity
        0.85
        >>> round(sim.similarity, 2)
        0.85

    """

    word: str
    similarity: float


@dataclass
class AnalogyResult:
    """Container for word analogy results.

    Attributes:
        words: List of words matching the analogy, ordered by descending
            similarity.
        similarities: Similarity score for each word. Parallel to ``words``;
            the producer (:meth:`SVDAlgebra.analogy`) guarantees
            ``len(words) == len(similarities)``, but this dataclass does not
            enforce it.

    Examples:
        >>> result = AnalogyResult(["queen"], [0.76])
        >>> result.words
        ['queen']
        >>> result.similarities
        [0.76]
        >>> len(result.words) == len(result.similarities)
        True

    """

    words: list[str]
    similarities: list[float]


class SVDAlgebra:
    """Implements word vector algebra using SVD-based embeddings.

    Builds PPMI-weighted word embeddings from a corpus via truncated SVD,
    using a Count-Min Sketch for memory-efficient counting and a
    Cython-optimized PPMI kernel.

    The model has a two-phase lifecycle: it is unusable until :meth:`train`
    (or :meth:`load_model`) has populated ``vocabulary`` and ``embeddings``.
    Query methods (:meth:`most_similar`, :meth:`distance`, :meth:`analogy`,
    :meth:`get_vector`) return an empty result or ``None`` if called before
    training rather than raising.

    Examples:
        >>> model = SVDAlgebra(n_components=2)
        >>> model.n_components
        2
        >>> model.window_size
        5

    """

    def __init__(
        self,
        n_components: int = 300,
        window_size: int = 5,
        min_word_length: int = 3,
        cms_width: int = 1_000_000,
        cms_depth: int = 5,
    ) -> None:
        """Initialize SVDAlgebra.

        Args:
            n_components: Number of SVD components (embedding dimensions) to
                keep. During :meth:`train` this is clamped to
                ``min(n_components, min(matrix.shape) - 1)`` and floored at 1,
                so a value larger than the vocabulary yields fewer dimensions.
            window_size: Skip-gram context window size.
            min_word_length: Minimum word length to consider; shorter tokens
                are dropped during training.
            cms_width: Width of the Count-Min Sketch tables. Larger values
                reduce count collisions at the cost of memory
                (``cms_width * cms_depth * 4`` bytes per sketch).
            cms_depth: Number of hash functions (rows) in the Count-Min Sketch.

        Note:
            Constructor arguments are stored without validation. The integer
            arguments are assumed positive; non-positive values surface later
            as errors from NumPy allocation (``cms_width``/``cms_depth``) or
            from :func:`scipy.sparse.linalg.svds` (``n_components``) during
            :meth:`train`, not here.

        Examples:
            >>> model = SVDAlgebra(n_components=100)
            >>> model.n_components
            100
            >>> model.embeddings is None
            True

        """
        self.n_components = n_components
        self.window_size = window_size
        self.min_word_length = min_word_length
        self.cms_width = cms_width
        self.cms_depth = cms_depth

        self.vocabulary: list[str] = []
        self._vocab_index: dict[str, int] = {}
        self.embeddings: NDArray[np.float64] | None = None
        self._ppmi_sparse: csr_matrix | None = None
        # Keep M_dense for backward compatibility with notebook code
        self.M_dense: np.ndarray | None = None

    def _build_vocab_index(self) -> None:
        """Build dict index for O(1) vocabulary lookups."""
        self._vocab_index = {w: i for i, w in enumerate(self.vocabulary)}

    def train(self, corpus: Generator[str, None, None]) -> None:
        """Train the model on a text corpus.

        Counts words and skip-grams with a Count-Min Sketch, builds the
        vocabulary from words above a fixed frequency threshold, computes a
        sparse PPMI matrix, and factorises it with truncated SVD. On success,
        populates ``vocabulary``, ``embeddings``, ``_ppmi_sparse`` and
        ``M_dense``.

        Args:
            corpus: Iterable of text lines (e.g. a generator). Each line is
                split on whitespace; tokens shorter than ``min_word_length``
                are discarded.

        Raises:
            ValueError: If no word clears the 0.01% heavy-hitter frequency
                threshold, i.e. the corpus is empty or too small to build a
                vocabulary (explicit check after counting).

        Note:
            Precondition: ``corpus`` is consumed exactly once; passing a
            single-use iterator that has already been exhausted produces an
            empty vocabulary and raises ``ValueError``.

            Words are frequency-filtered by a Count-Min Sketch, which can
            *overestimate* counts on hash collisions, so a rare word may
            occasionally be admitted to the vocabulary.

            Silenced failure: if sparse :func:`~scipy.sparse.linalg.svds`
            raises (common for tiny or degenerate matrices), the error is
            swallowed and the code falls back to a dense SVD of the matrix
            *with Gaussian noise (sigma 1e-10) added*. The fallback path is
            invisible to the caller and yields slightly different,
            non-deterministic embeddings from the sparse path.

        Examples:
            >>> model = SVDAlgebra(n_components=2)
            >>> text = ["the cat sat", "the dog ran"]
            >>> model.train(line for line in text)
            >>> len(model.vocabulary) > 0
            True
            >>> model.embeddings is not None
            True

        """
        word_counter = CountMinSketch(self.cms_width, self.cms_depth)
        skipgram_counter = CountMinSketch(
            self.cms_width, self.cms_depth, track_keys=False
        )

        logger.info("Counting words and skipgrams...")

        for line in corpus:
            words = [w for w in line.split() if len(w) >= self.min_word_length]
            for word in words:
                word_counter.update(word)
            skips = skipgrams(words, 2, self.window_size)
            for w1, w2 in skips:
                skipgram_counter.update(f"{w1}#{w2}")

        logger.info("Building vocabulary...")
        vocab_candidates = word_counter.get_heavy_hitters(0.0001)
        self.vocabulary = [word for word, count in vocab_candidates]

        if not self.vocabulary:
            raise ValueError("No words found meeting minimum frequency threshold")

        self._build_vocab_index()

        word_counts, word_seeds, width = word_counter.arrays
        skipgram_counts, _, _ = skipgram_counter.arrays

        computer = PPMIComputer(
            skipgram_counts=skipgram_counts,
            word_counts=word_counts,
            vocabulary=self.vocabulary,
            seeds=word_seeds,
            width=width,
            skip_total=float(skipgram_counter.total),
            word_total=float(word_counter.total),
            shift=1.0,
            alpha=0.75,
        )
        M = computer.compute_ppmi_matrix_with_sketch()
        self._ppmi_sparse = M

        # Use truncated SVD directly on sparse matrix
        k = min(self.n_components, min(M.shape) - 1)
        if k < 1:
            k = 1

        M_float = M.astype(np.float64)

        try:
            U, S, _Vt = svds(M_float, k=k)
        except Exception:
            # Fallback to dense SVD for very small/degenerate matrices
            M_dense = M.toarray().astype(np.float64)
            M_dense += np.random.normal(0, 1e-10, M_dense.shape)
            U, S, _Vt = np.linalg.svd(M_dense, full_matrices=False)
            U = U[:, :k]
            S = S[:k]

        # svds returns singular values in ascending order; reverse them
        idx = np.argsort(S)[::-1]
        U = U[:, idx].astype(np.float64)
        S = S[idx].astype(np.float64)

        self.embeddings = (U * np.sqrt(S)).astype(np.float64)

        # Keep M_dense for backward compatibility (lazy — only computed when accessed)
        self.M_dense = M.toarray().astype(np.float64)

        logger.info("Final embeddings shape: %s", self.embeddings.shape)
        logger.info("Embeddings non-zeros: %d", np.count_nonzero(self.embeddings))
        norms = np.linalg.norm(self.embeddings, axis=1)
        logger.info("Min norm: %f", float(np.min(norms)))

    def save_model(self, path: str | Path) -> None:
        """Save model embeddings and vocabulary to disk.

        Writes up to four files into ``path``: ``ppmi.npz`` (sparse PPMI),
        ``ppmi.npy`` (dense PPMI), ``embeddings.npy``, and ``vocabulary.pkl``.

        Args:
            path: Directory to write model files to. Created (including
                parents) if it does not exist.

        Raises:
            OSError: If ``path`` cannot be created or written (e.g. permission
                denied, read-only filesystem) — propagated from ``mkdir`` /
                ``np.save`` / ``open``.

        Note:
            Only attributes that are currently set are written: each of the
            PPMI matrices and ``embeddings`` is saved only when not
            ``None``. Calling this before :meth:`train` therefore writes a
            ``vocabulary.pkl`` holding an empty list and no ``embeddings.npy``,
            which makes the resulting directory unloadable by
            :meth:`load_model`. No warning is emitted for a partial save.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> model = SVDAlgebra(n_components=2)
            >>> model.vocabulary = ["cat", "dog"]
            >>> model.embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> model.M_dense = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     model.save_model(tmpdir)
            ...     saved_files = list(Path(tmpdir).glob("*"))
            ...     len(saved_files) > 0
            True

        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._ppmi_sparse is not None:
            save_npz(path / "ppmi.npz", self._ppmi_sparse)
        if self.M_dense is not None:
            np.save(path / "ppmi.npy", self.M_dense)
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)
        with Path.open(path / "vocabulary.pkl", "wb") as f:
            pickle.dump(self.vocabulary, f)

    def load_model(self, path: str | Path) -> None:
        """Load model embeddings and vocabulary from disk.

        Restores ``embeddings``, ``vocabulary`` and (if present) ``M_dense``
        from a directory previously written by :meth:`save_model`, and rebuilds
        the vocabulary index.

        Args:
            path: Directory containing the saved model files.

        Raises:
            ValueError: If ``path`` is not an existing directory (explicit
                check).
            FileNotFoundError: If ``embeddings.npy`` or ``vocabulary.pkl`` is
                missing from ``path`` (implicit, from ``np.load`` / ``open``) —
                e.g. when the directory came from a pre-training
                :meth:`save_model`.
            pickle.UnpicklingError: If ``vocabulary.pkl`` is corrupt.

        Warning:
            This method unpickles ``vocabulary.pkl``. Unpickling executes
            arbitrary code embedded in the file, so only load model directories
            from sources you trust.

        Note:
            ``ppmi.npy`` is restored only if present; ``ppmi.npz`` (the sparse
            PPMI matrix) is not reloaded, so ``_ppmi_sparse`` stays ``None``
            after loading.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> model = SVDAlgebra(n_components=2)
            >>> model.vocabulary = ["cat", "dog"]
            >>> model.embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> model.M_dense = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     model.save_model(tmpdir)
            ...     new_model = SVDAlgebra(n_components=2)
            ...     new_model.load_model(tmpdir)
            ...     len(new_model.vocabulary) == len(model.vocabulary)
            True

        """
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Directory not found: {path}")

        ppmi_npy = path / "ppmi.npy"
        if ppmi_npy.exists():
            self.M_dense = np.load(ppmi_npy)
        self.embeddings = np.load(path / "embeddings.npy")
        with Path.open(path / "vocabulary.pkl", "rb") as f:
            self.vocabulary = pickle.load(f)
        self._build_vocab_index()

    def get_vector(self, word: str) -> NDArray[np.float64] | None:
        """Get the embedding vector for a word.

        Args:
            word: Input word to look up.

        Returns:
            The word's embedding vector, or ``None`` if the model is untrained
            (``embeddings is None``) or ``word`` is not in the vocabulary. A
            ``None`` return does not distinguish these two cases.

        Examples:
            >>> model = SVDAlgebra(n_components=2)
            >>> model.vocabulary = ["cat", "dog"]
            >>> model._build_vocab_index()
            >>> model.embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> vector = model.get_vector("cat")
            >>> vector is not None
            True
            >>> vector.shape
            (2,)
            >>> model.get_vector("unknown") is None
            True

        """
        if self.embeddings is None:
            return None
        idx = self._vocab_index.get(word)
        if idx is None:
            return None
        return self.embeddings[idx]

    def most_similar(self, word: str, n: int = 10) -> list[WordSimilarity]:
        """Find the n most similar words by cosine similarity.

        Args:
            word: Query word.
            n: Maximum number of similar words to return.

        Returns:
            Up to ``n`` :class:`WordSimilarity` results, sorted by descending
            cosine similarity, excluding ``word`` itself. Empty if the model is
            untrained, ``word`` is unknown, or ``word``'s vector has
            effectively zero norm — these cases are indistinguishable from
            "no similar words found".

        Note:
            Silenced numerical issues: vocabulary norms are floored at 1e-10 to
            avoid division by zero, and any ``NaN`` similarity (from zero-norm
            embeddings) is replaced with -1.0 and then filtered out, so
            degenerate vectors are silently excluded rather than reported.

        Examples:
            >>> model = SVDAlgebra(n_components=2)
            >>> model.vocabulary = ["cat", "dog", "fish"]
            >>> model._build_vocab_index()
            >>> model.embeddings = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
            >>> results = model.most_similar("cat", n=2)
            >>> len(results)
            2
            >>> results[0].word
            'dog'
            >>> round(results[0].similarity, 2)
            0.97

        """
        vector = self.get_vector(word)
        if vector is None or self.embeddings is None:
            return []

        query_norm = float(np.linalg.norm(vector))
        if query_norm < 1e-10:
            return []

        vocab_norms = np.linalg.norm(self.embeddings, axis=1)
        vocab_norms = np.maximum(vocab_norms, 1e-10)

        similarities = self.embeddings @ vector
        similarities = similarities / (vocab_norms * query_norm)
        similarities = np.clip(similarities, -1.0, 1.0)
        similarities = np.nan_to_num(similarities, nan=-1.0)

        results: list[WordSimilarity] = []
        for idx in np.argsort(similarities)[::-1]:
            candidate = self.vocabulary[idx]
            similarity = float(similarities[idx])
            if candidate != word and similarity > -1:
                results.append(WordSimilarity(candidate, similarity))
                if len(results) == n:
                    break

        return results

    def distance(self, word1: str, word2: str) -> float | None:
        """Calculate cosine distance between two words.

        Args:
            word1: First word.
            word2: Second word.

        Returns:
            Cosine distance in the range [0, 1], or ``None`` if either word is
            unknown (or the model is untrained) or either vector has
            effectively zero norm. A ``None`` return does not distinguish these
            cases.

        Examples:
            >>> model = SVDAlgebra(n_components=2)
            >>> model.vocabulary = ["cat", "dog"]
            >>> model._build_vocab_index()
            >>> model.embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> round(model.distance("cat", "dog"), 2)
            1.0
            >>> model.distance("cat", "unknown") is None
            True

        """
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        if vec1 is None or vec2 is None:
            return None

        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))
        if norm1 < 1e-10 or norm2 < 1e-10:
            return None

        dist = float(cosine(vec1, vec2))
        return min(dist, 1.0)

    def analogy(
        self, positive: list[str], negative: str, n: int = 3
    ) -> AnalogyResult | None:
        """Solve word analogies (e.g., king - man + woman ≈ queen).

        Computes the target vector ``negative - positive[0] + positive[1]``,
        normalises it, and returns the vocabulary words closest to it.

        Args:
            positive: Exactly two positive words for the analogy.
            negative: The negative word.
            n: Maximum number of results to return.

        Returns:
            An :class:`AnalogyResult` (words sorted by descending similarity,
            excluding the three input words and tokens shorter than
            ``min_word_length``), or ``None``. Several distinct conditions all
            collapse to ``None``: ``positive`` does not have exactly two
            elements, the model is untrained, any input word is unknown, the
            target vector is degenerate (near-zero norm), or no candidate
            remains after filtering.

        Note:
            Silenced numerical issues: as in :meth:`most_similar`, vocabulary
            norms are floored at 1e-10 and ``NaN`` similarities are mapped to
            -1.0 and filtered out.

        Examples:
            >>> model = SVDAlgebra(n_components=2)
            >>> model.vocabulary = ["king", "man", "woman", "queen"]
            >>> model._build_vocab_index()
            >>> emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
            >>> model.embeddings = emb
            >>> result = model.analogy(["king", "man"], "woman")
            >>> result is not None
            True
            >>> "queen" in result.words  # doctest: +SKIP
            True

        """
        if len(positive) != 2 or self.embeddings is None:
            return None

        vectors: list[NDArray[np.float64]] = []
        for word in positive + [negative]:
            vec = self.get_vector(word)
            if vec is None:
                return None
            vectors.append(vec)

        vec1, vec2, vec3 = vectors[0], vectors[1], vectors[2]

        target = vec3 - vec1 + vec2
        target_norm = float(np.linalg.norm(target))
        if target_norm < 1e-10:
            return None

        target = target / target_norm

        if self.embeddings is None:
            return None

        vocab_norms = np.linalg.norm(self.embeddings, axis=1)
        vocab_norms = np.maximum(vocab_norms, 1e-10)

        similarities = self.embeddings @ target
        similarities = similarities / vocab_norms
        similarities = np.clip(similarities, -1.0, 1.0)
        similarities = np.nan_to_num(similarities, nan=-1.0)

        results: list[str] = []
        similarity_scores: list[float] = []
        seen = set(positive + [negative])

        for idx in np.argsort(similarities)[::-1]:
            word = self.vocabulary[idx]
            similarity = float(similarities[idx])
            if (
                word not in seen
                and len(word) >= self.min_word_length
                and similarity > -1
            ):
                results.append(word)
                similarity_scores.append(similarity)
                if len(results) == n:
                    break

        return AnalogyResult(results, similarity_scores) if results else None
