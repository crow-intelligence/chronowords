"""SVD-based word embedding implementation with memory-efficient counting."""

import json
import logging
import pickle
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse import load_npz
from scipy.sparse import save_npz
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine

from ..utils.cooccurrence import CooccurrenceCounts
from ..utils.cooccurrence import Corpus
from ..utils.cooccurrence import count_cooccurrences
from ..utils.cooccurrence import ppmi_matrix


logger = logging.getLogger(__name__)


def _version() -> str | None:
    try:
        return version("chronowords")
    except PackageNotFoundError:
        return None


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

    Builds PPMI-weighted word embeddings from a corpus via truncated SVD. Every
    word and skip-gram is counted exactly, in two passes, by the compiled core
    (:func:`~chronowords.utils.cooccurrence.count_cooccurrences`).

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
        *,
        vocab_size: int | None = 50_000,
        min_count: int = 100,
        shift: float = 1.0,
        alpha: float = 0.75,
        ppmi_min_count: int = 5,
        tokenizer: Literal["whitespace", "unicode"] = "whitespace",
        lowercase: bool = False,
        n_threads: int | None = None,
    ) -> None:
        """Initialize SVDAlgebra.

        Args:
            n_components: Number of SVD components (embedding dimensions) to
                keep. During :meth:`train` this is clamped to
                ``min(n_components, vocabulary size - 1)`` and floored at 1.
                Default 300.
            window_size: Skip-gram window; each token pairs with the next
                ``window_size + 1`` tokens. Default 5.
            min_word_length: Tokens shorter than this many code points are
                dropped before counting. Default 3.
            vocab_size: Keep at most this many words, most frequent first,
                ties alphabetical; ``None`` for no cap. Default 50,000.
            min_count: Keep only words occurring at least this many times.
                Default 100. Lower it for small corpora.
            shift: PMI shift; ``log(shift)`` is subtracted from every PMI.
                Default 1.0 (no shift).
            alpha: Context-distribution smoothing exponent, applied to the
                focus word's probability only. Default 0.75.
            ppmi_min_count: A word and a pair both need a count strictly
                greater than this to receive a PPMI entry. Default 5.
            tokenizer: ``"whitespace"`` (default; Python ``str.split()``) or
                ``"unicode"`` (UAX #29 words, punctuation dropped). See
                :func:`~chronowords.utils.cooccurrence.count_cooccurrences`.
            lowercase: Lowercase tokens before counting. Default ``False``.
            n_threads: Threads for counting file input; ``None`` (default)
                uses one per CPU core. Does not affect results.

        Raises:
            ValueError: If a parameter is out of range, or ``tokenizer`` is not
                one of its documented values.

        Examples:
            >>> model = SVDAlgebra(n_components=100)
            >>> model.n_components
            100
            >>> model.embeddings is None
            True
            >>> SVDAlgebra(tokenizer="bpe")
            Traceback (most recent call last):
            ...
            ValueError: tokenizer must be 'whitespace' or 'unicode', got 'bpe'

        """
        if tokenizer not in ("whitespace", "unicode"):
            raise ValueError(
                f"tokenizer must be 'whitespace' or 'unicode', got {tokenizer!r}"
            )
        for name, value, low in (
            ("n_components", n_components, 1),
            ("window_size", window_size, 0),
            ("min_word_length", min_word_length, 0),
            ("min_count", min_count, 1),
            ("ppmi_min_count", ppmi_min_count, 0),
        ):
            if value < low:
                raise ValueError(f"{name} must be >= {low}, got {value}")
        if n_threads is not None and n_threads < 1:
            raise ValueError(f"n_threads must be >= 1 or None, got {n_threads}")
        if vocab_size is not None and vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1 or None, got {vocab_size}")
        if shift <= 0:
            raise ValueError(f"shift must be > 0, got {shift}")

        self.n_components = n_components
        self.window_size = window_size
        self.min_word_length = min_word_length
        self.vocab_size = vocab_size
        self.min_count = min_count
        self.shift = shift
        self.alpha = alpha
        self.ppmi_min_count = ppmi_min_count
        self.tokenizer: Literal["whitespace", "unicode"] = tokenizer
        self.lowercase = lowercase
        self.n_threads = n_threads

        self.vocabulary: list[str] = []
        self._vocab_index: dict[str, int] = {}
        self.embeddings: NDArray[np.float64] | None = None
        self._ppmi_sparse: csr_matrix | None = None
        self.counts: CooccurrenceCounts | None = None
        self.training_record: dict[str, Any] | None = None

    @property
    def ppmi(self) -> csr_matrix | None:
        """The sparse PPMI matrix from the last :meth:`train`, or ``None``.

        Rows and columns follow ``vocabulary``. Restored by :meth:`load_model`
        when the model directory contains it.
        """
        return self._ppmi_sparse

    def _params(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "window_size": self.window_size,
            "min_word_length": self.min_word_length,
            "vocab_size": self.vocab_size,
            "min_count": self.min_count,
            "shift": self.shift,
            "alpha": self.alpha,
            "ppmi_min_count": self.ppmi_min_count,
            "tokenizer": self.tokenizer,
            "lowercase": self.lowercase,
            "n_threads": self.n_threads,
        }

    def _build_vocab_index(self) -> None:
        """Build dict index for O(1) vocabulary lookups."""
        self._vocab_index = {w: i for i, w in enumerate(self.vocabulary)}

    def train(self, corpus: Corpus) -> None:
        """Train the model on a text corpus.

        Counts words and skip-grams, keeps the vocabulary (the ``vocab_size``
        most frequent words with at least ``min_count`` occurrences, ties
        alphabetical), computes a sparse PPMI matrix and factorises it with
        truncated SVD. On success, populates ``vocabulary``, ``embeddings``,
        ``ppmi``, ``training_record`` and ``counts``, the exact counts behind
        the PPMI matrix. ``counts`` holds
        only pairs observed more than ``ppmi_min_count`` times, the only ones
        that can receive a PPMI entry; the totals include all pairs.

        Args:
            corpus: A :class:`pathlib.Path` or list of paths (UTF-8 text
                files), or an iterable of ``str`` lines, or an iterable of
                token sequences (pre-tokenised input). Lines are split on
                whitespace exactly like ``str.split()``. See
                :func:`~chronowords.utils.cooccurrence.count_cooccurrences`
                for how each form is read; a single-use iterable is copied to
                a temporary file.

        Raises:
            ValueError: If no word reaches ``min_count`` (an empty corpus, an
                exhausted iterator, or ``min_count`` too high for the corpus).
            TypeError: If ``corpus`` is a bare ``str``.

        Note:
            Swallowed failure: if sparse :func:`~scipy.sparse.linalg.svds`
            raises (common for tiny or degenerate matrices), the error is
            caught and the code falls back to a dense SVD of the matrix
            *with Gaussian noise (sigma 1e-10) added*, which yields slightly
            different, non-deterministic embeddings from the sparse path. The
            fallback does not raise; it logs a ``WARNING`` on this module's
            logger and is recorded in ``training_record["svd"]``. An empty
            PPMI matrix also logs a ``WARNING`` rather than raising.

            The sparse SVD starts from a fixed, seeded vector, so repeated
            training on the same corpus gives the same embeddings.

        Examples:
            >>> model = SVDAlgebra(n_components=2, min_count=1)
            >>> model.train(["the cat sat", "the dog ran"])
            >>> model.vocabulary
            ['the', 'cat', 'dog', 'ran', 'sat']
            >>> model.embeddings.shape
            (5, 2)
            >>> model.training_record["word_total"]
            6

        """
        M, record = self._exact_ppmi(corpus)
        self._ppmi_sparse = M
        if M.nnz == 0:
            logger.warning(
                "PPMI matrix is empty (vocabulary of %d words, no pair passed the "
                "count thresholds); embeddings will carry no co-occurrence signal",
                len(self.vocabulary),
            )

        k = max(1, min(self.n_components, min(M.shape) - 1))
        M_float = M.astype(np.float64)
        svd_path = "sparse"
        try:
            v0 = np.random.default_rng(0).uniform(-1.0, 1.0, size=min(M.shape))
            U, S, _Vt = svds(M_float, k=k, v0=v0)
        except Exception as exc:
            # Fallback to dense SVD for very small/degenerate matrices
            logger.warning(
                "Sparse SVD failed (%r); falling back to dense SVD with "
                "Gaussian noise (sigma 1e-10) — embeddings are not deterministic",
                exc,
            )
            svd_path = "dense_fallback"
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

        self.training_record = {
            "chronowords_version": _version(),
            "params": self._params(),
            **record,
            "vocab_size": len(self.vocabulary),
            "ppmi_nnz": int(M.nnz),
            "n_components_used": k,
            "svd": svd_path,
        }
        logger.info("Final embeddings shape: %s", self.embeddings.shape)

    def _exact_ppmi(self, corpus: Corpus) -> tuple[csr_matrix, dict[str, Any]]:
        logger.info("Counting words and skipgrams...")
        counts = count_cooccurrences(
            corpus,
            window_size=self.window_size,
            min_word_length=self.min_word_length,
            vocab_size=self.vocab_size,
            min_count=self.min_count,
            tokenizer=self.tokenizer,
            lowercase=self.lowercase,
            # pairs at or below the PPMI threshold can never get an entry
            min_pair_count=self.ppmi_min_count + 1,
            n_threads=self.n_threads,
        )
        logger.info("Building vocabulary...")
        self.counts = counts
        self.vocabulary = counts.vocabulary
        self._build_vocab_index()
        M = ppmi_matrix(
            counts, shift=self.shift, alpha=self.alpha, min_count=self.ppmi_min_count
        )
        record = {
            "corpus": counts.params["corpus"],
            "word_total": counts.word_total,
            "skip_total": counts.skip_total,
            "n_types": counts.n_types,
            "pairs_observed": len(counts.pair_counts),
        }
        return M, record

    def save_model(self, path: str | Path) -> None:
        """Save model embeddings and vocabulary to disk.

        Writes into ``path``: ``embeddings.npy``, ``vocabulary.pkl``,
        ``ppmi.npz`` (sparse PPMI) and ``training_record.json``, each only
        when the corresponding attribute is set.

        Args:
            path: Directory to write model files to. Created (including
                parents) if it does not exist.

        Raises:
            OSError: If ``path`` cannot be created or written (e.g. permission
                denied, read-only filesystem) — propagated from ``mkdir`` /
                ``np.save`` / ``open``.

        Note:
            Calling this before :meth:`train` writes a ``vocabulary.pkl``
            holding an empty list and no ``embeddings.npy``, which makes the
            resulting directory unloadable by :meth:`load_model`. No warning
            is emitted for a partial save. Exact ``counts`` are not saved.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> model = SVDAlgebra(n_components=2)
            >>> model.vocabulary = ["cat", "dog"]
            >>> model.embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     model.save_model(tmpdir)
            ...     sorted(p.name for p in Path(tmpdir).glob("*"))
            ['embeddings.npy', 'vocabulary.pkl']

        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._ppmi_sparse is not None:
            save_npz(path / "ppmi.npz", self._ppmi_sparse)
        if self.embeddings is not None:
            np.save(path / "embeddings.npy", self.embeddings)
        if self.training_record is not None:
            (path / "training_record.json").write_text(
                json.dumps(self.training_record, indent=2) + "\n", encoding="utf-8"
            )
        with Path.open(path / "vocabulary.pkl", "wb") as f:
            pickle.dump(self.vocabulary, f)

    def load_model(self, path: str | Path) -> None:
        """Load model embeddings and vocabulary from disk.

        Restores ``embeddings`` and ``vocabulary``, plus ``ppmi`` and
        ``training_record`` when present, from a directory written by
        :meth:`save_model`, and rebuilds the vocabulary index.

        Args:
            path: Directory containing the saved model files.

        Raises:
            ValueError: If ``path`` is not an existing directory (explicit
                check).
            FileNotFoundError: If ``embeddings.npy`` or ``vocabulary.pkl`` is
                missing from ``path`` (implicit, from ``np.load`` / ``open``).
            pickle.UnpicklingError: If ``vocabulary.pkl`` is corrupt.

        Warning:
            This method unpickles ``vocabulary.pkl``. Unpickling executes
            arbitrary code embedded in the file, so only load model directories
            from sources you trust.

        Note:
            Directories saved by chronowords < 0.4 may hold a dense
            ``ppmi.npy`` instead of ``ppmi.npz``; it is read and converted to
            a sparse matrix. ``counts`` is never restored (it is not saved).

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> model = SVDAlgebra(n_components=2)
            >>> model.vocabulary = ["cat", "dog"]
            >>> model.embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     model.save_model(tmpdir)
            ...     new_model = SVDAlgebra(n_components=2)
            ...     new_model.load_model(tmpdir)
            ...     new_model.vocabulary
            ['cat', 'dog']

        """
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Directory not found: {path}")

        self.embeddings = np.load(path / "embeddings.npy")
        with Path.open(path / "vocabulary.pkl", "rb") as f:
            self.vocabulary = pickle.load(f)
        self._build_vocab_index()

        self._ppmi_sparse = None
        if (path / "ppmi.npz").exists():
            # scipy-stubs 1.18 types the path as a generic alias ty cannot solve
            npz = load_npz(path / "ppmi.npz")  # ty: ignore[invalid-argument-type]
            self._ppmi_sparse = csr_matrix(npz)
        elif (path / "ppmi.npy").exists():
            self._ppmi_sparse = csr_matrix(np.load(path / "ppmi.npy"))
        record = path / "training_record.json"
        self.training_record = (
            json.loads(record.read_text(encoding="utf-8")) if record.exists() else None
        )
        self.counts = None

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
