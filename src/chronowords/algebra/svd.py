"""
Implementation of word vector algebra using SVD-based embeddings with probabilistic counting.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from datasketch import CountMinSketch
from nltk.util import skipgrams
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine


@dataclass
class WordSimilarity:
    """Container for word similarity results."""

    word: str
    similarity: float


@dataclass
class AnalogyResult:
    """Container for analogy results."""

    words: List[str]
    similarities: List[float]


class ProbabilisticCounter:
    """Counts word and skipgram frequencies using probabilistic data structures."""

    def __init__(self, width: int = 1_000_000, depth: int = 10, seed: int = 42):
        self.width = width
        self.depth = depth
        self.seed = seed

    def count_corpus(
        self, corpus: Generator[str, None, None], window_size: int = 5
    ) -> Tuple[Dict[str, int], List[Tuple[str, str, int]]]:
        """Count words and skipgrams using Count-Min Sketch."""
        word_cms = CountMinSketch(width=self.width, depth=self.depth, seed=self.seed)
        skipgram_cms = CountMinSketch(
            width=self.width, depth=self.depth, seed=self.seed
        )

        # First pass: count words and collect vocabulary
        vocabulary = set()
        for line in corpus:
            words = line.split()
            vocabulary.update(words)

            # Update word counts
            for word in words:
                word_cms.update(word.encode())

            # Generate and count skipgrams
            skips = skipgrams(words, 2, window_size)
            for w1, w2 in skips:
                skipgram = f"{w1}#{w2}".encode()
                skipgram_cms.update(skipgram)

        # Convert vocabulary to sorted list and get counts
        vocab_list = sorted(vocabulary)
        word_counts = {word: word_cms.query(word.encode()) for word in vocab_list}

        # Get skipgram counts
        skipgram_counts = []
        for i, w1 in enumerate(vocab_list):
            for w2 in vocab_list[i + 1 :]:
                skipgram = f"{w1}#{w2}".encode()
                count = skipgram_cms.query(skipgram)
                if count > 0:
                    skipgram_counts.append((w1, w2, count))

        return word_counts, skipgram_counts

    def create_cooc_matrix(
        self, vocab: List[str], skipgram_counts: List[Tuple[str, str, int]]
    ) -> csr_matrix:
        """Create co-occurrence matrix from skipgram counts."""
        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        rows, cols, data = [], [], []

        for w1, w2, count in skipgram_counts:
            i, j = word_to_idx[w1], word_to_idx[w2]
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([count, count])

        n = len(vocab)
        return csr_matrix((data, (rows, cols)), shape=(n, n))


class SVDAlgebra:
    """Implements word vector algebra using SVD-based embeddings."""

    def __init__(
        self,
        corpus_dir: Optional[Union[str, Path]] = None,
        n_components: int = 300,
        window_size: int = 5,
        min_word_length: int = 3,
        cms_width: int = 1_000_000,
        cms_depth: int = 10,
    ):
        """
        Initialize SVDAlgebra.

        Args:
            corpus_dir: Directory containing the text corpus
            n_components: Number of SVD components
            window_size: Window size for skipgrams
            min_word_length: Minimum word length to consider
            cms_width: Width of Count-Min Sketch tables
            cms_depth: Depth of Count-Min Sketch tables
        """
        self.n_components = n_components
        self.window_size = window_size
        self.min_word_length = min_word_length
        self.cms_width = cms_width
        self.cms_depth = cms_depth

        self.vocabulary: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

        if corpus_dir is not None:
            self._initialize_from_path(Path(corpus_dir))

    def _initialize_from_path(self, corpus_dir: Path) -> None:
        """Initialize model either from corpus or saved model files."""
        model_files = list(corpus_dir.glob("*.npy"))
        vocab_files = list(corpus_dir.glob("*.p"))

        if model_files and vocab_files:
            self.load_model(corpus_dir)
        else:
            self.train(self._read_corpus(corpus_dir))

    def _read_corpus(self, corpus_dir: Path) -> Generator[str, None, None]:
        """Read corpus files line by line."""
        for file_path in corpus_dir.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line.strip()

    def train(self, corpus: Generator[str, None, None]) -> None:
        """Train the model on a text corpus."""
        counter = ProbabilisticCounter(width=self.cms_width, depth=self.cms_depth)

        # Count words and skipgrams
        word_counts, skipgram_counts = counter.count_corpus(
            corpus, window_size=self.window_size
        )

        # Create vocabulary and co-occurrence matrix
        self.vocabulary = sorted(word_counts.keys())
        cooc_matrix = counter.create_cooc_matrix(self.vocabulary, skipgram_counts)

        # Calculate PPMI matrix
        total_tokens = sum(word_counts.values())
        word_probs = {word: count / total_tokens for word, count in word_counts.items()}
        M = self._calculate_ppmi(cooc_matrix, word_probs)

        # Perform SVD
        U, _, V = self._compute_svd(M)
        self.embeddings = self._normalize_vectors(U.T + V.T)

    def _calculate_ppmi(
        self,
        cooc_matrix: csr_matrix,
        word_probs: Dict[str, float],
        alpha: float = 0.75,
        shift: float = 1.0,
    ) -> csr_matrix:
        """Calculate Positive Pointwise Mutual Information matrix."""
        p_w = np.array([word_probs[w] for w in self.vocabulary])
        p_c = p_w**alpha
        expected = np.outer(p_w, p_c)

        with np.errstate(divide="ignore", invalid="ignore"):
            M = cooc_matrix.toarray() / (expected * cooc_matrix.sum())
            M = np.log(M) - np.log(shift)
            M[np.isneginf(M) | np.isnan(M)] = 0
            M[M < 0] = 0

        return csr_matrix(M)

    @staticmethod
    def _compute_svd(M: csr_matrix) -> tuple:
        """Compute SVD on sparse matrix."""
        try:
            from sparsesvd import sparsesvd

            return sparsesvd(M, 300)
        except ImportError:
            from scipy.sparse.linalg import svds

            return svds(M, k=300)

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length."""
        norms = np.sqrt(np.sum(vectors * vectors, axis=0, keepdims=True))
        norms[norms == 0] = 1
        return vectors / norms

    def save_model(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "vocabulary.pkl", "wb") as f:
            pickle.dump(self.vocabulary, f)

    def load_model(self, path: Union[str, Path]) -> None:
        """Load model from disk."""
        path = Path(path)
        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "vocabulary.pkl", "rb") as f:
            self.vocabulary = pickle.load(f)

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a word."""
        try:
            idx = self.vocabulary.index(word)
            return self.embeddings[idx]
        except ValueError:
            return None

    def most_similar(self, word: str, n: int = 10) -> List[WordSimilarity]:
        """Find the n most similar words."""
        vector = self.get_vector(word)
        if vector is None:
            return []

        sims = self.embeddings.dot(vector)
        most_similar_values = np.argsort(sims)[-n - 10 :][::-1]

        results = []
        for idx in most_similar_values:
            candidate = self.vocabulary[idx]
            if (
                candidate != word
                and len(candidate) > self.min_word_length
                and candidate.isalpha()
            ):
                results.append(WordSimilarity(candidate, float(sims[idx])))
                if len(results) == n:
                    break

        return results

    def distance(self, word1: str, word2: str) -> Optional[float]:
        """Calculate cosine distance between two words."""
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        if vec1 is None or vec2 is None:
            return None

        return min(cosine(vec1, vec2), 1.0)

    def analogy(
        self, positive: List[str], negative: str, n: int = 3
    ) -> Optional[AnalogyResult]:
        """Solve word analogies."""
        if len(positive) != 2:
            raise ValueError("Exactly two positive words are required")

        vectors = []
        for word in positive + [negative]:
            vec = self.get_vector(word)
            if vec is None:
                return None
            vectors.append(vec)

        target = vectors[2] - vectors[0] - vectors[1]
        sims = self.embeddings.dot(target)

        results = []
        similarities = []
        seen = set(positive + [negative])

        for idx in np.argsort(sims)[::-1]:
            word = self.vocabulary[idx]
            if word not in seen and len(word) > self.min_word_length and word.isalpha():
                results.append(word)
                similarities.append(float(sims[idx]))
                if len(results) == n:
                    break

        return AnalogyResult(results, similarities)
