"""
SVD-based word embedding implementation with memory-efficient counting.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from nltk.util import skipgrams
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

from ..utils.count_skipgrams import compute_ppmi_matrix_with_sketch
from ..utils.probabilistic_counter import CountMinSketch


@dataclass
class WordSimilarity:
    """Container for word similarity results."""

    word: str
    similarity: float


@dataclass
class AnalogyResult:
    """Container for word analogy results."""

    words: List[str]
    similarities: List[float]


class SVDAlgebra:
    """
    Implements word vector algebra using SVD-based embeddings.
    Uses Count-Min Sketch for memory-efficient counting and
    Cython-optimized PPMI computation.
    """

    def __init__(
        self,
        n_components: int = 300,
        window_size: int = 5,
        min_word_length: int = 3,
        cms_width: int = 1_000_000,
        cms_depth: int = 5,
    ):
        """
        Initialize SVDAlgebra.

        Args:
            n_components: Number of SVD components
            window_size: Window size for skipgrams
            min_word_length: Minimum word length to consider
            cms_width: Width of Count-Min Sketch tables
            cms_depth: Number of hash functions
        """
        self.n_components = n_components
        self.window_size = window_size
        self.min_word_length = min_word_length
        self.cms_width = cms_width
        self.cms_depth = cms_depth

        self.vocabulary: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def train(self, corpus: Generator[str, None, None]) -> None:
        """
        Train the model on a text corpus using Count-Min Sketch and Cython optimizations.

        Args:
            corpus: Generator yielding text lines
        """
        # Initialize Count-Min Sketches
        word_counter = CountMinSketch(self.cms_width, self.cms_depth)
        skipgram_counter = CountMinSketch(self.cms_width, self.cms_depth)

        print("Counting words and skipgrams...")

        # First pass: count words and skipgrams
        for line in corpus:
            words = [w for w in line.split() if len(w) >= self.min_word_length]

            # Update word counts
            for word in words:
                word_counter.update(word)

            # Generate and count skipgrams
            skips = skipgrams(words, 2, self.window_size)
            for w1, w2 in skips:
                skipgram_counter.update(f"{w1}#{w2}")

        print(f"Total words: {word_counter.total}")
        print(f"Total skipgrams: {skipgram_counter.total}")

        # Get vocabulary from heavy hitters
        print("Building vocabulary...")
        vocab_candidates = word_counter.get_heavy_hitters(
            0.0001
        )  # Words appearing in >0.01% of corpus
        self.vocabulary = [word for word, count in vocab_candidates]
        print(f"Vocabulary size: {len(self.vocabulary)}")

        if not self.vocabulary:
            raise ValueError("No words found meeting minimum frequency threshold")

        # Get arrays and parameters for Cython function
        word_counts, word_seeds, width = word_counter.arrays
        skipgram_counts, _, _ = skipgram_counter.arrays

        print("Computing PPMI matrix...")
        # Compute PPMI matrix using Cython
        M = compute_ppmi_matrix_with_sketch(
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

        print("PPMI matrix shape:", M.shape)
        print("PPMI matrix non-zeros:", M.nnz)

        print("Computing SVD...")
        # Convert to dense for SVD
        M_dense = M.toarray()

        # Add small noise to avoid zero matrices
        M_dense += np.random.normal(0, 1e-10, M_dense.shape)

        # Perform SVD
        try:
            U, S, Vt = np.linalg.svd(M_dense, full_matrices=False)
        except np.linalg.LinAlgError:
            print("SVD failed, trying with more stable algorithm...")
            U, S, Vt = np.linalg.svd(
                M_dense, full_matrices=False, lapack_driver="gesvd"
            )

        print("Singular values:", S[:10])  # Print first 10 singular values

        # Take top n_components
        U = U[:, : self.n_components]
        S = S[: self.n_components]

        # Create word embeddings (U scaled by singular values)
        self.embeddings = U * np.sqrt(S)

        print("Final embeddings shape:", self.embeddings.shape)
        print("Embeddings non-zeros:", np.count_nonzero(self.embeddings))

        # Verify embeddings
        norms = np.linalg.norm(self.embeddings, axis=1)
        print("Min norm:", np.min(norms))
        print("Max norm:", np.max(norms))
        print("Mean norm:", np.mean(norms))

        # Clear temporary data
        del word_counter
        del skipgram_counter
        del M

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: Directory to save model files
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "vocabulary.pkl", "wb") as f:
            pickle.dump(self.vocabulary, f)

    def load_model(self, path: Union[str, Path]) -> None:
        """
        Load model from disk.

        Args:
            path: Directory containing model files
        """
        path = Path(path)
        if not path.is_dir():
            raise ValueError(f"Directory not found: {path}")

        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "vocabulary.pkl", "rb") as f:
            self.vocabulary = pickle.load(f)

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get the embedding vector for a word.

        Args:
            word: Input word

        Returns:
            Word vector if word is in vocabulary, None otherwise
        """
        try:
            idx = self.vocabulary.index(word)
            return self.embeddings[idx]
        except ValueError:
            return None

    def most_similar(self, word: str, n: int = 10) -> List[WordSimilarity]:
        """
        Find the n most similar words.

        Args:
            word: Query word
            n: Number of similar words to return

        Returns:
            List of WordSimilarity objects sorted by similarity
        """
        vector = self.get_vector(word)
        if vector is None:
            return []

        # Calculate norms first
        query_norm = np.linalg.norm(vector)
        vocab_norms = np.linalg.norm(self.embeddings, axis=1)

        # Use a small epsilon to avoid division by zero
        epsilon = 1e-10
        if query_norm < epsilon:
            return []

        # Add epsilon to avoid division by zero
        vocab_norms = np.maximum(vocab_norms, epsilon)

        # Calculate similarities
        similarities = self.embeddings @ vector
        similarities = similarities / (vocab_norms * query_norm)

        # Clip to valid cosine similarity range and handle NaNs
        similarities = np.clip(similarities, -1.0, 1.0)
        similarities = np.nan_to_num(similarities, nan=-1.0)

        # Get top n results
        most_similar_indices = np.argsort(similarities)[::-1]

        results = []
        for idx in most_similar_indices:
            candidate = self.vocabulary[idx]
            similarity = float(similarities[idx])
            if candidate != word and similarity > -1:
                results.append(WordSimilarity(candidate, similarity))
                if len(results) == n:
                    break

        return results

    def distance(self, word1: str, word2: str) -> Optional[float]:
        """
        Calculate cosine distance between two words.

        Args:
            word1: First word
            word2: Second word

        Returns:
            Cosine distance between word vectors, or None if either word is unknown
        """
        vec1 = self.get_vector(word1)
        vec2 = self.get_vector(word2)

        if vec1 is None or vec2 is None:
            return None

        # Check for zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return None

        return min(cosine(vec1, vec2), 1.0)

    def analogy(
        self, positive: List[str], negative: str, n: int = 3
    ) -> Optional[AnalogyResult]:
        """
        Solve word analogies (e.g., king - man + woman ≈ queen).

        Args:
            positive: List of two positive words for the analogy
            negative: The negative word
            n: Number of results to return

        Returns:
            AnalogyResult containing similar words and their similarity scores,
            or None if any of the input words are not in vocabulary
        """
        if len(positive) != 2:
            raise ValueError("Exactly two positive words are required")

        # Get vectors for all words
        vectors = []
        for word in positive + [negative]:
            vec = self.get_vector(word)
            if vec is None:
                return None
            vectors.append(vec)

        # Calculate target vector: negative - positive[0] - positive[1]
        target = vectors[2] - vectors[0] - vectors[1]

        # Check for zero vector
        target_norm = np.linalg.norm(target)
        if target_norm < 1e-10:
            return None

        # Normalize target vector
        target = target / target_norm

        # Calculate similarities with all words
        vocab_norms = np.linalg.norm(self.embeddings, axis=1)
        similarities = self.embeddings @ target

        # Normalize and clip similarities
        similarities = np.clip(similarities / vocab_norms, -1.0, 1.0)
        similarities = np.nan_to_num(similarities, nan=-1.0)

        # Get top results
        results = []
        similarity_scores = []
        seen = set(positive + [negative])

        # Get indices of top matches
        most_similar_indices = np.argsort(similarities)[::-1]

        for idx in most_similar_indices:
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

        if not results:
            return None

        return AnalogyResult(results, similarity_scores)
