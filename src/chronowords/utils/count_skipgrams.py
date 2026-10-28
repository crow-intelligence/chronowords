"""PPMI from Count-Min Sketch tables (sketch mode), backed by the Rust core."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from chronowords import _core


class PPMIComputer:
    """Compute a PPMI matrix directly from Count-Min Sketch count tables.

    Avoids materialising a dense co-occurrence matrix: word and skip-gram
    counts are read back from the sketches via the same hashing scheme
    (MurmurHash3, bit-identical to ``mmh3.hash``) that
    :class:`~chronowords.utils.probabilistic_counter.CountMinSketch` uses to
    fill them. The kernel runs in compiled Rust (``chronowords._core``) and
    releases the GIL.

    The PPMI for a word pair ``(a, b)`` is ``max(0, log(P(a,b) / (P(a)^alpha *
    P(b))) - log(shift))``, where probabilities come from the sketch counts.
    Both words and the pair must have a sketch count strictly greater than
    ``min_count`` (default 5),
    and only strictly-positive PMI values are retained, so the result is
    sparse and non-negative.

    Cost is O(V^2 * depth) sketch lookups for a vocabulary of V words, because
    every pair is queried whether or not it was ever observed.

    Examples:
        Hand-computed cell: six lines of ``"cat dog"`` give word counts 6 and
        6 (total 12) and six ``cat#dog`` skip-grams (total 6), so
        PPMI(cat, dog) = log(1 / (0.5**0.75 * 0.5)) = 1.75 * log(2).

        >>> from chronowords.utils.probabilistic_counter import CountMinSketch
        >>> words = CountMinSketch(width=1000, depth=3)
        >>> pairs = CountMinSketch(width=1000, depth=3, track_keys=False)
        >>> for _ in range(6):
        ...     words.update("cat")
        ...     words.update("dog")
        ...     pairs.update("cat#dog")
        >>> computer = PPMIComputer(
        ...     skipgram_counts=pairs.counts,
        ...     word_counts=words.counts,
        ...     vocabulary=["cat", "dog"],
        ...     seeds=words.seeds,
        ...     width=words.width,
        ...     skip_total=float(pairs.total),
        ...     word_total=float(words.total),
        ... )
        >>> m = computer.compute_ppmi_matrix_with_sketch()
        >>> round(float(m[0, 1]), 6), round(1.75 * float(np.log(2)), 6)
        (1.213008, 1.213008)
        >>> m.nnz
        1

    """

    def __init__(
        self,
        skipgram_counts: NDArray[np.int32],
        word_counts: NDArray[np.int32],
        vocabulary: list[str],
        seeds: list[int] | None,
        width: int,
        skip_total: float,
        word_total: float,
        shift: float = 1.0,
        alpha: float = 0.75,
        min_count: int = 5,
    ) -> None:
        """Initialize the PPMI computer with Count-Min Sketch data.

        Args:
            skipgram_counts: Count-Min Sketch table for skip-gram pairs, a 2-D
                int32 array of shape ``(depth, width)`` (from
                ``CountMinSketch.arrays``).
            word_counts: Count-Min Sketch table for single words, same
                ``(depth, width)`` int32 shape and seeds as ``skipgram_counts``.
            vocabulary: Words to compute PPMI over; defines the matrix axes.
            seeds: One MurmurHash3 seed per sketch row (length ``depth``).
                Defaults to ``[1, 2, 3]`` if ``None``.
            width: Sketch table width; used as the modulus for hash indices.
            skip_total: Total number of skip-gram observations (denominator for
                joint probabilities). Must be > 0.
            word_total: Total number of word observations (denominator for
                marginal probabilities). Must be > 0.
            shift: PMI shift; ``log(shift)`` is subtracted from each PMI (a
                shifted-PPMI / negative-sampling analogue). Default 1.0 (no
                shift). Must be > 0.
            alpha: Context-distribution smoothing exponent applied to the focus
                word's marginal. Default 0.75.
            min_count: Words and pairs need a sketch count strictly greater
                than this to receive an entry. Default 5.

        Raises:
            ValueError: If any element of ``seeds`` is not an ``int``
                (explicit check here).

        Note:
            Table shapes are checked when the matrix is computed, not here:
            :meth:`compute_ppmi_batch` raises ``ValueError`` if either table
            is not ``(len(seeds), width)``, and ``TypeError`` if a table is
            not int32. ``word_total``/``skip_total``/``shift`` are not
            validated; a value of 0 yields ``inf``/``nan`` PPMI entries.

            The smoothing exponent ``alpha`` is applied only to the focus
            word's probability, not the context word's. This asymmetry is
            intentional (SGNS-style context smoothing) — see the project
            pre-mortem.

        """
        self.skipgram_counts = skipgram_counts
        self.word_counts = word_counts
        self.vocabulary = list(vocabulary)
        if seeds is None:
            self.seeds = [1, 2, 3]
        else:
            if not all(isinstance(s, int) for s in seeds):
                raise ValueError("All seeds must be integers")
            self.seeds = list(seeds)
        self.width = width
        self.skip_total = skip_total
        self.word_total = word_total
        self.shift = shift
        self.alpha = alpha
        self.min_count = min_count

    def compute_ppmi_batch(
        self, start_idx: int, end_idx: int
    ) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
        """Compute PPMI entries for focus words in ``[start_idx, end_idx)``.

        For each focus word in the half-open range (clamped to the vocabulary
        size) every context word is scanned, and an entry is emitted when both
        words and the pair clear the count-5 threshold and the PMI is
        strictly positive.

        Args:
            start_idx: First focus-word index (inclusive).
            end_idx: One past the last focus-word index (exclusive); clamped to
                ``len(vocabulary)``.

        Returns:
            A tuple ``(data, row_indices, col_indices)`` of NumPy arrays giving
            the COO components of the PPMI entries for this batch. All ``data``
            values are strictly positive. The arrays are empty when no pair in
            the batch qualifies.

        Raises:
            ValueError: If a table's shape is not ``(len(seeds), width)``, or
                ``width`` is not positive.
            TypeError: If a table is not a 2-D int32 array.
            OverflowError: If ``start_idx``/``end_idx`` or a seed is negative.

        """
        return _core.sketch_ppmi(
            self.skipgram_counts,
            self.word_counts,
            self.vocabulary,
            self.seeds,
            self.width,
            float(self.skip_total),
            float(self.word_total),
            float(self.shift),
            float(self.alpha),
            start_idx,
            end_idx,
            self.min_count,
        )

    def compute_ppmi_matrix_with_sketch(self, batch_size: int = 1024) -> csr_matrix:
        """Compute the complete PPMI matrix.

        Args:
            batch_size: Kept for API compatibility; the Rust kernel processes
                the whole vocabulary in one call, so this has no effect on
                the result or on memory. Default 1024.

        Returns:
            An ``(n, n)`` :class:`scipy.sparse.csr_matrix` (``n`` =
            vocabulary size) of non-negative float64 PPMI values. Returns an
            empty ``(n, n)`` matrix when no pair qualifies.

        Raises:
            ValueError, TypeError: As :meth:`compute_ppmi_batch`.

        """
        n = len(self.vocabulary)
        data, rows, cols = self.compute_ppmi_batch(0, n)
        if len(data) == 0:
            return csr_matrix((n, n))
        return csr_matrix((data, (rows, cols)), shape=(n, n))
