# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

cimport cython
cimport numpy as np
from libc.math cimport log, pow
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.utility cimport pair
from libcpp.vector cimport vector

import mmh3
import numpy as np
from scipy.sparse import csr_matrix

np.import_array()
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

cdef class PPMIComputer:
    """Encapsulates PPMI computation state and methods.

    Computes a Positive Pointwise Mutual Information (PPMI) matrix directly
    from Count-Min Sketch count tables, avoiding materialisation of a dense
    co-occurrence matrix. Word and skip-gram counts are read back from the
    sketch via the same hashing scheme used to populate it.

    The PPMI for a word pair ``(a, b)`` is ``max(0, log(P(a,b) / (P(a)^alpha *
    P(b))) - log(shift))``, where probabilities come from the sketch counts.
    Both words must have a sketch count strictly greater than 5, and only
    strictly-positive PMI values are retained, so the result is sparse and
    non-negative.
    """
    cdef:
        public np.ndarray skipgram_counts
        public np.ndarray word_counts
        public list vocabulary
        public list seeds
        public int width
        public double skip_total
        public double word_total
        public double shift
        public double alpha
        public bint initialized
        vector[double] word_probs

    def __init__(self,
                 np.ndarray skipgram_counts,
                 np.ndarray word_counts,
                 list vocabulary,
                 list seeds,
                 int width,
                 double skip_total,
                 double word_total,
                 double shift=1.0,
                 double alpha=0.75):
        """Initialize the PPMI computer with Count-Min Sketch data.

        Args:
            skipgram_counts: Count-Min Sketch table for skip-gram pairs, a 2-D
                int32 array of shape ``(depth, width)`` (from
                ``CountMinSketch.arrays``).
            word_counts: Count-Min Sketch table for single words, same
                ``(depth, width)`` int32 shape and seeds as ``skipgram_counts``.
            vocabulary: Words to compute PPMI over; defines the matrix axes.
            seeds: One mmh3 hash seed per sketch row (length ``depth``).
                Defaults to ``[1, 2, 3]`` if ``None``.
            width: Sketch table width; used as the modulus for hash indices.
                Must match the tables' second dimension.
            skip_total: Total number of skip-gram observations (denominator for
                joint probabilities). Must be > 0.
            word_total: Total number of word observations (denominator for
                marginal probabilities). Must be > 0.
            shift: PMI shift; ``log(shift)`` is subtracted from each PMI (an
                a shifted-PPMI / negative-sampling analogue). Must be > 0.
            alpha: Context-distribution smoothing exponent applied to the focus
                word's marginal (typically 0.75).

        Raises:
            ValueError: If any element of ``seeds`` is not an ``int``.

        Note:
            Preconditions not enforced:
                - Array shapes, dtypes, and ``width`` consistency are assumed,
                  not checked; a mismatch produces wrong counts or an
                  out-of-bounds access (bounds checking is disabled).
                - ``word_total``/``skip_total`` must be positive; a value of 0
                  yields ``inf``/``nan`` PPMI entries. ``shift`` must be
                  positive, otherwise ``log(shift)`` is ``nan``/``-inf``.
                - The smoothing exponent ``alpha`` is applied only to the focus
                  word's probability (``pa``), not the context word's (``pb``);
                  this asymmetry is intentional but easy to misread — see the
                  project pre-mortem.
        """
        self.skipgram_counts = skipgram_counts
        self.word_counts = word_counts
        self.vocabulary = vocabulary
        # Validate seeds are integers
        if seeds is None:
            self.seeds = [1, 2, 3]
        else:
            if not all(isinstance(s, int) for s in seeds):
                raise ValueError("All seeds must be integers")
            self.seeds = seeds
        self.width = width
        self.skip_total = skip_total
        self.word_total = word_total
        self.shift = shift
        self.alpha = alpha
        self.initialized = False
        self.word_probs = vector[double]()

    cdef int _get_min_count(self, bytes word_bytes) except -1:
        """Return the Count-Min Sketch estimate for ``word_bytes``.

        Hashes the key with each row seed and returns the minimum count across
        rows — the standard CMS point query, which never underestimates the
        true count.
        """
        cdef:
            int i, idx, min_count
            int depth = self.skipgram_counts.shape[0]
            np.ndarray[DTYPE_t, ndim=2] counts = self.skipgram_counts

        min_count = 0x7FFFFFFF

        for i in range(depth):
            idx = mmh3.hash(word_bytes, self.seeds[i]) % self.width
            if counts[i, idx] < min_count:
                min_count = counts[i, idx]

        return min_count

    cdef void _precompute_word_probabilities(self):
        """Pre-compute per-word marginal probabilities into ``word_probs``.

        For each vocabulary word, queries the sketch and stores
        ``count / word_total`` if the count is strictly greater than 5,
        otherwise 0.0. A stored probability of 0.0 marks the word as too rare
        and excludes it from all PPMI pairs.
        """
        cdef:
            int i, count
            int n = len(self.vocabulary)
            bytes word_bytes
            double epsilon = 1e-10

        self.word_probs.resize(n)

        for i in range(n):
            word_bytes = self.vocabulary[i].encode()
            count = self._get_min_count(word_bytes)
            if count > 5:  # Minimum count threshold
                self.word_probs[i] = count / self.word_total  # Store raw probability
            else:
                self.word_probs[i] = 0.0

    def compute_ppmi_batch(self, int start_idx, int end_idx):
        """Compute PPMI entries for focus words in ``[start_idx, end_idx)``.

        Lazily precomputes word probabilities on the first call. For each focus
        word in the half-open range (clamped to the vocabulary size) it scans
        every context word and emits a PPMI entry when both words clear the
        count-5 threshold and the resulting PMI is strictly positive.

        Args:
            start_idx: First focus-word index (inclusive).
            end_idx: One past the last focus-word index (exclusive); clamped to
                ``len(vocabulary)``.

        Returns:
            A tuple ``(data, row_indices, col_indices)`` of NumPy arrays giving
            the COO components of the PPMI entries for this batch. All ``data``
            values are strictly positive. The arrays are empty when no pair in
            the batch qualifies.
        """
        cdef:
            int i, j, pair_count
            double pa, pb, pab, pmi
            double epsilon = 1e-10
            vector[double] data
            vector[int] row_indices, col_indices
            bytes word_bytes, skipgram_key

        if not self.initialized:
            self._precompute_word_probabilities()
            self.initialized = True

        for i in range(start_idx, min(end_idx, len(self.vocabulary))):
            if self.word_probs[i] == 0:
                continue

            word_bytes = self.vocabulary[i].encode()
            pa = pow(self.word_probs[i] + epsilon, self.alpha)  # Apply smoothing here

            for j in range(len(self.vocabulary)):
                if self.word_probs[j] == 0:
                    continue

                skipgram_key = f"{self.vocabulary[i]}#{self.vocabulary[j]}".encode()
                pair_count = self._get_min_count(skipgram_key)

                if pair_count <= 5:  # Minimum count threshold
                    continue

                pb = self.word_probs[j] + epsilon
                pab = (pair_count + epsilon) / self.skip_total

                # Only compute PMI if joint prob > product of marginals
                if pab > pa * pb:
                    pmi = log(pab / (pa * pb)) - log(self.shift)
                    if pmi > 0:
                        data.push_back(pmi)
                        row_indices.push_back(i)
                        col_indices.push_back(j)

        return np.array(data), np.array(row_indices), np.array(col_indices)

    def compute_ppmi_matrix_with_sketch(self, int batch_size=1024):
        """Compute the complete PPMI matrix using batched processing.

        Calls :meth:`compute_ppmi_batch` over successive focus-word batches and
        assembles the results into a single sparse matrix.

        Args:
            batch_size: Number of focus words processed per batch. Affects
                memory/throughput only, not the result.

        Returns:
            An ``(n, n)`` :class:`scipy.sparse.csr_matrix` (``n`` =
            vocabulary size) of non-negative PPMI values. Returns an empty
            ``(n, n)`` matrix when no pair qualifies.
        """
        cdef:
            int n = len(self.vocabulary)
            list all_data = []
            list all_rows = []
            list all_cols = []

        for start_idx in range(0, n, batch_size):
            data, rows, cols = self.compute_ppmi_batch(start_idx, start_idx + batch_size)
            if len(data) > 0:
                all_data.append(data)
                all_rows.append(rows)
                all_cols.append(cols)

        if not all_data:
            return csr_matrix((n, n))

        return csr_matrix(
            (
                np.concatenate(all_data),
                (np.concatenate(all_rows), np.concatenate(all_cols))
            ),
            shape=(n, n)
        )
