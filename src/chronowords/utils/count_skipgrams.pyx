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

    This class computes PPMI (Positive Pointwise Mutual Information) matrices
    using Count-Min Sketch for memory efficiency. It handles large-scale text data
    through batched processing and memory-efficient counting.

    The PPMI computation follows these steps:
    1. Store word and skipgram counts using Count-Min Sketch
    2. Pre-compute word probabilities with context distribution smoothing
    3. Compute PPMI values in batches
    4. Return sparse matrix representation

    Examples:
        >>> import numpy as np
        >>> from scipy.sparse import csr_matrix
        >>> # Create sample Count-Min Sketch arrays
        >>> counts = np.zeros((3, 100), dtype=np.int32)
        >>> counts[0, 0] = 10  # Add some counts
        >>> counts[1, 0] = 10
        >>> counts[2, 0] = 10
        >>> # Initialize computer
        >>> computer = PPMIComputer(
        ...     skipgram_counts=counts,
        ...     word_counts=counts,
        ...     vocabulary=["cat", "dog"],
        ...     seeds=[42, 43, 44],
        ...     width=100,
        ...     skip_total=20.0,
        ...     word_total=20.0
        ... )
        >>> # Compute PPMI matrix
        >>> matrix = computer.compute_ppmi_matrix_with_sketch()
        >>> isinstance(matrix, csr_matrix)
        True
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
            skipgram_counts: Count-Min Sketch array for skipgrams
            word_counts: Count-Min Sketch array for words
            vocabulary: List of words
            seeds: Hash function seeds
            width: Width of Count-Min Sketch tables
            skip_total: Total skipgram count
            word_total: Total word count
            shift: PMI shift parameter
            alpha: Context distribution smoothing
        """
        self.skipgram_counts = skipgram_counts
        self.word_counts = word_counts
        self.vocabulary = vocabulary
        self.seeds = seeds
        self.width = width
        self.skip_total = skip_total
        self.word_total = word_total
        self.shift = shift
        self.alpha = alpha
        self.initialized = False
        self.word_probs = vector[double]()

    cdef int _get_min_count(self, bytes word_bytes) except -1:
        """Get minimum count from Count-Min Sketch arrays.

        Args:
            word_bytes: Encoded word to look up

        Returns:
            Minimum count across hash functions

        Raises:
            -1 on error
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
        """Pre-compute word probabilities for efficiency."""
        cdef:
            int i, count
            int n = len(self.vocabulary)
            bytes word_bytes

        self.word_probs.resize(n)

        for i in range(n):
            word_bytes = self.vocabulary[i].encode()
            count = self._get_min_count(word_bytes)
            if count > 0:
                self.word_probs[i] = pow(count / self.word_total, self.alpha)
            else:
                self.word_probs[i] = 0.0

    def compute_ppmi_batch(self, int start_idx, int end_idx):
        """Compute PPMI for a batch of words.

        Args:
            start_idx: Starting vocabulary index
            end_idx: Ending vocabulary index (exclusive)

        Returns:
            Tuple of (data, row_indices, column_indices) for sparse matrix

        Examples:
            >>> import numpy as np
            >>> counts = np.ones((3, 100), dtype=np.int32)
            >>> computer = PPMIComputer(
            ...     counts, counts, ["word1", "word2"],
            ...     [42], 100, 10.0, 10.0
            ... )
            >>> data, rows, cols = computer.compute_ppmi_batch(0, 2)
            >>> len(data) == len(rows) == len(cols)
            True
        """
        cdef:
            int i, j, pair_count
            double pa, pb, pab, pmi
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
            pa = self.word_probs[i] ** (1 / self.alpha)

            for j in range(len(self.vocabulary)):
                if self.word_probs[j] == 0:
                    continue

                skipgram_key = f"{self.vocabulary[i]}#{self.vocabulary[j]}".encode()
                pair_count = self._get_min_count(skipgram_key)

                if pair_count == 0:
                    continue

                pb = self.word_probs[j]
                pab = pair_count / self.skip_total
                pmi = log(pab / (pa * pb)) - log(self.shift)

                if pmi > 0:
                    data.push_back(pmi)
                    row_indices.push_back(i)
                    col_indices.push_back(j)

        return np.array(data), np.array(row_indices), np.array(col_indices)

    def compute_ppmi_matrix_with_sketch(self, int batch_size=1024):
        """Compute complete PPMI matrix using batched processing.

        Creates a sparse PPMI matrix using memory-efficient Count-Min Sketch
        data structures and batch processing for large vocabularies.

        Args:
            batch_size: Number of words to process per batch (default: 1024)

        Returns:
            Sparse PPMI matrix as scipy.sparse.csr_matrix

        Examples:
            >>> import numpy as np
            >>> counts = np.ones((3, 100), dtype=np.int32)
            >>> vocab = ["word1", "word2"]
            >>> computer = PPMIComputer(
            ...     counts, counts, vocab, [42], 100,
            ...     skip_total=10.0, word_total=10.0
            ... )
            >>> matrix = computer.compute_ppmi_matrix_with_sketch()
            >>> isinstance(matrix, csr_matrix)
            True
            >>> matrix.shape == (2, 2)
            True
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
