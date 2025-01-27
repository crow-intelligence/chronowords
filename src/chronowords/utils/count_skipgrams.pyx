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

# Declare the numpy data types we'll use
np.import_array()
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

cdef class PPMIComputer:
    """Encapsulates PPMI computation state and methods."""

    cdef:
        object skipgram_counts  # np.ndarray but kept as object for gil
        object word_counts  # np.ndarray but kept as object for gil
        object vocabulary  # list of words
        list seeds  # hash function seeds
        int width
        double skip_total
        double word_total
        double shift
        double alpha
        vector[double] word_probs
        bint initialized

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
        """Initialize the PPMI computer with Count-Min Sketch data."""
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

    cdef int _get_min_count(self, bytes word_bytes) except -1:
        """Get minimum count from Count-Min Sketch arrays."""
        cdef:
            int i, idx, min_count = 0x7FFFFFFF
            int depth = (<np.ndarray> self.skipgram_counts).shape[0]
            np.ndarray[DTYPE_t, ndim=2] counts = self.skipgram_counts

        with nogil:
            for i in range(depth):
                with gil:
                    # Need GIL for mmh3.hash
                    idx = mmh3.hash(word_bytes, self.seeds[i]) % self.width
                if counts[i, idx] < min_count:
                    min_count = counts[i, idx]

        return min_count

    def _precompute_word_probabilities(self):
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
        """Compute PPMI for a batch of words."""
        cdef:
            int i, j, pair_count
            double pa, pb, pab, pmi
            vector[double] data
            vector[int] row_indices, col_indices
            bytes w1_bytes, w2_bytes, skipgram_key
            str word1, word2

        if not self.initialized:
            self._precompute_word_probabilities()
            self.initialized = True

        for i in range(start_idx, min(end_idx, len(self.vocabulary))):
            if self.word_probs[i] == 0:
                continue

            word1 = self.vocabulary[i]
            w1_bytes = word1.encode()
            pa = self.word_probs[i] ** (1 / self.alpha)

            for j in range(len(self.vocabulary)):
                if self.word_probs[j] == 0:
                    continue

                word2 = self.vocabulary[j]
                skipgram_key = f"{word1}#{word2}".encode()
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

def compute_ppmi_matrix_with_sketch(
        np.ndarray[DTYPE_t, ndim=2] skipgram_counts,
        np.ndarray[DTYPE_t, ndim=2] word_counts,
        list vocabulary,
        list seeds,
        int width,
        double skip_total,
        double word_total,
        double shift=1.0,
        double alpha=0.75,
        int batch_size=1024
) -> csr_matrix:
    """
    Compute PPMI matrix using Count-Min Sketch data with batched processing.
    """
    cdef:
        int n = len(vocabulary)
        list all_data = []
        list all_rows = []
        list all_cols = []

    computer = PPMIComputer(
        skipgram_counts, word_counts, vocabulary, seeds,
        width, skip_total, word_total, shift, alpha
    )

    for start_idx in range(0, n, batch_size):
        data, rows, cols = computer.compute_ppmi_batch(start_idx, start_idx + batch_size)
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