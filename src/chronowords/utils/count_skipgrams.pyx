# distutils: language=c++
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

cimport cython
cimport numpy as np
from libc.math cimport log
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np
from scipy.sparse import csr_matrix

ctypedef unordered_map[string, double] DoubleMap
ctypedef unordered_map[string, int] IntMap

def compute_ppmi_matrix(
        dict skipgram_counts,
        dict word_counts,
        list vocabulary,
        double shift=1.0,
        double alpha=0.75
) -> csr_matrix:
    """
    Compute PPMI matrix from skipgram counts using efficient C++ data structures.

    Args:
        skipgram_counts: Dictionary of skipgram counts from Count-Min Sketch
        word_counts: Dictionary of word counts from Count-Min Sketch
        vocabulary: List of words in the vocabulary
        shift: PMI shifting factor (default: 1.0)
        alpha: Context distribution smoothing (default: 0.75)

    Returns:
        Sparse PPMI matrix
    """
    cdef:
        double skip_total = sum(skipgram_counts.values())
        double word_total = sum(word_counts.values())
        int n = len(vocabulary)
        vector[double] data
        vector[int] row_indices, col_indices
        DoubleMap word_probs
        IntMap word_to_idx
        string w1_enc, w2_enc
        double pa, pb, pab, pmi
        int idx
        str word, w1, w2

    # Build word probability map and index map
    for idx in range(n):
        word = vocabulary[idx]
        count = word_counts.get(word, 0)
        if count > 0:
            word_probs[word.encode()] = count / word_total
            word_to_idx[word.encode()] = idx

    # Compute PPMI values
    for (w1, w2), count in skipgram_counts.items():
        w1_enc = w1.encode()
        w2_enc = w2.encode()

        if w1_enc in word_probs and w2_enc in word_probs:
            pa = word_probs[w1_enc]
            pb = pow(word_probs[w2_enc], alpha)
            pab = count / skip_total

            pmi = log(pab / (pa * pb)) - log(shift)

            if pmi > 0:
                idx = word_to_idx[w1_enc]
                data.push_back(pmi)
                row_indices.push_back(word_to_idx[w1_enc])
                col_indices.push_back(word_to_idx[w2_enc])

                # Add symmetric counterpart
                data.push_back(pmi)
                row_indices.push_back(word_to_idx[w2_enc])
                col_indices.push_back(word_to_idx[w1_enc])

    # Create sparse matrix
    return csr_matrix(
        (
            np.array(data, dtype=np.float64),
            (
                np.array(row_indices, dtype=np.int32),
                np.array(col_indices, dtype=np.int32)
            )
        ),
        shape=(n, n)
    )

# write a short descirption of the project
# PPMI matrix based language models and diachronic word embeddings