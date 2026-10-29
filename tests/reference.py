"""Pure-Python reference implementations the compiled core is tested against."""

from collections.abc import Sequence

import numpy as np


def skipgrams(tokens: Sequence[str], window: int) -> list[tuple[str, str]]:
    """Directional pairs as ``nltk.util.skipgrams(tokens, 2, window)`` yields them.

    Each token pairs with each of the next ``window + 1`` tokens, in order.

    >>> skipgrams("a b c d".split(), 1)
    [('a', 'b'), ('a', 'c'), ('b', 'c'), ('b', 'd'), ('c', 'd')]
    """
    return [
        (tokens[i], tokens[j])
        for i in range(len(tokens))
        for j in range(i + 1, min(len(tokens), i + window + 2))
    ]


# Output of ``list(nltk.util.skipgrams("a b c d e f g h i".split(), 2, 2))``
# with nltk 3.9, recorded so the reference stays anchored to nltk's semantics
# without depending on nltk.
NLTK_9_TOKENS_K2 = [
    ("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("b", "e"),
    ("c", "d"), ("c", "e"), ("c", "f"), ("d", "e"), ("d", "f"), ("d", "g"),
    ("e", "f"), ("e", "g"), ("e", "h"), ("f", "g"), ("f", "h"), ("f", "i"),
    ("g", "h"), ("g", "i"), ("h", "i"),
]  # fmt: skip


def ppmi(
    pair_counts: dict[tuple[int, int], int],
    vocab_size: int,
    *,
    alpha: float = 0.75,
    shift: float = 1.0,
    min_count: int = 5,
) -> np.ndarray:
    """Dense PPMI with context-distribution smoothing, straight from the paper.

    Levy, Goldberg & Dagan (2015): P(w) and P(c) are the row and column
    marginals of the full count matrix, and the context distribution is
    smoothed as col(c)**alpha / sum(col**alpha). Pairs with a count of at
    most ``min_count`` get no entry but still count in the marginals.
    """
    m = np.zeros((vocab_size, vocab_size))
    for (i, j), n in pair_counts.items():
        m[i, j] = n
    total = m.sum()
    out = np.zeros_like(m)
    if total == 0:
        return out
    p_w = m.sum(axis=1) / total
    col = m.sum(axis=0)
    p_c = col**alpha / (col**alpha).sum()
    for (i, j), n in pair_counts.items():
        if n > min_count:
            pmi = np.log((n / total) / (p_w[i] * p_c[j])) - np.log(shift)
            out[i, j] = max(pmi, 0.0)
    return out
