"""Exact two-pass word and skip-gram counting, and PPMI from exact counts.

Pass 1 counts every token and fixes the vocabulary. Pass 2 counts directional
skip-gram pairs between vocabulary words only. Memory is bounded by the number
of distinct word types plus the number of distinct observed vocabulary pairs,
not by corpus size. Both passes run in the compiled core
(``chronowords._core``); file input is read there directly, with the GIL
released.
"""

import os
import tempfile
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from chronowords import _core


Corpus = os.PathLike | Sequence[os.PathLike] | Iterable[str] | Iterable[Sequence[str]]
"""Accepted corpus forms; see :func:`count_cooccurrences`."""


@dataclass(frozen=True)
class CooccurrenceCounts:
    """Exact counts from one corpus, plus everything needed to audit them.

    Attributes:
        vocabulary: Vocabulary words, most frequent first (ties alphabetical).
            Defines the row/column order of every matrix built from these
            counts.
        word_counts: Corpus count of each vocabulary word (int64, parallel to
            ``vocabulary``).
        pair_rows: Vocabulary index of the focus word of each observed pair.
        pair_cols: Vocabulary index of the context word of each observed pair.
        pair_counts: Number of times each ``(row, col)`` pair was observed.
            Only pairs with a nonzero count are stored; sorted by row, then
            column.
        word_total: Number of tokens kept after the ``min_word_length``
            filter, including out-of-vocabulary tokens.
        skip_total: Number of skip-grams in the corpus, including those with an
            out-of-vocabulary word.
        n_types: Number of distinct token types seen in pass 1.
        params: The counting parameters and corpus description that produced
            these counts.

    """

    vocabulary: list[str]
    word_counts: NDArray[np.int64]
    pair_rows: NDArray[np.int64]
    pair_cols: NDArray[np.int64]
    pair_counts: NDArray[np.int64]
    word_total: int
    skip_total: int
    n_types: int
    params: dict[str, Any] = field(default_factory=dict)

    def matrix(self) -> csr_matrix:
        """Return the raw pair counts as a ``(V, V)`` sparse int64 matrix.

        Examples:
            >>> counts = count_cooccurrences(["cat dog"] * 3, min_count=1)
            >>> counts.matrix().toarray().tolist()
            [[0, 3], [0, 0]]

        """
        n = len(self.vocabulary)
        return csr_matrix(
            (self.pair_counts, (self.pair_rows, self.pair_cols)),
            shape=(n, n),
            dtype=np.int64,
        )


def _paths(corpus: object) -> list[str] | None:
    """Return file paths if ``corpus`` names files, else ``None``."""
    if isinstance(corpus, os.PathLike):
        return [os.fspath(cast("os.PathLike[str]", corpus))]
    if (
        isinstance(corpus, list | tuple)
        and corpus
        and all(isinstance(p, os.PathLike) for p in corpus)
    ):
        return [os.fspath(cast("os.PathLike[str]", p)) for p in corpus]
    return None


def _describe(corpus: object, paths: list[str] | None) -> dict[str, Any]:
    if paths is not None:
        return {
            "kind": "files",
            "files": [{"path": p, "bytes": Path(p).stat().st_size} for p in paths],
        }
    return {"kind": type(corpus).__name__}


def count_cooccurrences(
    corpus: Corpus,
    *,
    window_size: int = 5,
    min_word_length: int = 3,
    vocab_size: int | None = 50_000,
    min_count: int = 100,
) -> CooccurrenceCounts:
    """Count words and skip-gram pairs exactly, in two passes.

    Tokenisation is whitespace splitting exactly as Python's ``str.split()``,
    then dropping tokens shorter than ``min_word_length`` code points. Lines are
    context boundaries. Each kept token pairs with the next ``window_size + 1``
    kept tokens (the pairs ``nltk.util.skipgrams(tokens, 2, window_size)``
    yields); out-of-vocabulary tokens still occupy window positions. Pairs are
    directional: ``(a, b)`` means ``a`` came first.

    Args:
        corpus: One of:

            - a :class:`pathlib.Path` (or any ``os.PathLike``), or a list/tuple
              of them: UTF-8 text files read in the compiled core, one line per
              line, universal newlines;
            - a list/tuple of ``str`` lines, or of token sequences
              (pre-tokenised input; tokens must be non-empty and contain no
              whitespace): iterated twice;
            - any other iterable of ``str`` lines or token sequences (a
              generator, an open file): iterated **once**, and copied to a
              temporary file during pass 1 so pass 2 can re-read it. That costs
              disk space roughly equal to the corpus text, in the directory
              :func:`tempfile.gettempdir` picks (``TMPDIR``).
        window_size: Skip-gram window; each token pairs with the next
            ``window_size + 1`` tokens. Default 5.
        min_word_length: Minimum token length in code points. Default 3.
        vocab_size: Keep at most this many words, most frequent first, ties
            broken alphabetically. ``None`` keeps every word that passes
            ``min_count``. Default 50,000.
        min_count: Keep only words occurring at least this many times.
            Default 100.

    Returns:
        A :class:`CooccurrenceCounts` with the vocabulary, its word counts, the
        observed pair counts, both totals and the parameters used.

    Raises:
        TypeError: If ``corpus`` is a bare ``str`` (pass a ``Path`` for a
            file), or an item is neither ``str`` nor a sequence of ``str``.
        ValueError: If a parameter is out of range, a pre-tokenised token is
            empty or contains whitespace, a file is not valid UTF-8, or no
            word reaches ``min_count`` (an empty vocabulary).
        OSError: If a file cannot be read or the temporary copy cannot be
            written.

    Examples:
        Hand-computed: "cat dog" three times gives cat=3, dog=3, six tokens,
        three ``(cat, dog)`` pairs and no ``(dog, cat)`` pair.

        >>> counts = count_cooccurrences(["cat dog"] * 3, min_count=1)
        >>> counts.vocabulary, counts.word_counts.tolist()
        (['cat', 'dog'], [3, 3])
        >>> counts.word_total, counts.skip_total
        (6, 3)
        >>> list(zip(counts.pair_rows.tolist(), counts.pair_cols.tolist(),
        ...          counts.pair_counts.tolist()))
        [(0, 1, 3)]

    """
    if window_size < 0:
        raise ValueError(f"window_size must be >= 0, got {window_size}")
    if min_word_length < 0:
        raise ValueError(f"min_word_length must be >= 0, got {min_word_length}")
    if vocab_size is not None and vocab_size < 1:
        raise ValueError(f"vocab_size must be >= 1 or None, got {vocab_size}")
    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}")
    if isinstance(corpus, str | bytes):
        raise TypeError(
            "corpus is a single string; pass an iterable of lines, or a "
            "pathlib.Path for a file"
        )

    paths = _paths(corpus)
    # Past the path check, corpus is an iterable of lines or token sequences;
    # the compiled core validates each item.
    items = cast("Iterable[str] | Iterable[Sequence[str]]", corpus)
    params: dict[str, Any] = {
        "window_size": window_size,
        "min_word_length": min_word_length,
        "vocab_size": vocab_size,
        "min_count": min_count,
        "tokenizer": "str.split",
        "corpus": _describe(corpus, paths),
    }

    with tempfile.TemporaryDirectory(prefix="chronowords-") as tmp:
        if paths is not None:
            words, counts, word_total = _core.count_words_files(paths, min_word_length)
        elif isinstance(corpus, list | tuple):
            words, counts, word_total = _core.count_words(items, min_word_length)
        else:
            spill = str(Path(tmp) / "corpus.txt")
            words, counts, word_total = _core.count_words(items, min_word_length, spill)
            paths = [spill]

        keep = int(np.searchsorted(-counts, -min_count, side="right"))
        if vocab_size is not None:
            keep = min(keep, vocab_size)
        if keep == 0:
            top = f"{words[0]!r} with {counts[0]}" if words else "no tokens at all"
            raise ValueError(
                f"Empty vocabulary: no word occurs at least min_count={min_count} "
                f"times (most frequent: {top}). Lower min_count for small corpora."
            )
        vocabulary = words[:keep]

        if paths is not None:
            rows, cols, pcounts, skip_total = _core.count_pairs_files(
                paths, vocabulary, window_size, min_word_length
            )
        else:
            rows, cols, pcounts, skip_total = _core.count_pairs(
                items, vocabulary, window_size, min_word_length
            )

    return CooccurrenceCounts(
        vocabulary=vocabulary,
        word_counts=counts[:keep].copy(),
        pair_rows=rows,
        pair_cols=cols,
        pair_counts=pcounts,
        word_total=int(word_total),
        skip_total=int(skip_total),
        n_types=len(words),
        params=params,
    )


def ppmi_matrix(
    counts: CooccurrenceCounts,
    *,
    shift: float = 1.0,
    alpha: float = 0.75,
    min_count: int = 5,
) -> csr_matrix:
    """Build the PPMI matrix from exact counts, in O(nnz).

    With ``PMI(a, b) = log(P(a, b) / (P(a)**alpha * P(b)))``,
    ``P(a, b) = n(a, b) / skip_total`` and ``P(w) = n(w) / word_total``, a cell
    is stored when ``PMI(a, b) > 0`` *and* ``PMI(a, b) - log(shift) > 0``,
    with that shifted value. For ``shift >= 1`` this is exactly
    ``max(0, PMI - log(shift))``. For ``shift < 1``, cells with
    ``PMI <= 0`` stay empty even though ``PMI - log(shift)`` would be
    positive. This rule matches the sketch kernel.

    Context-distribution smoothing (``alpha``) is applied to the focus word
    ``a`` only. That asymmetry is intentional and matches the sketch kernel.
    Each probability carries a ``1e-10`` epsilon, as in the sketch kernel.

    Args:
        counts: Output of :func:`count_cooccurrences`.
        shift: PMI shift; ``log(shift)`` is subtracted. Must be > 0.
            Default 1.0 (no shift).
        alpha: Smoothing exponent on the focus word's probability. Default 0.75.
        min_count: A word and a pair both need a count strictly greater than
            this to receive an entry. Default 5.

    Returns:
        A ``(V, V)`` float64 :class:`scipy.sparse.csr_matrix` of strictly
        positive entries; unobserved and non-positive cells are absent.

    Raises:
        ValueError: If ``shift`` is not positive, or the corpus has no
            skip-grams at all (every line has fewer than two kept tokens).

    Examples:
        Hand-computed: six "cat dog" lines give P(cat) = P(dog) = 1/2 and
        P(cat, dog) = 1, so PPMI(cat, dog) = log(1 / (0.5**0.75 * 0.5))
        = 1.75 * log(2).

        >>> m = ppmi_matrix(count_cooccurrences(["cat dog"] * 6, min_count=1))
        >>> round(float(m[0, 1]), 6), round(1.75 * float(np.log(2)), 6)
        (1.213008, 1.213008)
        >>> m.nnz
        1

    """
    if counts.skip_total == 0:
        raise ValueError(
            "The corpus has no skip-grams: every line has fewer than two tokens "
            "after the min_word_length filter, so there is no co-occurrence to "
            "weight."
        )
    data, rows, cols = _core.ppmi_from_counts(
        counts.pair_rows,
        counts.pair_cols,
        counts.pair_counts,
        counts.word_counts,
        float(counts.word_total),
        float(counts.skip_total),
        shift=shift,
        alpha=alpha,
        min_count=min_count,
    )
    n = len(counts.vocabulary)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
