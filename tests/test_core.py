"""Contract tests for the compiled core, ``chronowords._core``.

The Rust kernels must reproduce the Python/Cython semantics they replace:
``str.split()`` + ``len()`` tokenisation, ``nltk.util.skipgrams`` pair
enumeration (via ``tests.reference``), and the PPMI formula.
"""

from collections import Counter
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from chronowords import _core
from tests.reference import NLTK_9_TOKENS_K2
from tests.reference import skipgrams


# Text drawn from a small alphabet plus every kind of separator Python's
# str.split() honours, including U+001C..U+001F which Rust's is_whitespace
# does not treat as whitespace.
SEPARATORS = " \t\n\x0b\x0c\r\x1c\x1d\x1e\x1f\x85\xa0\u2003\u3000"
lines_strategy = st.lists(
    st.text(alphabet=st.sampled_from("abcé" + SEPARATORS), max_size=40),
    max_size=8,
)


def py_tokens(line: str, min_len: int) -> list[str]:
    """Python reference tokenisation: ``str.split()`` filtered by ``len``."""
    return [w for w in line.split() if len(w) >= min_len]


@given(lines=lines_strategy, min_len=st.integers(0, 4))
def test_count_words_matches_python_split(lines, min_len):
    """Rust word counts equal ``Counter`` over ``str.split()`` tokens."""
    words, counts, total = _core.count_words(lines, min_len)
    expected = Counter(w for line in lines for w in py_tokens(line, min_len))
    assert dict(zip(words, counts.tolist(), strict=True)) == dict(expected)
    assert total == sum(expected.values())
    # descending count, ties alphabetical
    assert [(-c, w) for w, c in zip(words, counts, strict=True)] == sorted(
        (-c, w) for w, c in zip(words, counts, strict=True)
    )


@given(
    lines=lines_strategy,
    window=st.integers(0, 6),
    min_len=st.integers(0, 3),
    data=st.data(),
)
@settings(deadline=None)
def test_count_pairs_matches_nltk_skipgrams(lines, window, min_len, data):
    """Rust pair counts equal ``nltk.util.skipgrams`` restricted to the vocabulary."""
    all_words = sorted({w for line in lines for w in py_tokens(line, min_len)})
    vocab = (
        data.draw(st.lists(st.sampled_from(all_words), unique=True))
        if all_words
        else []
    )

    rows, cols, counts, total = _core.count_pairs(lines, vocab, window, min_len)

    expected: Counter = Counter()
    expected_total = 0
    for line in lines:
        for a, b in skipgrams(py_tokens(line, min_len), window):
            expected_total += 1
            if a in vocab and b in vocab:
                expected[(vocab.index(a), vocab.index(b))] += 1
    got = {
        (r, c): n
        for r, c, n in zip(rows.tolist(), cols.tolist(), counts.tolist(), strict=True)
    }
    assert got == dict(expected)
    assert total == expected_total
    keys = list(zip(rows.tolist(), cols.tolist(), strict=True))
    assert keys == sorted(keys)


@given(a=lines_strategy, b=lines_strategy, window=st.integers(0, 4))
@settings(deadline=None)
def test_count_pairs_is_additive_over_line_shards(a, b, window):
    """Counting two shards and summing equals counting their concatenation."""
    vocab = sorted({w for line in a + b for w in line.split()})

    def as_dict(lines):
        r, c, n, total = _core.count_pairs(lines, vocab, window, 0)
        return Counter(
            dict(zip(zip(r.tolist(), c.tolist(), strict=True), n.tolist(), strict=True))
        ), total

    (ca, ta), (cb, tb), (cab, tab) = as_dict(a), as_dict(b), as_dict(a + b)
    assert ca + cb == cab
    assert ta + tb == tab


def test_ppmi_from_counts_golden_cell():
    """Six ``cat dog`` lines: PPMI(cat, dog) = 1.75 * ln 2 (hand-computed)."""
    data, rows, cols = _core.ppmi_from_counts(
        np.array([0], dtype=np.int32),
        np.array([1], dtype=np.int32),
        np.array([6]),
        np.array([6, 6]),
        12.0,
        6.0,
    )
    assert rows.tolist() == [0] and cols.tolist() == [1]
    # the formula's 1e-10 epsilons move the value by ~3e-10
    assert data[0] == pytest.approx(1.75 * np.log(2), abs=1e-8)


def test_count_pairs_rejects_duplicate_vocabulary():
    """A duplicated vocabulary word would make matrix rows ambiguous."""
    with pytest.raises(ValueError, match="duplicates"):
        _core.count_pairs(["a b"], ["a", "a"], 2, 0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"word_total": 0.0}, "word_total"),
        ({"skip_total": 0.0}, "skip_total"),
        ({"shift": 0.0}, "shift"),
        ({"word_counts": np.array([6])}, "out of range"),
    ],
)
def test_ppmi_from_counts_fails_loudly(kwargs, match):
    """Degenerate totals and out-of-range indices raise ``ValueError``."""
    args: dict[str, Any] = {
        "rows": np.array([0], dtype=np.int32),
        "cols": np.array([1], dtype=np.int32),
        "pair_counts": np.array([6]),
        "word_counts": np.array([6, 6]),
        "word_total": 12.0,
        "skip_total": 6.0,
    }
    with pytest.raises(ValueError, match=match):
        _core.ppmi_from_counts(**{**args, **kwargs})


def test_reference_skipgrams_match_recorded_nltk_output():
    """The test reference reproduces nltk's skip-gram enumeration exactly."""
    assert skipgrams("a b c d e f g h i".split(), 2) == NLTK_9_TOKENS_K2
