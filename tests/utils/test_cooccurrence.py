"""Contract tests for exact two-pass counting and PPMI from exact counts."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from nltk.util import skipgrams

from chronowords.utils.cooccurrence import CooccurrenceCounts
from chronowords.utils.cooccurrence import count_cooccurrences
from chronowords.utils.cooccurrence import ppmi_matrix


WORDS = ["cat", "dog", "fish", "bird", "cow", "élő", "ég"]
lines_st = st.lists(
    st.lists(st.sampled_from(WORDS), min_size=0, max_size=9).map(" ".join),
    min_size=1,
    max_size=25,
)
# Raw text including the line-break characters that universal newlines and
# spilling must handle.
raw_text_st = st.lists(st.sampled_from(["cat", "dog", " ", "\n", "\r", "\t", "é"])).map(
    "".join
)
COOC = settings(
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def as_tuple(c: CooccurrenceCounts):
    """Everything that defines the counts, minus the corpus description."""
    return (
        c.vocabulary,
        c.word_counts.tolist(),
        c.pair_rows.tolist(),
        c.pair_cols.tolist(),
        c.pair_counts.tolist(),
        c.word_total,
        c.skip_total,
        c.n_types,
    )


def reference_counts(lines, window, min_len, vocab):
    """Pure-Python reference: str.split, len filter, nltk skipgrams."""
    toks = [[w for w in line.split() if len(w) >= min_len] for line in lines]
    words = Counter(w for t in toks for w in t)
    pairs = Counter(p for t in toks for p in skipgrams(t, 2, window))
    return words, pairs


@given(lines=lines_st, window=st.integers(0, 4), min_len=st.integers(0, 3))
@COOC
def test_counts_match_reference(lines, window, min_len):
    """Exact counts equal a Counter over ``str.split`` + ``nltk.skipgrams``."""
    assume(any(len(w) >= min_len for line in lines for w in line.split()))
    c = count_cooccurrences(
        lines, window_size=window, min_word_length=min_len, min_count=1
    )
    words, pairs = reference_counts(lines, window, min_len, c.vocabulary)

    assert c.word_total == sum(words.values())
    assert c.skip_total == sum(pairs.values())
    assert dict(zip(c.vocabulary, c.word_counts.tolist(), strict=True)) == dict(words)
    got = Counter(
        {
            (c.vocabulary[r], c.vocabulary[k]): n
            for r, k, n in zip(c.pair_rows, c.pair_cols, c.pair_counts, strict=True)
        }
    )
    assert got == pairs


@given(lines=lines_st, vocab_size=st.integers(1, 8), min_count=st.integers(1, 6))
@COOC
def test_vocabulary_rule(lines, vocab_size, min_count):
    """Top ``vocab_size`` words with count >= ``min_count``; ties alphabetical."""
    words = Counter(w for line in lines for w in line.split() if len(w) >= 3)
    expected = sorted(
        (w for w, n in words.items() if n >= min_count), key=lambda w: (-words[w], w)
    )[:vocab_size]
    if not expected:
        with pytest.raises(ValueError, match="Empty vocabulary"):
            count_cooccurrences(lines, vocab_size=vocab_size, min_count=min_count)
        return
    c = count_cooccurrences(lines, vocab_size=vocab_size, min_count=min_count)
    assert c.vocabulary == expected
    assert c.word_counts.tolist() == [words[w] for w in expected]
    # totals count every token, in vocabulary or not
    assert c.word_total == sum(words.values())


@given(lines=lines_st)
@COOC
def test_input_forms_agree(lines, tmp_path_factory):
    """List, tuple, generator, token lists, one file and several files all
    produce identical counts.
    """
    assume(any(line.split() for line in lines))
    tmp = tmp_path_factory.mktemp("forms")
    one = tmp / "all.txt"
    one.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    half = len(lines) // 2
    a, b = tmp / "a.txt", tmp / "b.txt"
    a.write_text("".join(line + "\n" for line in lines[:half]), encoding="utf-8")
    b.write_text("".join(line + "\n" for line in lines[half:]), encoding="utf-8")

    def run(corpus):
        return as_tuple(count_cooccurrences(corpus, min_word_length=0, min_count=1))

    expected = run(lines)
    assert run(tuple(lines)) == expected
    assert run(line for line in lines) == expected
    assert run([line.split() for line in lines]) == expected
    assert run(line.split() for line in lines) == expected
    assert run(one) == expected
    assert run([a, b]) == expected


@given(text=raw_text_st)
@COOC
def test_file_reading_matches_python_text_mode(text, tmp_path_factory):
    """Reading a file in the core equals iterating ``open(path)`` in Python
    (UTF-8, universal newlines), and spilling a generator round-trips.
    """
    path = tmp_path_factory.mktemp("raw") / "raw.txt"
    path.write_bytes(text.encode("utf-8"))
    with path.open(encoding="utf-8") as fh:
        py_lines = list(fh)
    if not any(len(w) >= 3 for line in py_lines for w in line.split()):
        return

    from_file = as_tuple(count_cooccurrences(path, min_count=1))
    assert from_file == as_tuple(count_cooccurrences(py_lines, min_count=1))
    # generator items may contain '\n'/'\r'; spilling must keep them in one line
    from_gen = as_tuple(count_cooccurrences((line for line in [text]), min_count=1))
    assert from_gen == as_tuple(count_cooccurrences([text], min_count=1))


@given(
    lines=lines_st,
    shift=st.sampled_from([0.5, 1.0, 2.0]),
    alpha=st.sampled_from([0.75, 1.0]),
    ppmi_min=st.integers(0, 3),
)
@COOC
def test_ppmi_matches_reference_formula(lines, shift, alpha, ppmi_min):
    """``ppmi_matrix`` equals the documented formula, cell by cell."""
    assume(any(len(w) >= 3 for line in lines for w in line.split()))
    c = count_cooccurrences(lines, min_count=1)
    if c.skip_total == 0:
        with pytest.raises(ValueError, match="no skip-grams"):
            ppmi_matrix(c)
        return
    m = ppmi_matrix(c, shift=shift, alpha=alpha, min_count=ppmi_min).toarray()

    eps = 1e-10
    wc = dict(zip(c.vocabulary, c.word_counts.tolist(), strict=True))
    expected = np.zeros_like(m)
    for r, k, n in zip(c.pair_rows, c.pair_cols, c.pair_counts, strict=True):
        a, b = c.vocabulary[r], c.vocabulary[k]
        if n <= ppmi_min or wc[a] <= ppmi_min or wc[b] <= ppmi_min:
            continue
        pa = (wc[a] / c.word_total + eps) ** alpha
        pb = wc[b] / c.word_total + eps
        pab = (n + eps) / c.skip_total
        pmi = np.log(pab / (pa * pb))
        # documented rule: PMI must be positive before the shift is applied
        if pmi > 0:
            expected[r, k] = max(0.0, pmi - np.log(shift))

    assert (m >= 0).all()
    np.testing.assert_allclose(m, expected, rtol=1e-12, atol=1e-15)


def test_matrix_is_the_pair_counts():
    """``CooccurrenceCounts.matrix()`` is the COO data as a sparse matrix."""
    c = count_cooccurrences(["cat dog fish"] * 2, min_count=1, window_size=0)
    assert c.vocabulary == ["cat", "dog", "fish"]
    assert c.matrix().toarray().tolist() == [[0, 2, 0], [0, 0, 2], [0, 0, 0]]


def test_params_record_the_corpus(tmp_path):
    """The counts carry the parameters and a description of the corpus."""
    f = tmp_path / "c.txt"
    f.write_text("cat dog\n", encoding="utf-8")
    c = count_cooccurrences(f, min_count=1)
    assert c.params["corpus"] == {
        "kind": "files",
        "files": [{"path": str(f), "bytes": 8}],
    }
    assert c.params["window_size"] == 5
    assert c.params["min_count"] == 1
    assert count_cooccurrences(["cat dog"], min_count=1).params["corpus"] == {
        "kind": "list"
    }


@pytest.mark.parametrize(
    ("corpus", "exc", "match"),
    [
        ("cat dog", TypeError, "single string"),
        (b"cat dog", TypeError, "single string"),
        ([["cat", "big dog"]], ValueError, "contains whitespace"),
        ([["cat", ""]], ValueError, "empty"),
        ([["cat", 3]], TypeError, None),
        ([3], TypeError, "str or a sequence"),
    ],
)
def test_bad_corpus_fails_loudly(corpus, exc, match):
    """Malformed corpora raise instead of being silently re-tokenised."""
    with pytest.raises(exc, match=match):
        count_cooccurrences(corpus, min_count=1)


def test_bad_files_fail_loudly(tmp_path):
    """Missing files raise OSError; invalid UTF-8 raises ValueError."""
    with pytest.raises(OSError):
        count_cooccurrences(tmp_path / "missing.txt", min_count=1)
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"cat dog\n\xff\xfe\n")
    with pytest.raises(ValueError, match="invalid UTF-8"):
        count_cooccurrences(bad, min_count=1)


@pytest.mark.parametrize(
    "kwargs",
    [{"window_size": -1}, {"min_word_length": -1}, {"vocab_size": 0}, {"min_count": 0}],
)
def test_parameters_are_validated(kwargs):
    """Out-of-range parameters raise ``ValueError`` before any counting."""
    with pytest.raises(ValueError):
        count_cooccurrences(["cat dog"], **kwargs)


def test_spill_file_is_removed(tmp_path, monkeypatch):
    """The temporary copy of a single-use iterable does not outlive the call."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", None)
    count_cooccurrences((line for line in ["cat dog"] * 3), min_count=1)
    assert list(Path(tmp_path).iterdir()) == []
