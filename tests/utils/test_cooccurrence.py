"""Contract tests for exact two-pass counting and PPMI from exact counts."""

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from chronowords import _core
from chronowords.utils.cooccurrence import CooccurrenceCounts
from chronowords.utils.cooccurrence import count_cooccurrences
from chronowords.utils.cooccurrence import ppmi_matrix
from tests import reference
from tests.reference import skipgrams


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
    """Pure-Python reference: str.split, len filter, nltk-style skip-grams."""
    toks = [[w for w in line.split() if len(w) >= min_len] for line in lines]
    words = Counter(w for t in toks for w in t)
    pairs = Counter(p for t in toks for p in skipgrams(t, window))
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
    """``ppmi_matrix`` equals a NumPy implementation of the published formula
    (tests/reference.py), cell by cell, including the marginals.
    """
    assume(any(len(w) >= 3 for line in lines for w in line.split()))
    c = count_cooccurrences(lines, min_count=1)
    if int(c.col_sums.sum()) == 0:
        with pytest.raises(ValueError, match="no skip-gram"):
            ppmi_matrix(c)
        return
    m = ppmi_matrix(c, shift=shift, alpha=alpha, min_count=ppmi_min).toarray()
    pairs = dict(
        zip(
            zip(c.pair_rows.tolist(), c.pair_cols.tolist(), strict=True),
            c.pair_counts.tolist(),
            strict=True,
        )
    )
    # the stored marginals are the full matrix's (nothing pruned here)
    dense = np.zeros_like(m)
    for (i, j), n in pairs.items():
        dense[i, j] = n
    assert c.row_sums.tolist() == dense.sum(axis=1).astype(int).tolist()
    assert c.col_sums.tolist() == dense.sum(axis=0).astype(int).tolist()

    expected = reference.ppmi(
        pairs, len(c.vocabulary), alpha=alpha, shift=shift, min_count=ppmi_min
    )
    assert (m >= 0).all()
    np.testing.assert_allclose(m, expected, rtol=1e-10, atol=1e-12)


def test_matrix_is_the_pair_counts():
    """``CooccurrenceCounts.matrix()`` is the COO data as a sparse matrix."""
    c = count_cooccurrences(["cat dog fish"] * 2, min_count=1, window_size=0)
    assert c.vocabulary == ["cat", "dog", "fish"]
    assert c.matrix().toarray().tolist() == [[0, 2, 0], [0, 0, 2], [0, 0, 0]]


def test_params_record_the_corpus(tmp_path):
    """The counts carry the parameters and a description of the corpus."""
    f = tmp_path / "c.txt"
    f.write_bytes(b"cat dog\n")  # write_text would add \r on Windows
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


# --- tokenizer options -------------------------------------------------------

RAW = [
    "The Cat, the dog!  It's 3.14 o'clock.",
    "THE DOG; the cat — élő ÁRVÍZTŰRŐ tükörfúrógép.",
] * 3


@given(text=st.text(max_size=60))
def test_whitespace_tokenizer_is_str_split(text):
    """The default tokenizer is exactly ``str.split()`` for any Unicode text."""
    assert _core.tokenize(text, 0) == text.split()


@given(
    text=st.text(
        alphabet=st.sampled_from("abcABCéÉőŐ\u03c3\u03a3\u03c2 ,.!'-\t\n0123"),
        max_size=60,
    )
)
def test_unicode_tokens_are_clean_and_lowercase_is_idempotent(text):
    """Unicode tokens contain no whitespace; lowercasing twice changes nothing."""
    toks = _core.tokenize(text, 0, "unicode", True)
    assert all(t and t.split() == [t] for t in toks)
    assert _core.tokenize(" ".join(toks), 0, "whitespace", True) == toks


def test_unicode_tokenizer_example():
    """Punctuation is dropped; contractions and decimals stay whole."""
    assert _core.tokenize("The Cat, it's 3.14 — ÁRVÍZTŰRŐ!", 0, "unicode", True) == [
        "the",
        "cat",
        "it's",
        "3.14",
        "árvíztűrő",
    ]


def test_unicode_tokenizer_segments_scripts_without_spaces():
    """Chinese gets dictionary words, not single characters; Thai is segmented
    without losing or duplicating any text.
    """
    assert _core.tokenize("我们今天去北京大学学习。", 0, "unicode") == [
        "我们",
        "今天",
        "去",
        "北京",
        "大学",
        "学习",
    ]
    thai = _core.tokenize("วันนี้อากาศดีมาก", 0, "unicode")
    assert "".join(thai) == "วันนี้อากาศดีมาก"
    assert "อากาศ" in thai and len(thai) > 2


def test_tokenizer_options_agree_across_input_forms(tmp_path):
    """Files, lists and generators tokenise identically under every option."""
    f = tmp_path / "raw.txt"
    f.write_text("\n".join(RAW) + "\n", encoding="utf-8")
    for tokenizer in ("whitespace", "unicode"):
        for lowercase in (False, True):
            kw: dict[str, Any] = {
                "tokenizer": tokenizer,
                "lowercase": lowercase,
                "min_count": 1,
            }
            expected = as_tuple(count_cooccurrences(RAW, **kw))
            assert as_tuple(count_cooccurrences(f, **kw)) == expected
            assert as_tuple(count_cooccurrences(iter(RAW), **kw)) == expected


def test_unicode_tokenizer_changes_the_vocabulary():
    """The option is not cosmetic: punctuation-attached forms merge."""
    ws = count_cooccurrences(RAW, min_count=1).vocabulary
    uni = count_cooccurrences(RAW, tokenizer="unicode", lowercase=True, min_count=1)
    assert "dog!" in ws and "Cat," in ws
    assert "dog!" not in uni.vocabulary and "cat" in uni.vocabulary
    assert uni.params["tokenizer"] == "unicode" and uni.params["lowercase"] is True


def test_pretokenised_input_requires_whitespace_tokenizer():
    """Token lists are used verbatim, never re-segmented."""
    with pytest.raises(ValueError, match="verbatim"):
        count_cooccurrences([["e-mail", "can't"]], tokenizer="unicode", min_count=1)
    c = count_cooccurrences([["e-mail", "CAN'T"]] * 2, lowercase=True, min_count=1)
    assert c.vocabulary == ["can't", "e-mail"]


def test_unknown_tokenizer_is_rejected():
    """A typo in ``tokenizer`` fails before counting."""
    with pytest.raises(ValueError, match="tokenizer"):
        count_cooccurrences(["cat dog"], tokenizer="bpe", min_count=1)  # ty: ignore[invalid-argument-type]


# --- parallel file counting, pruning, wide keys -------------------------------


@given(
    text=raw_text_st,
    chunk=st.integers(1, 40),
    threads=st.integers(1, 4),
    window=st.integers(0, 3),
)
@COOC
def test_parallel_file_counting_ignores_chunking_and_threads(
    text, chunk, threads, window, tmp_path_factory
):
    """Cutting files into (tiny) chunks and counting on several threads gives
    exactly the sequential counts of the same lines.
    """
    path = tmp_path_factory.mktemp("par") / "t.txt"
    path.write_bytes((text + "\ncat dog cat\n").encode("utf-8"))
    with path.open(encoding="utf-8") as fh:
        lines = list(fh)

    words, counts, total = _core.count_words_files(
        [str(path)], 0, n_threads=threads, chunk_bytes=chunk
    )
    e_words, e_counts, e_total = _core.count_words(lines, 0)
    assert (words, counts.tolist(), total) == (e_words, e_counts.tolist(), e_total)

    got = _core.count_pairs_files(
        [str(path)], words, window, 0, n_threads=threads, chunk_bytes=chunk
    )
    exp = _core.count_pairs(lines, words, window, 0)
    assert [a.tolist() for a in got[:5]] + [got[5]] == [a.tolist() for a in exp[:5]] + [
        exp[5]
    ]


@given(lines=lines_st, min_pair=st.integers(1, 4))
@COOC
def test_min_pair_count_prunes_without_touching_totals_or_ppmi(lines, min_pair):
    """Pruning keeps exactly the pairs seen >= min_pair_count times; totals and
    PPMI cells above the threshold are unchanged.
    """
    assume(any(len(w) >= 3 for line in lines for w in line.split()))
    full = count_cooccurrences(lines, min_count=1)
    pruned = count_cooccurrences(lines, min_count=1, min_pair_count=min_pair)

    keep = full.pair_counts >= min_pair
    assert pruned.pair_rows.tolist() == full.pair_rows[keep].tolist()
    assert pruned.pair_cols.tolist() == full.pair_cols[keep].tolist()
    assert pruned.pair_counts.tolist() == full.pair_counts[keep].tolist()
    assert (pruned.word_total, pruned.skip_total) == (full.word_total, full.skip_total)
    assert pruned.row_sums.tolist() == full.row_sums.tolist()
    assert pruned.col_sums.tolist() == full.col_sums.tolist()
    assert pruned.params["min_pair_count"] == min_pair
    if int(full.col_sums.sum()):
        np.testing.assert_array_equal(
            ppmi_matrix(pruned, min_count=min_pair - 1).toarray(),
            ppmi_matrix(full, min_count=min_pair - 1).toarray(),
        )


def test_vocabularies_beyond_65536_words_use_wide_keys(tmp_path):
    """Pair keys switch from 32 to 64 bits above 65,536 words; ids >= 65,536
    must survive the round trip.
    """
    vocab = [f"w{i}" for i in range(70_000)]
    lines = ["w0 w69999 w65536", "w65536 w1 w69999"]
    f = tmp_path / "wide.txt"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    expected = Counter(p for line in lines for p in skipgrams(line.split(), 1))
    for rows, cols, counts, *_ in (
        _core.count_pairs(lines, vocab, 1, 0),
        _core.count_pairs_files([str(f)], vocab, 1, 0, chunk_bytes=5),
    ):
        got = {
            (vocab[r], vocab[c]): n
            for r, c, n in zip(
                rows.tolist(), cols.tolist(), counts.tolist(), strict=True
            )
        }
        assert got == dict(expected)


def test_svdalgebra_keeps_only_pairs_that_can_enter_ppmi():
    """SVDAlgebra prunes stored pairs at its PPMI threshold; PPMI is unaffected."""
    from chronowords.algebra.svd import SVDAlgebra

    lines = ["cat dog fish", "cat dog bird", "fish bird cow"] * 4
    model = SVDAlgebra(n_components=2, min_count=1, ppmi_min_count=5)
    model.train(lines)
    assert model.counts is not None
    assert model.counts.pair_counts.min() > 5
    assert model.counts.params["min_pair_count"] == 6
    full = count_cooccurrences(lines, min_count=1)
    assert model.ppmi is not None
    np.testing.assert_array_equal(
        model.ppmi.toarray(), ppmi_matrix(full, min_count=5).toarray()
    )


def test_n_threads_is_validated():
    """``n_threads`` must be a positive integer or None."""
    with pytest.raises(ValueError, match="n_threads"):
        count_cooccurrences(["cat dog"], min_count=1, n_threads=0)


# --- bounded memory -------------------------------------------------------------


@given(
    lines=lines_st,
    budget=st.sampled_from([2048, 16_384, 1 << 20]),
    min_pair=st.integers(2, 4),
    window=st.integers(0, 3),
)
@COOC
def test_max_memory_gives_identical_counts(lines, budget, min_pair, window):
    """The bounded path's counts equal the exact path's for every kept word and
    pair, whatever the budget: the sketch only decides what is counted.
    """
    assume(any(len(w) >= 3 for line in lines for w in line.split()))
    kw: dict[str, Any] = {
        "min_count": 1,
        "min_pair_count": min_pair,
        "window_size": window,
    }
    exact = count_cooccurrences(lines, **kw)
    try:
        bounded = count_cooccurrences(lines, max_memory=budget, **kw)
    except MemoryError:
        # only legitimate when the exact candidates cannot fit in half the budget
        assert budget <= 16_384
        return
    assert as_tuple(bounded)[:7] == as_tuple(exact)[:7]
    assert bounded.row_sums.tolist() == exact.row_sums.tolist()
    assert bounded.col_sums.tolist() == exact.col_sums.tolist()
    assert bounded.n_types is None
    mem = bounded.params["memory"]
    assert mem["pairs"]["kept"] == len(exact.pair_counts)
    assert mem["pairs"]["candidates"] >= mem["pairs"]["kept"]


def test_max_memory_files_and_generators_agree(tmp_path):
    """Files are read directly; other input is spilled first. Same counts."""
    f = tmp_path / "c.txt"
    f.write_text("\n".join(RAW) + "\n", encoding="utf-8")
    kw: dict[str, Any] = {"min_count": 1, "min_pair_count": 2, "max_memory": "64KB"}
    a = as_tuple(count_cooccurrences(f, **kw))
    assert as_tuple(count_cooccurrences(RAW, **kw)) == a
    assert as_tuple(count_cooccurrences(iter(RAW), **kw)) == a
    assert as_tuple(count_cooccurrences([line.split() for line in RAW], **kw)) == a


def test_max_memory_too_small_raises():
    """Candidates that cannot fit raise MemoryError instead of silently degrading."""
    lines = [" ".join(f"w{i}x w{j}y" for j in range(30)) for i in range(60)]
    with pytest.raises(MemoryError, match="raise max_memory"):
        count_cooccurrences(lines, min_count=1, min_pair_count=1, max_memory=4096)


@pytest.mark.parametrize("bad", ["lots", "-1GB", "0", 0, "5 parsecs"])
def test_max_memory_must_be_a_size(bad):
    """Unreadable or non-positive sizes are rejected before any counting."""
    with pytest.raises(ValueError):
        count_cooccurrences(["cat dog"], min_count=1, max_memory=bad)
