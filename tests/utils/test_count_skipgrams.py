from collections import Counter

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from nltk.util import skipgrams

from chronowords.utils.count_skipgrams import PPMIComputer  # ty: ignore
from chronowords.utils.probabilistic_counter import CountMinSketch
from tests.strategies import ppmi_inputs


@pytest.fixture
def ppmi_computer():
    """Create a PPMIComputer instance with toy corpus data."""
    rng = np.random.default_rng(0)
    skipgram_counts = rng.integers(0, 10, size=(3, 10), dtype=np.int32)
    word_counts = rng.integers(0, 10, size=(3, 10), dtype=np.int32)
    seeds = [12345, 67890, 11111]

    return PPMIComputer(
        skipgram_counts=skipgram_counts,
        word_counts=word_counts,
        vocabulary=["word1", "word2", "word3"],
        seeds=seeds,
        width=10,
        skip_total=6.0,
        word_total=3.0,
        shift=1.0,
        alpha=0.75,
    )


def test_ppmi_matrix_positive_values(ppmi_computer):
    """Test that PPMI matrix values are non-negative."""
    ppmi_matrix = ppmi_computer.compute_ppmi_matrix_with_sketch(batch_size=2)
    assert np.all(ppmi_matrix.data >= 0)


def test_ppmi_empty_vocabulary():
    """Test PPMI computation with empty vocabulary."""
    counts = np.zeros((3, 10), dtype=np.int32)
    computer = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=[],
        seeds=[1, 2, 3],
        width=10,
        skip_total=0.0,
        word_total=0.0,
    )
    matrix = computer.compute_ppmi_matrix_with_sketch()
    assert matrix.shape == (0, 0)


def test_ppmi_single_word():
    """Test PPMI computation with single word vocabulary."""
    counts = np.array([[10], [10], [10]], dtype=np.int32)
    computer = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=["only"],
        seeds=[1, 2, 3],
        width=1,
        skip_total=10.0,
        word_total=10.0,
    )
    matrix = computer.compute_ppmi_matrix_with_sketch()
    assert matrix.shape == (1, 1)


def test_ppmi_all_zero_counts():
    """Test PPMI with all-zero count arrays."""
    counts = np.zeros((3, 10), dtype=np.int32)
    computer = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=["a", "b", "c"],
        seeds=[1, 2, 3],
        width=10,
        skip_total=1.0,
        word_total=1.0,
    )
    matrix = computer.compute_ppmi_matrix_with_sketch()
    assert matrix.nnz == 0


def test_ppmi_batch_boundaries():
    """Test PPMI batch processing at exact boundaries."""
    vocab_size = 4
    counts = np.zeros((3, 100), dtype=np.int32)
    counts[0, :vocab_size] = 10
    vocabulary = [f"w{i}" for i in range(vocab_size)]

    computer = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=vocabulary,
        seeds=[1, 2, 3],
        width=100,
        skip_total=40.0,
        word_total=40.0,
    )

    m1 = computer.compute_ppmi_matrix_with_sketch(batch_size=4)
    assert m1.shape == (vocab_size, vocab_size)

    computer2 = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=vocabulary,
        seeds=[1, 2, 3],
        width=100,
        skip_total=40.0,
        word_total=40.0,
    )
    m2 = computer2.compute_ppmi_matrix_with_sketch(batch_size=100)
    assert m2.shape == (vocab_size, vocab_size)


def test_ppmi_invalid_seeds():
    """Test that non-integer seeds raise ValueError."""
    counts = np.zeros((3, 10), dtype=np.int32)
    with pytest.raises(ValueError, match="All seeds must be integers"):
        PPMIComputer(
            skipgram_counts=counts,
            word_counts=counts,
            vocabulary=["a"],
            seeds=[1.5, 2.0, 3.0],
            width=10,
            skip_total=1.0,
            word_total=1.0,
        )


@given(inputs=ppmi_inputs())
@settings(deadline=None, max_examples=60)
def test_ppmi_values_are_non_negative(inputs):
    """Every PPMI entry is non-negative for any valid count inputs.

    PPMI keeps only strictly-positive PMI values, so the resulting sparse
    matrix must never contain a negative entry. Non-negativity is a hard
    precondition for the downstream NMF topic model.
    """
    computer = PPMIComputer(**inputs)
    matrix = computer.compute_ppmi_matrix_with_sketch()
    n = len(inputs["vocabulary"])
    assert matrix.shape == (n, n)
    assert np.all(matrix.data >= 0)


def _sketches(word_events, pair_events, width=1000, depth=3):
    """Fill a word sketch and a skip-gram sketch the way ``SVDAlgebra`` does."""
    words = CountMinSketch(width=width, depth=depth, seed=42)
    pairs = CountMinSketch(width=width, depth=depth, seed=42, track_keys=False)
    for w in word_events:
        words.update(w)
    for a, b in pair_events:
        pairs.update(f"{a}#{b}")
    return words, pairs


def _computer(words, pairs, vocabulary):
    return PPMIComputer(
        skipgram_counts=pairs.counts,
        word_counts=words.counts,
        vocabulary=vocabulary,
        seeds=words.seeds,
        width=words.width,
        skip_total=float(pairs.total),
        word_total=float(words.total),
        shift=1.0,
        alpha=0.75,
    )


def test_ppmi_golden_cell_uses_word_marginals():
    """Hand-computed PPMI cell; marginals must come from the *word* sketch.

    Six lines of ``"cat dog"``: word counts cat=6, dog=6 (total 12), one
    skip-gram ``cat#dog`` per line (total 6). So P(cat)=P(dog)=1/2 and
    P(cat,dog)=1. With alpha=0.75 on the focus word and shift=1:

        PPMI = log(1 / (0.5**0.75 * 0.5)) = log(2**1.75) = 1.75 * log(2)

    Regression: the kernel used to read word counts from the skip-gram
    sketch, where single words are never inserted, so every marginal was 0
    and the matrix came out empty.
    """
    words, pairs = _sketches(["cat", "dog"] * 6, [("cat", "dog")] * 6)
    matrix = _computer(words, pairs, ["cat", "dog"]).compute_ppmi_matrix_with_sketch()

    assert matrix.nnz == 1
    assert matrix[0, 1] == pytest.approx(1.75 * np.log(2), abs=1e-8)
    assert matrix[1, 0] == 0  # skip-grams are directional; no dog#cat


def test_ppmi_matches_exact_count_reference():
    """Kernel output equals an exact-count Python implementation of the formula.

    On a small corpus with a wide sketch there are no collisions among the
    queried keys, so the Count-Min estimates are exact and the two must
    agree cell for cell.
    """
    lines = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat chased the dog",
        "a dog and a cat met on the mat",
    ] * 8
    window = 5
    word_events, pair_events = [], []
    for line in lines:
        toks = [w for w in line.split() if len(w) >= 3]
        word_events += toks
        pair_events += list(skipgrams(toks, 2, window))
    words, pairs = _sketches(word_events, pair_events, width=100_000, depth=5)
    vocab = sorted(set(word_events))

    matrix = _computer(words, pairs, vocab).compute_ppmi_matrix_with_sketch()

    wc, pc = Counter(word_events), Counter(pair_events)
    eps, wt, st_ = 1e-10, len(word_events), len(pair_events)
    expected = np.zeros((len(vocab), len(vocab)))
    for i, a in enumerate(vocab):
        for j, b in enumerate(vocab):
            if wc[a] <= 5 or wc[b] <= 5 or pc[(a, b)] <= 5:
                continue
            pa = (wc[a] / wt + eps) ** 0.75
            pb = wc[b] / wt + eps
            pmi = np.log(((pc[(a, b)] + eps) / st_) / (pa * pb))
            expected[i, j] = max(pmi, 0.0)

    assert matrix.nnz > 0
    np.testing.assert_allclose(matrix.toarray(), expected, atol=1e-9)
