"""Hypothesis strategies for chronowords property-based tests.

Each strategy models the valid input space for a function or group of
functions, so the property tests exercise realistic inputs rather than random
noise.
"""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hynp


# -- Atomic strategies --

# A fixed pool of realistic tokens used to build corpora and vocabularies.
_WORD_POOL = [
    "king",
    "queen",
    "palace",
    "royal",
    "throne",
    "kingdom",
    "crown",
    "prince",
    "man",
    "woman",
    "child",
    "family",
    "home",
    "army",
    "battle",
    "victory",
]


@st.composite
def trainable_corpus(draw):
    """Build a corpus that reliably trains an SVDAlgebra model.

    Builds a handful of sentences from a shared vocabulary and repeats them
    enough times that words and skip-grams clear the internal count>5 PPMI
    thresholds, guaranteeing a non-empty vocabulary (so ``train`` does not raise
    ``ValueError``).
    """
    n_sentences = draw(st.integers(min_value=4, max_value=8))
    sentences = []
    for _ in range(n_sentences):
        length = draw(st.integers(min_value=3, max_value=6))
        words = draw(
            st.lists(st.sampled_from(_WORD_POOL), min_size=length, max_size=length)
        )
        sentences.append(" ".join(words))
    repeats = draw(st.integers(min_value=12, max_value=20))
    return sentences * repeats


@st.composite
def orthogonal_matrix(draw, dim=None):
    """Draw a random orthogonal matrix, built via QR of a random matrix."""
    d = dim if dim is not None else draw(st.integers(min_value=2, max_value=8))
    base = draw(
        hynp.arrays(
            dtype=np.float64,
            shape=(d, d),
            elements=st.floats(
                min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    q, _ = np.linalg.qr(base + np.eye(d))
    return q


@st.composite
def embedding_pair_with_orthogonal_target(draw):
    """Source embeddings and a target that is an exact orthogonal rotation.

    Returns ``(source, target, vocab, rotation)`` where
    ``target == source @ rotation`` and ``rotation`` is orthogonal. Rows are
    kept away from zero so per-row normalisation in ``fit`` stays well defined.
    """
    n = draw(st.integers(min_value=5, max_value=15))
    d = draw(st.integers(min_value=2, max_value=8))
    source = draw(
        hynp.arrays(
            dtype=np.float64,
            shape=(n, d),
            elements=st.floats(
                min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
            ),
        )
    )
    # Keep every row non-trivial so normalisation doesn't collapse an anchor.
    if not np.all(np.linalg.norm(source, axis=1) > 1e-3):
        source = source + 0.5
    rotation = draw(orthogonal_matrix(dim=d))
    target = source @ rotation
    vocab = [f"w{i}" for i in range(n)]
    return source, target, vocab, rotation


@st.composite
def ppmi_inputs(draw):
    """Build valid constructor inputs for :class:`PPMIComputer`.

    Generates Count-Min-Sketch-shaped count tables, matching seeds, a small
    vocabulary, and positive totals — the contract the kernel expects.
    """
    depth = draw(st.integers(min_value=2, max_value=4))
    width = draw(st.integers(min_value=8, max_value=64))
    n_vocab = draw(st.integers(min_value=1, max_value=8))
    count_elems = st.integers(min_value=0, max_value=200)
    skipgram_counts = draw(
        hynp.arrays(dtype=np.int32, shape=(depth, width), elements=count_elems)
    )
    word_counts = draw(
        hynp.arrays(dtype=np.int32, shape=(depth, width), elements=count_elems)
    )
    seeds = draw(
        st.lists(
            st.integers(min_value=1, max_value=1_000_000),
            min_size=depth,
            max_size=depth,
            unique=True,
        )
    )
    vocabulary = [f"w{i}" for i in range(n_vocab)]
    return {
        "skipgram_counts": skipgram_counts,
        "word_counts": word_counts,
        "vocabulary": vocabulary,
        "seeds": seeds,
        "width": width,
        "skip_total": float(max(1, int(skipgram_counts.sum()))),
        "word_total": float(max(1, int(word_counts.sum()))),
    }
