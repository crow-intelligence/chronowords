import tempfile

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from chronowords.algebra.svd import SVDAlgebra
from tests.strategies import trainable_corpus


def test_svd_algebra_full_pipeline(small_corpus):
    """Test complete word embedding pipeline."""
    model = SVDAlgebra(
        n_components=5,
        window_size=3,
        min_word_length=2,
        min_count=1,
        ppmi_min_count=0,
    )

    model.train(small_corpus)

    assert len(model.vocabulary) > 0
    assert model.embeddings is not None
    assert model.embeddings.shape[0] == len(model.vocabulary)
    assert not np.any(np.isnan(model.embeddings))

    king_vec = model.get_vector("king")
    assert king_vec is not None
    assert np.any(king_vec != 0)

    for word in ["king", "queen"]:
        if word in model.vocabulary:
            similars = model.most_similar(word, n=3)
            assert len(similars) > 0
            assert all(0 <= sim.similarity <= 1 for sim in similars)


def test_train_produces_nonempty_ppmi(caplog):
    """Training on a corpus with repeated co-occurrences yields real PPMI signal.

    Regression: word marginals were read from the skip-gram sketch, so every
    vocabulary word looked unseen, the PPMI matrix was empty and the
    embeddings came from the noisy dense-SVD fallback.
    """
    corpus = ["cat dog fish", "bird cow pig"] * 20
    model = SVDAlgebra(n_components=2, window_size=2, min_count=1)

    with caplog.at_level("WARNING", logger="chronowords.algebra.svd"):
        model.train(line for line in corpus)

    assert model._ppmi_sparse is not None
    assert model._ppmi_sparse.nnz > 0
    assert "PPMI matrix is empty" not in caplog.text
    # cat precedes dog on every line; cat never precedes cow
    i = model._vocab_index
    assert model._ppmi_sparse[i["cat"], i["dog"]] > 0
    assert model._ppmi_sparse[i["cat"], i["cow"]] == 0


def test_train_warns_on_empty_ppmi(caplog):
    """An empty PPMI matrix is reported, not silently factorised."""
    corpus = ["alpha beta gamma delta"]  # every count is 1, below the >5 threshold
    model = SVDAlgebra(n_components=2, min_count=1)

    with caplog.at_level("WARNING", logger="chronowords.algebra.svd"):
        model.train(line for line in corpus)

    assert "PPMI matrix is empty" in caplog.text


def test_model_persistence(tmp_path, small_corpus):
    """Test model save/load functionality."""
    model = SVDAlgebra(n_components=5, min_count=1, ppmi_min_count=0)
    model.train(small_corpus)

    save_path = tmp_path / "model"
    model.save_model(save_path)

    loaded_model = SVDAlgebra(n_components=5)
    loaded_model.load_model(save_path)

    assert model.vocabulary == loaded_model.vocabulary
    assert model.embeddings is not None
    assert loaded_model.embeddings is not None
    np.testing.assert_array_almost_equal(model.embeddings, loaded_model.embeddings)


def test_distance():
    """Test cosine distance between words."""
    model = SVDAlgebra(n_components=2)
    model.vocabulary = ["cat", "dog", "fish"]
    model._build_vocab_index()
    model.embeddings = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])

    # Same word should have distance ~0
    dist = model.distance("cat", "cat")
    assert dist is not None
    assert dist < 0.01

    # Orthogonal vectors should have distance ~1
    dist = model.distance("cat", "fish")
    assert dist is not None
    assert abs(dist - 1.0) < 0.01

    # Unknown word returns None
    assert model.distance("cat", "unknown") is None
    assert model.distance("unknown", "cat") is None


def test_distance_zero_norm():
    """Test distance with zero-norm vectors."""
    model = SVDAlgebra(n_components=2)
    model.vocabulary = ["zero", "cat"]
    model._build_vocab_index()
    model.embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])

    assert model.distance("zero", "cat") is None


def test_analogy():
    """Test word analogy with synthetic embeddings."""
    model = SVDAlgebra(n_components=3)
    model.vocabulary = ["king", "man", "woman", "queen", "prince"]
    model._build_vocab_index()
    model.embeddings = np.array(
        [
            [1.0, 1.0, 0.0],  # king
            [0.0, 1.0, 0.0],  # man
            [0.0, 0.0, 1.0],  # woman
            [1.0, 0.0, 1.0],  # queen = king - man + woman
            [0.8, 0.8, 0.0],  # prince
        ]
    )

    result = model.analogy(["king", "man"], "woman", n=2)
    assert result is not None
    assert len(result.words) > 0
    assert len(result.words) == len(result.similarities)


def test_analogy_unknown_word():
    """Test analogy with unknown word returns None."""
    model = SVDAlgebra(n_components=2)
    model.vocabulary = ["a", "b"]
    model._build_vocab_index()
    model.embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert model.analogy(["a", "unknown"], "b") is None


def test_analogy_wrong_args():
    """Test analogy with wrong number of positive words."""
    model = SVDAlgebra(n_components=2)
    model.vocabulary = ["a", "b", "c"]
    model._build_vocab_index()
    model.embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    assert model.analogy(["a"], "b") is None


def test_analogy_no_embeddings():
    """Test analogy when model has no embeddings."""
    model = SVDAlgebra(n_components=2)
    assert model.analogy(["a", "b"], "c") is None


def test_get_vector_unknown():
    """Test get_vector for unknown word and no embeddings."""
    model = SVDAlgebra(n_components=2)
    assert model.get_vector("anything") is None

    model.vocabulary = ["a"]
    model._build_vocab_index()
    model.embeddings = np.array([[1.0, 0.0]])
    assert model.get_vector("unknown") is None


def test_most_similar_empty():
    """Test most_similar with unknown word."""
    model = SVDAlgebra(n_components=2)
    model.vocabulary = ["a"]
    model._build_vocab_index()
    model.embeddings = np.array([[1.0, 0.0]])
    assert model.most_similar("unknown") == []


@pytest.fixture(scope="module")
def trained_model():
    """Train a small SVDAlgebra once for property-based tests."""
    corpus = [
        "king queen crown palace royal throne",
        "king rules kingdom palace throne",
        "queen rules kingdom crown royal",
        "prince princess royal palace crown",
        "man woman child family home",
        "boy girl child plays home",
        "father mother parent child family",
        "king leads army battle victory",
        "queen commands navy battle victory",
    ]
    model = SVDAlgebra(n_components=5, min_word_length=2, min_count=1, ppmi_min_count=0)
    model.train(line for line in corpus)
    return model


@given(data=st.data())
@settings(deadline=None, max_examples=50)
def test_distance_is_symmetric(trained_model, data):
    """Cosine distance must be symmetric: distance(a, b) == distance(b, a).

    Asymmetry would break the metric assumptions underlying `most_similar`
    and `analogy`, since both rely on distance being a well-defined relation
    between pairs rather than an ordered operation.
    """
    vocab = trained_model.vocabulary
    assume_nonempty = len(vocab) >= 2
    if not assume_nonempty:
        return
    word1 = data.draw(st.sampled_from(vocab))
    word2 = data.draw(st.sampled_from(vocab))

    d12 = trained_model.distance(word1, word2)
    d21 = trained_model.distance(word2, word1)

    if d12 is None or d21 is None:
        assert d12 is None and d21 is None
    else:
        assert abs(d12 - d21) < 1e-9


@given(data=st.data(), n=st.integers(min_value=1, max_value=20))
@settings(deadline=None, max_examples=50)
def test_most_similar_output_contract(trained_model, data, n):
    """`most_similar(word, n)` must honor its documented output contract.

    For any vocab word with a non-zero embedding the returned list must
    (i) have at most `n` entries, (ii) only contain distinct vocabulary
    words other than the query, (iii) carry similarities in [-1, 1],
    and (iv) be sorted by similarity descending. These invariants are what
    downstream callers (analogy, temporal diff tooling) rely on.
    """
    vocab = trained_model.vocabulary
    if len(vocab) < 2:
        return
    word = data.draw(st.sampled_from(vocab))

    results = trained_model.most_similar(word, n=n)

    assert len(results) <= n
    vocab_set = set(vocab)
    seen: set[str] = set()
    for item in results:
        assert item.word in vocab_set
        assert item.word != word
        assert item.word not in seen
        seen.add(item.word)
        assert -1.0 <= item.similarity <= 1.0

    sims = [item.similarity for item in results]
    assert sims == sorted(sims, reverse=True)


@given(corpus=trainable_corpus())
@settings(deadline=None, max_examples=15)
def test_save_load_round_trip(corpus):
    """save_model then load_model reproduces the vocabulary and embeddings.

    Models are trained once and reloaded many times downstream, so a reloaded
    model must be indistinguishable from the original.
    """
    model = SVDAlgebra(n_components=5, min_word_length=2, min_count=1)
    model.train(iter(corpus))

    with tempfile.TemporaryDirectory() as tmp:
        model.save_model(tmp)
        reloaded = SVDAlgebra()
        reloaded.load_model(tmp)

    assert reloaded.vocabulary == model.vocabulary
    assert model.embeddings is not None
    assert reloaded.embeddings is not None
    assert np.allclose(reloaded.embeddings, model.embeddings)
    assert reloaded.ppmi is not None and model.ppmi is not None
    np.testing.assert_array_equal(reloaded.ppmi.toarray(), model.ppmi.toarray())
    assert reloaded.training_record == model.training_record


# --- exact counting, counts and training record ------------------------------

SIGNAL_CORPUS = [
    "king queen crown palace royal throne",
    "man woman child family home",
    "king rules kingdom palace throne",
    "boy girl child plays home",
] * 10


def test_m_dense_is_gone():
    """The dense V x V copy is no longer built or exposed."""
    model = SVDAlgebra(n_components=2, min_count=1)
    model.train(SIGNAL_CORPUS)
    assert not hasattr(model, "M_dense")
    assert model.ppmi is model._ppmi_sparse


def test_exact_mode_records_counts_and_parameters():
    """Training returns the counts and the parameters behind the PPMI matrix."""
    model = SVDAlgebra(n_components=2, min_count=10, vocab_size=5)
    model.train(SIGNAL_CORPUS)

    assert model.counts is not None
    assert model.counts.vocabulary == model.vocabulary
    assert len(model.vocabulary) == 5
    assert model.counts.word_counts.min() >= 10
    rec = model.training_record
    assert rec is not None
    assert rec["params"]["min_count"] == 10
    assert rec["word_total"] == sum(len(line.split()) for line in SIGNAL_CORPUS)
    assert rec["vocab_size"] == 5
    assert model.ppmi is not None
    assert rec["ppmi_nnz"] == model.ppmi.nnz


def test_training_is_deterministic():
    """Same corpus, same parameters -> identical vocabulary and embeddings."""
    a = SVDAlgebra(n_components=3, min_count=1)
    b = SVDAlgebra(n_components=3, min_count=1)
    a.train(SIGNAL_CORPUS)
    b.train(iter(SIGNAL_CORPUS))
    assert a.vocabulary == b.vocabulary
    np.testing.assert_array_equal(a.embeddings, b.embeddings)


def test_vocabulary_order_does_not_depend_on_hash_seed(tmp_path):
    """Vocabulary order is identical across processes with different string
    hash seeds (it used to follow set iteration order on count ties).
    """
    import json
    import os
    import subprocess
    import sys

    script = (
        "import json; from chronowords.algebra.svd import SVDAlgebra\n"
        f"m = SVDAlgebra(n_components=2, min_count=1); m.train({SIGNAL_CORPUS!r})\n"
        "print(json.dumps(m.vocabulary))"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-c", script],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2")
    ]
    assert json.loads(runs[0]) == json.loads(runs[1])


def test_train_accepts_paths_lines_and_tokens(tmp_path):
    """A file, a list of lines and pre-tokenised lines give the same model."""
    corpus_file = tmp_path / "corpus.txt"
    corpus_file.write_text("\n".join(SIGNAL_CORPUS) + "\n", encoding="utf-8")
    models = []
    for corpus in (
        corpus_file,
        [corpus_file],
        SIGNAL_CORPUS,
        (line.split() for line in SIGNAL_CORPUS),
    ):
        m = SVDAlgebra(n_components=3, min_count=1)
        m.train(corpus)
        models.append(m)
    for m in models[1:]:
        assert m.vocabulary == models[0].vocabulary
        np.testing.assert_allclose(m.embeddings, models[0].embeddings)


def test_train_rejects_a_bare_string():
    """A str corpus would be iterated character by character; refuse it."""
    with pytest.raises(TypeError, match="single string"):
        SVDAlgebra(min_count=1).train("king queen crown")


def test_empty_vocabulary_explains_min_count():
    """The default min_count is too high for tiny corpora, and says so."""
    with pytest.raises(ValueError, match="Lower min_count"):
        SVDAlgebra().train(["the cat sat"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokenizer": "bpe"},
        {"n_components": 0},
        {"min_count": 0},
        {"vocab_size": 0},
        {"shift": 0.0},
        {"window_size": -1},
    ],
)
def test_constructor_validates(kwargs):
    """Out-of-range parameters fail at construction, not deep inside train."""
    with pytest.raises(ValueError):
        SVDAlgebra(**kwargs)


def test_load_model_reads_legacy_dense_ppmi(tmp_path):
    """Model directories from chronowords < 0.4 stored a dense ppmi.npy."""
    import pickle

    np.save(tmp_path / "embeddings.npy", np.eye(2))
    np.save(tmp_path / "ppmi.npy", np.array([[0.0, 1.5], [0.0, 0.0]]))
    with (tmp_path / "vocabulary.pkl").open("wb") as fh:
        pickle.dump(["cat", "dog"], fh)

    model = SVDAlgebra()
    model.load_model(tmp_path)

    assert model.ppmi is not None
    assert model.ppmi.nnz == 1
    assert model.ppmi[0, 1] == 1.5
    assert model.training_record is None


def test_sketch_parameters_are_gone():
    """The Count-Min Sketch mode was removed in 0.4; its arguments now fail loudly."""
    with pytest.raises(TypeError):
        SVDAlgebra(counting="sketch")  # ty: ignore[unknown-argument]
    with pytest.raises(TypeError):
        SVDAlgebra(100, 5, 3, 1_000_000, 5)  # ty: ignore[too-many-positional-arguments]


# --- vectors without information ---------------------------------------------

SLICE_1775 = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "data"
    / "corpus"
    / "speeches_1775-1799_lemmatized.txt"
)


def test_empty_ppmi_rows_get_exactly_zero_vectors():
    """A word heading no PPMI entry has no information, so its vector is 0.

    In exact arithmetic U = M V S^-1 makes such rows zero; the solver leaves
    round-off that normalisation would turn into a random direction.
    """
    # the old default threshold, which leaves many rows empty on this slice
    model = SVDAlgebra(n_components=20, min_count=20, ppmi_min_count=5)
    model.train(SLICE_1775)
    assert model.ppmi is not None and model.embeddings is not None
    empty = np.diff(model.ppmi.indptr) == 0
    assert empty.any()
    assert (model.embeddings[empty] == 0.0).all()
    zero = (model.embeddings == 0.0).all(axis=1)
    assert model.training_record is not None
    assert model.training_record["zero_vectors"] == int(zero.sum())
    for i in np.flatnonzero(empty)[:5]:
        assert not model.has_vector(model.vocabulary[i])


def test_no_neighbour_lacks_a_vector():
    """Regression: on this small slice, 69 of 117 queries used to return
    round-off vectors (pure noise) among their top-5 neighbours.
    """
    model = SVDAlgebra(n_components=100, min_count=20, ppmi_min_count=5)
    model.train(SLICE_1775)
    queried = 0
    for word in model.vocabulary:
        hits = model.most_similar(word, n=5)
        if not model.has_vector(word):
            assert hits == []
            continue
        queried += 1
        assert all(model.has_vector(h.word) for h in hits)
    assert queried > 0
    no_vector = next(w for w in model.vocabulary if not model.has_vector(w))
    assert model.distance(no_vector, model.vocabulary[0]) is None
    assert model.analogy([no_vector, model.vocabulary[0]], model.vocabulary[1]) is None


def test_eig_scales_singular_values():
    """Embeddings are U * S**eig: changing eig rescales each dimension only."""
    a = SVDAlgebra(n_components=3, min_count=1, eig=0.0)
    b = SVDAlgebra(n_components=3, min_count=1, eig=1.0)
    a.train(SIGNAL_CORPUS)
    b.train(SIGNAL_CORPUS)
    assert a.embeddings is not None and b.embeddings is not None
    has = (a.embeddings != 0).any(axis=1)
    ratio = b.embeddings[has] / a.embeddings[has]
    # each column is scaled by one constant (its singular value)
    np.testing.assert_allclose(ratio, np.broadcast_to(ratio[0], ratio.shape), rtol=1e-8)
    assert b.training_record is not None
    assert b.training_record["params"]["eig"] == 1.0


@pytest.mark.parametrize("kwargs", [{"alpha": 0.0}, {"eig": -0.5}])
def test_smoothing_parameters_are_validated(kwargs):
    """Alpha must be positive and eig non-negative."""
    with pytest.raises(ValueError):
        SVDAlgebra(**kwargs)
