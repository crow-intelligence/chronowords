import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from scipy.sparse import csr_matrix

from chronowords.algebra.svd import SVDAlgebra
from chronowords.utils.count_skipgrams import PPMIComputer  # ty: ignore


def test_ppmi_computation():
    """Test PPMI matrix computation with various parameters."""
    vocab_size = 5
    counts = np.zeros((3, 100), dtype=np.int32)
    for i in range(vocab_size):
        counts[0, i] = 10 * (i + 1)
        counts[1, i] = 5 * (i + 1)

    vocabulary = [f"word{i}" for i in range(vocab_size)]
    seeds = [42, 43, 44]

    params = [
        {"shift": 1.0, "alpha": 0.75},
        {"shift": 2.0, "alpha": 0.75},
        {"shift": 1.0, "alpha": 1.0},
    ]

    for param in params:
        computer = PPMIComputer(
            skipgram_counts=counts,
            word_counts=counts,
            vocabulary=vocabulary,
            seeds=seeds,
            width=100,
            skip_total=sum(counts[0]),
            word_total=sum(counts[0]),
            **param,
        )
        matrix = computer.compute_ppmi_matrix_with_sketch()

        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (vocab_size, vocab_size)
        assert matrix.data.dtype == np.float64
        assert np.all(matrix.data >= 0)


def test_svd_algebra_full_pipeline(small_corpus):
    """Test complete word embedding pipeline."""
    model = SVDAlgebra(
        n_components=5,
        window_size=3,
        min_word_length=2,
        cms_width=100,
        cms_depth=3,
    )

    model.train(iter(small_corpus))

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


@pytest.mark.parametrize("batch_size", [2, 5, 10])
def test_ppmi_batch_processing(batch_size):
    """Test PPMI computation with different batch sizes."""
    vocab_size = 10
    counts = np.zeros((3, 100), dtype=np.int32)
    counts[0, :vocab_size] = 10
    vocabulary = [f"word{i}" for i in range(vocab_size)]

    computer = PPMIComputer(
        skipgram_counts=counts,
        word_counts=counts,
        vocabulary=vocabulary,
        seeds=[42, 43, 44],
        width=100,
        skip_total=float(vocab_size * 10),
        word_total=float(vocab_size * 10),
    )
    matrix = computer.compute_ppmi_matrix_with_sketch(batch_size=batch_size)

    assert matrix.shape == (vocab_size, vocab_size)


def test_model_persistence(tmp_path, small_corpus):
    """Test model save/load functionality."""
    model = SVDAlgebra(n_components=5, cms_width=100, cms_depth=3)
    model.train(iter(small_corpus))

    save_path = tmp_path / "model"
    model.save_model(save_path)

    loaded_model = SVDAlgebra(n_components=5, cms_width=100, cms_depth=3)
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
    model = SVDAlgebra(n_components=5, cms_width=100, cms_depth=3, min_word_length=2)
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
