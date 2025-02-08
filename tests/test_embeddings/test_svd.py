import numpy as np
import pytest
from scipy.sparse import csr_matrix

from chronowords.algebra.svd import SVDAlgebra
from chronowords.utils.count_skipgrams import PPMIComputer
from chronowords.utils.probabilistic_counter import CountMinSketch


@pytest.fixture
def small_corpus():
    """Create a test corpus with richer patterns and relationships."""
    return [
        # Royal/monarchy semantic field
        "king queen crown palace royal throne",
        "king rules kingdom palace throne",
        "queen rules kingdom crown royal",
        "prince princess royal palace crown",
        "king commands royal subjects throne",
        "queen leads royal court palace",
        # People and relationships
        "man woman child family home",
        "boy girl child plays home",
        "father mother parent child family",
        "brother sister sibling family home",
        "husband wife marriage family home",
        # Professional contexts
        "king leads army battle victory",
        "queen commands navy battle victory",
        "soldier fights army battle sword",
        "warrior battles enemy sword shield",
        "knight serves king castle honor",
        # Abstract concepts
        "king represents power authority rule",
        "queen embodies wisdom grace power",
        "power comes with great responsibility",
        "leadership requires wisdom strength",
        # Common patterns for analogies
        "king man queen woman",  # gender relations
        "king prince queen princess",  # hierarchical relations
        "king throne queen crown",  # symbol relations
        # Animal kingdom (for contrast)
        "lion tiger bear wolf hunt",
        "cat dog pet animal home",
        "eagle hawk bird flies high",
        "fish swims ocean water deep",
        # Repeat key patterns to strengthen them
        "king queen crown palace royal throne",
        "king rules kingdom palace throne",
        "queen rules kingdom crown royal",
        "man woman child family home",
        "king leads army battle victory",
        "queen commands navy battle victory",
    ]


@pytest.fixture
def small_sketch():
    """Create a small Count-Min Sketch for testing."""
    return CountMinSketch(width=100, depth=3, seed=42)


@pytest.fixture
def filled_sketch(small_sketch, small_corpus):
    """Create a Count-Min Sketch filled with test data."""
    for line in small_corpus:
        for word in line.split():
            small_sketch.update(word)
    return small_sketch


def test_count_min_sketch_basic(small_sketch):
    """Test basic CountMinSketch functionality."""
    # Test single update
    small_sketch.update("test")
    assert small_sketch.query("test") == 1

    # Test multiple updates
    small_sketch.update("test", 5)
    assert small_sketch.query("test") == 6

    # Test unseen word
    assert small_sketch.query("unseen") == 0

    # Test string vs bytes handling
    small_sketch.update(b"bytes_test")
    assert small_sketch.query("bytes_test") == 1
    assert small_sketch.query(b"bytes_test") == 1


def test_count_min_sketch_heavy_hitters(filled_sketch):
    """Test heavy hitters identification with debug output."""
    # Print total counts
    print(f"\nTotal counts in sketch: {filled_sketch.total}")

    # Count actual occurrences in test corpus
    word_counts = {}
    for line in [
        "king queen crown palace",
        "king queen crown palace",
        "man woman child person",
        "man woman child person",
        "king man queen woman",
        "king man queen woman",
        "dog cat pet animal",
        "dog cat pet animal",
    ]:
        for word in line.split():
            word_counts[word] = word_counts.get(word, 0) + 1

    print("\nActual word counts:")
    for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{word}: {count} (in corpus) vs {filled_sketch.query(word)} (in sketch)")

    # Since we know 'king' appears 4 times in 32 total words, use threshold of 3/32
    threshold = 3.0 / filled_sketch.total  # This should catch words appearing 4 times
    heavy_hitters = filled_sketch.get_heavy_hitters(threshold)

    # Debug output
    print(f"\nUsing threshold {threshold}:")
    for word, count in heavy_hitters:
        print(f"{word}: {count}")

    assert len(heavy_hitters) > 0, "No heavy hitters found"

    # Results should be (word, count) tuples sorted by count
    assert all(
        isinstance(word, str) and isinstance(count, int)
        for word, count in heavy_hitters
    )
    assert all(
        heavy_hitters[i][1] >= heavy_hitters[i + 1][1]
        for i in range(len(heavy_hitters) - 1)
    )

    # Common words should be detected
    common_words = {word for word, _ in heavy_hitters}
    assert (
        "king" in common_words or "queen" in common_words
    ), f"Expected 'king' or 'queen' in {common_words}. Counts: {word_counts}"


def test_count_min_sketch_merge(small_sketch):
    """Test sketch merging functionality."""
    other_sketch = CountMinSketch(width=100, depth=3, seed=42)

    # Add different words to each sketch
    small_sketch.update("word1", 5)
    other_sketch.update("word2", 3)

    # Merge sketches
    small_sketch.merge(other_sketch)

    # Check if counts are preserved
    assert small_sketch.query("word1") == 5
    assert small_sketch.query("word2") == 3

    # Test merging incompatible sketches
    incompatible = CountMinSketch(width=200, depth=3, seed=42)
    with pytest.raises(ValueError):
        small_sketch.merge(incompatible)


def test_ppmi_computation():
    """Test PPMI matrix computation with various parameters."""
    # Create test data
    vocab_size = 5
    counts = np.zeros((3, 100), dtype=np.int32)
    # Add some test patterns
    for i in range(vocab_size):
        counts[0, i] = 10 * (i + 1)  # Increasing counts
        counts[1, i] = 5 * (i + 1)  # Different pattern

    vocabulary = [f"word{i}" for i in range(vocab_size)]
    seeds = [42, 43, 44]

    # Test with different parameters
    params = [
        {"shift": 1.0, "alpha": 0.75},  # Default params
        {"shift": 2.0, "alpha": 0.75},  # Higher shift
        {"shift": 1.0, "alpha": 1.0},  # No smoothing
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

        # Check matrix properties
        assert isinstance(matrix, csr_matrix)
        assert matrix.shape == (vocab_size, vocab_size)
        assert isinstance(matrix.data, np.ndarray)
        assert matrix.data.dtype == np.float64
        assert np.all(matrix.data >= 0)  # PPMI should be non-negative


def test_svd_algebra_full_pipeline(small_corpus):
    """Test complete word embedding pipeline with debug output."""
    print("\nTesting with corpus size:", len(small_corpus))

    model = SVDAlgebra(
        n_components=20,  # Increased components
        window_size=3,  # Increased window size
        min_word_length=2,  # Keep short words
        cms_width=1000,  # Increased width for better counting
        cms_depth=5,  # Increased depth for better counting
    )

    # Train model
    print("\nTraining model...")
    model.train(iter(small_corpus))

    # Verify embeddings
    print("\nVerifying embeddings...")
    assert len(model.vocabulary) > 0, "Vocabulary should not be empty"
    assert model.embeddings is not None, "Embeddings should not be None"
    assert model.embeddings.shape[0] == len(model.vocabulary)
    assert not np.any(np.isnan(model.embeddings)), "Found NaN values in embeddings"

    # Verify vector for 'king'
    king_vec = model.get_vector("king")
    assert king_vec is not None, "'king' not found in vocabulary"
    assert np.any(king_vec != 0), "Zero vector found for 'king'"

    print("\nTesting word similarities...")
    # Test similarities for multiple words
    for word in ["king", "queen", "man", "woman"]:
        if word in model.vocabulary:
            similars = model.most_similar(word, n=5)
            print(f"\nMost similar to {word}:")
            for sim in similars:
                print(f"  {sim.word}: {sim.similarity:.4f}")

            # Verify we got results
            assert len(similars) > 0, f"No similar words found for '{word}'"
            # Verify similarities are valid
            assert all(
                0 <= sim.similarity <= 1 for sim in similars
            ), f"Invalid similarities found for '{word}'"

    print("\nAll tests completed successfully!")


@pytest.mark.parametrize("batch_size", [2, 5, 10])
def test_ppmi_batch_processing(batch_size):
    """Test PPMI computation with different batch sizes."""
    vocab_size = 10
    counts = np.zeros((3, 100), dtype=np.int32)
    counts[0, :vocab_size] = 10  # Set some counts
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

    # Save model
    save_path = tmp_path / "model"
    model.save_model(save_path)

    # Load model
    loaded_model = SVDAlgebra(n_components=5, cms_width=100, cms_depth=3)
    loaded_model.load_model(save_path)

    # Check if models are equivalent
    assert model.vocabulary == loaded_model.vocabulary
    np.testing.assert_array_almost_equal(model.embeddings, loaded_model.embeddings)
