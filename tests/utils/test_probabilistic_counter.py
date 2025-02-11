import pytest

from chronowords.utils.probabilistic_counter import (
    CountMinSketch,  # Adjust import based on your file structure
)


@pytest.fixture
def cms():
    """Fixture to create a CountMinSketch instance before each test."""
    return CountMinSketch(width=1000, depth=5, seed=42)


def test_initialization(cms):
    """Test the initialization of CountMinSketch."""
    assert cms.width == 1000
    assert cms.depth == 5
    assert cms.total == 0


def test_update_and_query(cms):
    """Test updating and querying counts."""
    cms.update("apple")
    cms.update("banana", count=5)
    assert cms.query("apple") == 1
    assert cms.query("banana") == 5
    assert cms.query("unseen_word") == 0


def test_get_heavy_hitters(cms):
    """Test getting heavy hitters."""
    for _ in range(100):
        cms.update("frequent")
    for _ in range(10):
        cms.update("rare")

    heavy = cms.get_heavy_hitters(threshold=0.05)
    assert len(heavy) > 0
    assert heavy[0][0] == "frequent"


def test_merge(cms):
    """Test merging two CountMinSketch objects."""
    cms1 = CountMinSketch(width=1000, depth=5, seed=42)
    cms2 = CountMinSketch(width=1000, depth=5, seed=42)
    cms1.update("word", count=3)
    cms2.update("word", count=2)
    cms1.merge(cms2)
    assert cms1.query("word") == 5


def test_merge_incompatible_sketches(cms):
    """Test merging incompatible CountMinSketch objects."""
    cms1 = CountMinSketch(width=1000, depth=5, seed=42)
    cms2 = CountMinSketch(width=500, depth=5, seed=42)
    with pytest.raises(ValueError):
        cms1.merge(cms2)


def test_estimate_error(cms):
    """Test estimating the error."""
    for _ in range(1000):
        cms.update("word")
    error = cms.estimate_error(confidence=0.95)
    assert error > 0
    assert error < cms.total
