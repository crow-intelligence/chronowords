from collections import Counter

import pytest
from hypothesis import given
from hypothesis import strategies as st

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


@given(stream=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=200))
def test_query_never_underestimates_true_count(stream):
    """CMS guarantee: query(k) >= true_count(k) for every key in any stream.

    The defining property of Count-Min Sketch: hash collisions can inflate a
    counter but never deflate it, so the returned estimate is an upper bound
    on the true frequency. A violation would invalidate the error bounds the
    whole library relies on.
    """
    sketch = CountMinSketch(width=500, depth=4, seed=42)
    for item in stream:
        sketch.update(item)

    truth = Counter(stream)
    for key, true_count in truth.items():
        assert sketch.query(key) >= true_count


@given(
    stream_a=st.lists(st.text(min_size=1, max_size=10), max_size=100),
    stream_b=st.lists(st.text(min_size=1, max_size=10), max_size=100),
)
def test_merge_equals_single_combined_stream(stream_a, stream_b):
    """Merging two sketches is equivalent to a single sketch over both streams.

    `merge` adds counter arrays elementwise (same width/depth/seeds), so for
    every key the merged sketch's query must match what a single sketch would
    report after ingesting stream_a followed by stream_b. Any divergence would
    silently break distributed counting — the whole point of `merge`.
    """
    width, depth, seed = 500, 4, 42

    merged = CountMinSketch(width=width, depth=depth, seed=seed)
    other = CountMinSketch(width=width, depth=depth, seed=seed)
    for item in stream_a:
        merged.update(item)
    for item in stream_b:
        other.update(item)
    merged.merge(other)

    combined = CountMinSketch(width=width, depth=depth, seed=seed)
    for item in stream_a + stream_b:
        combined.update(item)

    assert merged.total == combined.total
    for key in set(stream_a) | set(stream_b):
        assert merged.query(key) == combined.query(key)
