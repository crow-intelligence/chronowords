import mmh3
import numpy as np


class CountMinSketch:
    """Count-Min Sketch implementation for memory-efficient counting.

    Uses multiple hash functions to approximate frequencies with bounded error.
    Memory usage: width * depth * 4 bytes
    Error bound: ≈ 2/width with probability 1 - 1/2^depth

    Examples
    --------
        >>> cms = CountMinSketch(width=1000, depth=5, seed=42)
        >>> cms.width
        1000
        >>> cms.depth
        5

    """

    def __init__(self, width: int = 1_000_000, depth: int = 5, seed: int = 42):
        """Initialize Count-Min Sketch.

        Args:
        ----
            width: Number of counters per hash function (controls accuracy)
            depth: Number of hash functions (controls probability bound)
            seed: Random seed for hash function initialization

        """
        self.width = width
        self.depth = depth
        self.seed = seed
        self.total: int = 0

        # Initialize counting array
        self.counts = np.zeros((depth, width), dtype=np.int32)

        # Generate hash function seeds
        rng = np.random.RandomState(seed)
        self.seeds = [int(s) for s in rng.randint(0, 1_000_000, size=depth)]

        # Keep track of all observed keys
        self._observed_keys: set[str] = set()

    def update(self, key: str | bytes, count: int = 1) -> None:
        """Update count for a key.

        Args:
        ----
            key: Item to count (string or bytes)
            count: Amount to increment (default: 1)

        Examples:
        --------
            >>> cms = CountMinSketch(width=1000, depth=5, seed=42)
            >>> cms.update("apple")
            >>> cms.update("apple")
            >>> cms.query("apple")
            2
            >>> cms.update("banana", count=5)
            >>> cms.query("banana")
            5
            >>> cms.total
            7

        """
        if isinstance(key, str):
            key = key.encode()
            # Store original string key
            self._observed_keys.add(key.decode())
        else:
            self._observed_keys.add(key.decode())

        self.total += count

        # Update each row
        for i, seed in enumerate(self.seeds):
            idx = mmh3.hash(key, seed) % self.width
            self.counts[i, idx] += count

    def query(self, key: str | bytes) -> int:
        """Query count for a key.

        Examples
        --------
            >>> cms = CountMinSketch(width=1000, depth=5, seed=42)
            >>> cms.update("rare_word")
            >>> cms.query("rare_word")
            1
            >>> cms.query("unseen_word")
            0

        """
        if isinstance(key, str):
            key = key.encode()

        # Return minimum count across all hash functions
        min_count = float("inf")
        for i, seed in enumerate(self.seeds):
            idx = mmh3.hash(key, seed) % self.width
            min_count = min(min_count, self.counts[i, idx])

        return int(min_count)

    def get_heavy_hitters(self, threshold: float) -> list[tuple[str, int]]:
        """Get items that appear more than threshold * total times.

        Args:
        ----
            threshold: Minimum frequency as fraction of total counts

        Returns:
        -------
            List of (item, count) pairs sorted by count descending

        Examples:
        --------
            >>> cms = CountMinSketch(width=1000, depth=5, seed=42)
            >>> # Add a frequent word
            >>> for _ in range(100):
            ...     cms.update("frequent")
            >>> # Add some less frequent words
            >>> for _ in range(10):
            ...     cms.update("rare")
            >>> heavy = cms.get_heavy_hitters(threshold=0.05)  # 5% threshold
            >>> len(heavy) > 0
            True
            >>> "frequent" == heavy[0][0]  # Most frequent word
            True

        """
        threshold_count = int(self.total * threshold)
        candidates = {}

        # Check counts for all observed keys
        for key in self._observed_keys:
            count = self.query(key)
            if count > threshold_count:
                candidates[key] = count

        # Sort by count in descending order
        return sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    def merge(self, other: "CountMinSketch") -> None:
        """Merge another sketch into this one.

        Examples
        --------
            >>> cms1 = CountMinSketch(width=1000, depth=5, seed=42)
            >>> cms2 = CountMinSketch(width=1000, depth=5, seed=42)
            >>> cms1.update("word", count=3)
            >>> cms2.update("word", count=2)
            >>> cms1.merge(cms2)
            >>> cms1.query("word")
            5
            >>> cms1.total
            5

            >>> # Error case - incompatible sketches
            >>> cms3 = CountMinSketch(width=500, depth=5, seed=42)
            >>> cms1.merge(cms3)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
            ValueError: Can only merge compatible sketches

        """
        if (
            self.width != other.width
            or self.depth != other.depth
            or self.seeds != other.seeds
        ):
            raise ValueError("Can only merge compatible sketches")

        self.counts += other.counts
        self.total += other.total
        self._observed_keys.update(other._observed_keys)

    def estimate_error(self, confidence: float = 0.95) -> float:
        """Estimate maximum counting error.

        Args:
        ----
            confidence: Confidence level for the error bound

        Returns:
        -------
            Maximum expected counting error at given confidence level

        Examples:
        --------
            >>> cms = CountMinSketch(width=1000, depth=5, seed=42)
            >>> for _ in range(1000):
            ...     cms.update("word")
            >>> error = cms.estimate_error(confidence=0.95)
            >>> error > 0  # Should have some error estimation
            True
            >>> error < cms.total  # Error should be less than total counts
            True

        """
        epsilon = 2.0 / self.width
        delta = pow(2.0, -self.depth)

        if confidence > 0:
            delta = delta / confidence

        return epsilon * self.total

    @property
    def arrays(self) -> tuple[np.ndarray, list[int], int]:
        """Get raw arrays and parameters for Cython code.

        Examples
        --------
            >>> cms = CountMinSketch(width=3, depth=2, seed=42)
            >>> counts, seeds, width = cms.arrays
            >>> counts.shape
            (2, 3)
            >>> isinstance(seeds, list)
            True
            >>> width
            3

        """
        return self.counts, self.seeds, self.width
