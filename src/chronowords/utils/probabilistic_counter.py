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

    def __init__(
        self,
        width: int = 1_000_000,
        depth: int = 5,
        seed: int = 42,
        track_keys: bool = True,
    ):
        """Initialize Count-Min Sketch.

        Args:
        ----
            width: Number of counters per hash function (controls accuracy)
            depth: Number of hash functions (controls probability bound)
            seed: Random seed for hash function initialization
            track_keys: Whether to track observed keys (disable for memory savings)

        """
        self.width = width
        self.depth = depth
        self.seed = seed
        self.total: int = 0
        self._track_keys = track_keys

        self.counts = np.zeros((depth, width), dtype=np.int32)

        rng = np.random.RandomState(seed)
        self.seeds = [int(s) for s in rng.randint(0, 1_000_000, size=depth)]

        self._observed_keys: set[str] = set()
        self._row_indices = np.arange(self.depth)

    def _hash_indices(self, key: bytes) -> np.ndarray:
        """Compute hash indices for all rows at once."""
        return np.array(
            [mmh3.hash(key, seed) % self.width for seed in self.seeds],
            dtype=np.intp,
        )

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
            key_bytes = key.encode()
            if self._track_keys:
                self._observed_keys.add(key)
        else:
            key_bytes = key
            if self._track_keys:
                self._observed_keys.add(key.decode())

        self.total += count

        indices = self._hash_indices(key_bytes)
        self.counts[self._row_indices, indices] += count

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

        indices = self._hash_indices(key)
        return int(np.min(self.counts[self._row_indices, indices]))

    def get_heavy_hitters(self, threshold: float) -> list[tuple[str, int]]:
        """Get items that appear more than threshold * total times.

        Args:
        ----
            threshold: Minimum frequency as fraction of total counts

        Returns:
        -------
            List of (item, count) pairs sorted by count descending

        Raises:
        ------
            RuntimeError: If track_keys was disabled

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
        if not self._track_keys:
            raise RuntimeError("Cannot get heavy hitters when track_keys=False")

        threshold_count = int(self.total * threshold)
        candidates = {}

        for key in self._observed_keys:
            count = self.query(key)
            if count > threshold_count:
                candidates[key] = count

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
