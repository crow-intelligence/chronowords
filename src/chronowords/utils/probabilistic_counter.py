from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import mmh3
import numpy as np


class CountMinSketch:
    """
    Count-Min Sketch implementation for memory-efficient counting.

    Uses multiple hash functions to approximate frequencies with bounded error.
    Memory usage: width * depth * 4 bytes
    Error bound: ≈ 2/width with probability 1 - 1/2^depth
    """

    def __init__(self, width: int = 1_000_000, depth: int = 5, seed: int = 42):
        """
        Initialize Count-Min Sketch.

        Args:
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
        self._observed_keys = set()

    def update(self, key: Union[str, bytes], count: int = 1) -> None:
        """
        Update count for a key.

        Args:
            key: Item to count (string or bytes)
            count: Amount to increment (default: 1)
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

    def query(self, key: Union[str, bytes]) -> int:
        """Query count for a key."""
        if isinstance(key, str):
            key = key.encode()

        # Return minimum count across all hash functions
        min_count = float("inf")
        for i, seed in enumerate(self.seeds):
            idx = mmh3.hash(key, seed) % self.width
            min_count = min(min_count, self.counts[i, idx])

        return int(min_count)

    def get_heavy_hitters(self, threshold: float) -> List[Tuple[str, int]]:
        """
        Get items that appear more than threshold * total times.

        Args:
            threshold: Minimum frequency as fraction of total counts

        Returns:
            List of (item, count) pairs sorted by count descending
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
        """Merge another sketch into this one."""
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
        """
        Estimate maximum counting error.

        Args:
            confidence: Confidence level for the error bound

        Returns:
            Maximum expected counting error at given confidence level
        """
        epsilon = 2.0 / self.width
        delta = pow(2.0, -self.depth)

        if confidence > 0:
            delta = delta / confidence

        return epsilon * self.total

    @property
    def arrays(self) -> Tuple[np.ndarray, List[int], int]:
        """Get raw arrays and parameters for Cython code."""
        return self.counts, self.seeds, self.width
