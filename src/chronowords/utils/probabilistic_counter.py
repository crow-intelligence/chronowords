from typing import Optional, Union

import mmh3
import numpy as np


class CountMinSketch:
    """
    Count-Min Sketch implementation for memory-efficient counting.
    Uses MurmurHash3 for hashing.
    """

    def __init__(
        self,
        width: int = 1_000_000,  # Number of counters per hash function
        depth: int = 5,  # Number of hash functions
        seed: int = 42,
    ):
        """
        Initialize Count-Min Sketch.

        Args:
            width: Number of counters per hash function
            depth: Number of hash functions
            seed: Random seed for hash functions
        """
        self.width = width
        self.depth = depth
        self.seed = seed
        self.total = 0

        # Initialize count matrix
        self.counts = np.zeros((depth, width), dtype=np.int32)

        # Generate seeds for each hash function
        self.seeds = np.random.RandomState(seed).randint(0, 1_000_000, size=depth)

    def update(self, key: Union[str, bytes], count: int = 1) -> None:
        """
        Update the count for a key.

        Args:
            key: The key to update
            count: Amount to increment (default: 1)
        """
        if isinstance(key, str):
            key = key.encode()

        self.total += count

        # Update each row using a different hash function
        for i, seed in enumerate(self.seeds):
            idx = mmh3.hash(key, seed) % self.width
            self.counts[i, idx] += count

    def query(self, key: Union[str, bytes]) -> int:
        """
        Get the approximate count for a key.

        Args:
            key: The key to query

        Returns:
            Estimated count for the key
        """
        if isinstance(key, str):
            key = key.encode()

        # Get count from each hash function and take minimum
        counts = []
        for i, seed in enumerate(self.seeds):
            idx = mmh3.hash(key, seed) % self.width
            counts.append(self.counts[i, idx])

        return int(min(counts))  # Return the minimum count as estimate

    def get_heavy_hitters(self, threshold: float) -> list[tuple[str, int]]:
        """
        Get items that appear more than threshold * total_count times.

        Args:
            threshold: Fraction of total count to be considered frequent

        Returns:
            List of (item, count) pairs for frequent items
        """
        threshold_count = self.total * threshold
        heavy_hitters = set()

        # Check all non-zero counts in first row
        for idx in np.nonzero(self.counts[0] > threshold_count)[0]:
            count = min(self.counts[i, idx] for i in range(self.depth))
            if count > threshold_count:
                heavy_hitters.add((f"item_{idx}", count))

        return sorted(heavy_hitters, key=lambda x: x[1], reverse=True)

    def merge(self, other: "CountMinSketch") -> None:
        """
        Merge another Count-Min Sketch into this one.

        Args:
            other: Another CountMinSketch instance to merge
        """
        if (
            self.width != other.width
            or self.depth != other.depth
            or not np.array_equal(self.seeds, other.seeds)
        ):
            raise ValueError("Can only merge sketches with identical parameters")

        self.counts += other.counts
        self.total += other.total

    def estimate_error(self, confidence: float = 0.95) -> float:
        """
        Estimate the error bound with given confidence.

        Args:
            confidence: Confidence level (between 0 and 1)

        Returns:
            Estimated maximum error in counts
        """
        # Error bound based on width
        epsilon = 2.0 / self.width

        # Error probability based on depth
        delta = 1.0 / (1 << self.depth)

        # Scale for confidence level
        if confidence > 0:
            delta /= confidence

        return epsilon * self.total
