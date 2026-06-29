import mmh3
import numpy as np


class CountMinSketch:
    """Count-Min Sketch implementation for memory-efficient counting.

    Uses ``depth`` hash functions over ``width`` counters each to approximate
    item frequencies in fixed memory. Queries never underestimate the true
    count; they may overestimate it due to hash collisions.

    - Memory usage: ``width * depth * 4`` bytes (int32 counters).
    - Error bound: an overestimate of about ``2 / width`` of the total count,
      with probability at least ``1 - 1 / 2**depth``.

    Examples:
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
            width: Number of counters per hash function (controls accuracy).
                Must be a positive integer.
            depth: Number of hash functions / rows (controls the probability
                bound). Must be a positive integer.
            seed: Seed for deriving the per-row hash seeds; fixes the sketch's
                hashing so that two sketches with the same ``seed`` (and
                ``width``/``depth``) are merge-compatible.
            track_keys: Whether to record observed keys so
                :meth:`get_heavy_hitters` can enumerate them. Disable to save
                memory; :meth:`get_heavy_hitters` then raises.

        Note:
            Arguments are not validated. ``width``/``depth`` must be positive
            or the underlying ``numpy.zeros((depth, width))`` allocation fails.

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
            key: Item to count. ``str`` keys are UTF-8 encoded; ``bytes`` keys
                are used as-is (and decoded for key tracking).
            count: Amount to increment by (default 1). Added to ``total`` and
                to each row counter as-is; no positivity check is performed.

        Examples:
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
        """Query the estimated count for a key.

        Args:
            key: Item to look up (``str`` is UTF-8 encoded; ``bytes`` used
                as-is).

        Returns:
            The minimum counter across rows, which is the Count-Min Sketch
            estimate. This never underestimates the true count and returns 0
            for an unseen key (barring collisions).

        Examples:
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
        """Get items that appear more than ``threshold * total`` times.

        Args:
            threshold: Minimum frequency as a fraction of the total count,
                normally in (0, 1). Not validated; the comparison threshold is
                ``int(total * threshold)`` (truncated toward zero).

        Returns:
            ``(item, count)`` pairs whose estimated count is strictly greater
            than ``int(total * threshold)``, sorted by descending count.
            Counts are CMS estimates, so a returned count may overestimate the
            true value (and a borderline item may be a false positive), but no
            genuine heavy hitter is missed.

        Raises:
            RuntimeError: If the sketch was created with ``track_keys=False``,
                since observed keys are then not retained.

        Examples:
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
        """Merge another sketch into this one, in place.

        Adds ``other``'s counters and total into ``self`` and unions the
        tracked keys. Because both sketches share hashing parameters, the
        result is identical to a single sketch built from the concatenation of
        the two input streams.

        Args:
            other: Another sketch with the same ``width``, ``depth`` and
                derived ``seeds`` as ``self``.

        Raises:
            ValueError: If ``other`` is not merge-compatible (differing
                ``width``, ``depth`` or ``seeds``).

        Examples:
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
        """Estimate the maximum counting error.

        Args:
            confidence: Intended confidence level for the bound.

        Returns:
            The expected maximum overestimate, ``(2 / width) * total``.

        Note:
            The ``confidence`` argument currently has **no effect** on the
            returned value: an internal ``delta`` term is computed from
            ``confidence`` but discarded before the return. The result depends
            only on ``width`` and ``total``. Flagged in the project pre-mortem;
            kept as-is to preserve behaviour.

        Examples:
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
        """Get raw arrays and parameters for the Cython PPMI kernel.

        Returns:
            A tuple ``(counts, seeds, width)`` exposing the internal count
            table (shape ``(depth, width)``), the per-row hash seeds, and the
            table width — the inputs :class:`~chronowords.utils.count_skipgrams.PPMIComputer`
            needs to re-query the sketch.

        Examples:
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
