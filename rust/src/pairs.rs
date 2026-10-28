//! Sharded exact pair counts with compact keys.
//!
//! A pair `(a, b)` of vocabulary ids is packed into one integer key, `a` in
//! the high bits, so sorting keys sorts pairs by row then column. With at
//! most 65,536 vocabulary words the key is a `u32` (8 bytes per map entry
//! with the `u32` count), otherwise a `u64`. Shard `s` holds the pairs whose
//! focus id satisfies `a % n_shards == s`, so shards can be filled in
//! parallel without locks and without merging.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::hash::Hash;

/// Marks an out-of-vocabulary token in an id sequence.
pub const OOV: u32 = u32::MAX;

pub trait PairKey: Copy + Eq + Hash + Ord + Send + Sync {
    fn pack(a: u32, b: u32) -> Self;
    fn unpack(self) -> (u32, u32);
}

impl PairKey for u32 {
    #[inline]
    fn pack(a: u32, b: u32) -> Self {
        (a << 16) | b
    }
    #[inline]
    fn unpack(self) -> (u32, u32) {
        (self >> 16, self & 0xffff)
    }
}

impl PairKey for u64 {
    #[inline]
    fn pack(a: u32, b: u32) -> Self {
        ((a as u64) << 32) | b as u64
    }
    #[inline]
    fn unpack(self) -> (u32, u32) {
        ((self >> 32) as u32, self as u32)
    }
}

/// Largest vocabulary that fits the compact `u32` key.
pub const COMPACT_MAX_VOCAB: usize = 1 << 16;

pub struct Shard<K> {
    pub map: FxHashMap<K, u32>,
    pub overflow: bool,
}

impl<K: PairKey> Shard<K> {
    fn new() -> Self {
        Self {
            map: FxHashMap::default(),
            overflow: false,
        }
    }

    #[inline]
    fn add(&mut self, a: u32, b: u32) {
        let c = self.map.entry(K::pack(a, b)).or_insert(0);
        match c.checked_add(1) {
            Some(v) => *c = v,
            None => self.overflow = true,
        }
    }

    /// Add every in-vocabulary pair of one line whose focus id belongs to
    /// this shard (`a % n_shards == shard`).
    #[inline]
    pub fn add_line(&mut self, ids: &[u32], window: usize, shard: u32, n_shards: u32) {
        let n = ids.len();
        for i in 0..n {
            let a = ids[i];
            if a == OOV || a % n_shards != shard {
                continue;
            }
            let hi = (i + window + 1).min(n - 1);
            for &b in &ids[i + 1..=hi] {
                if b != OOV {
                    self.add(a, b);
                }
            }
        }
    }
}

pub struct PairTable<K> {
    pub shards: Vec<Shard<K>>,
}

/// COO pair counts, sorted by row then column.
pub struct Coo {
    pub rows: Vec<i32>,
    pub cols: Vec<i32>,
    pub counts: Vec<i64>,
}

impl<K: PairKey> PairTable<K> {
    pub fn new(n_shards: usize) -> Self {
        Self {
            shards: (0..n_shards.max(1)).map(|_| Shard::new()).collect(),
        }
    }

    pub fn overflowed(&self) -> bool {
        self.shards.iter().any(|s| s.overflow)
    }

    /// Distinct pairs currently held, across shards.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.map.len()).sum()
    }

    /// Drain into sorted COO arrays, keeping pairs with count >= `min_count`.
    pub fn into_coo(self, min_count: u32) -> Coo {
        let mut v: Vec<(K, u32)> = Vec::with_capacity(self.len());
        for shard in self.shards {
            v.extend(shard.map.into_iter().filter(|&(_, c)| c >= min_count));
        }
        v.par_sort_unstable_by_key(|&(k, _)| k);
        let mut coo = Coo {
            rows: Vec::with_capacity(v.len()),
            cols: Vec::with_capacity(v.len()),
            counts: Vec::with_capacity(v.len()),
        };
        for (k, c) in v {
            let (a, b) = k.unpack();
            coo.rows.push(a as i32);
            coo.cols.push(b as i32);
            coo.counts.push(c as i64);
        }
        coo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_round_trip_and_sort_by_row_then_col() {
        for (a, b) in [(0u32, 0u32), (1, 65_535), (65_535, 1), (300, 7)] {
            assert_eq!(<u32 as PairKey>::pack(a, b).unpack(), (a, b));
            assert_eq!(<u64 as PairKey>::pack(a, b).unpack(), (a, b));
        }
        assert!(<u32 as PairKey>::pack(1, 0) > <u32 as PairKey>::pack(0, 65_535));
        assert_eq!(<u64 as PairKey>::pack(70_000, 3).unpack(), (70_000, 3));
    }

    #[test]
    fn shards_partition_pairs_by_focus() {
        let ids = [0, 1, OOV, 2, 3];
        let mut whole = PairTable::<u32>::new(1);
        whole.shards[0].add_line(&ids, 1, 0, 1);
        let mut split = PairTable::<u32>::new(2);
        for (s, shard) in split.shards.iter_mut().enumerate() {
            shard.add_line(&ids, 1, s as u32, 2);
        }
        let (w, p) = (whole.into_coo(1), split.into_coo(1));
        assert_eq!((w.rows, w.cols, w.counts), (p.rows, p.cols, p.counts));
    }

    #[test]
    fn min_count_prunes() {
        let mut t = PairTable::<u64>::new(1);
        t.shards[0].add_line(&[0, 1, 0, 1, 0, 2], 0, 0, 1);
        let coo = t.into_coo(2);
        assert_eq!(
            (coo.rows, coo.cols, coo.counts),
            (vec![0, 1], vec![1, 0], vec![2, 2])
        );
    }
}
