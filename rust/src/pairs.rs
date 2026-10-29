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
    /// Pair-count marginals over *all* in-vocabulary pairs this shard saw,
    /// before any pruning: `row[a]` = pairs with focus `a`, `col[b]` = pairs
    /// with context `b`.
    pub row: Vec<u64>,
    pub col: Vec<u64>,
}

impl<K: PairKey> Shard<K> {
    fn new(vocab_size: usize) -> Self {
        Self {
            map: FxHashMap::default(),
            overflow: false,
            row: vec![0; vocab_size],
            col: vec![0; vocab_size],
        }
    }

    #[inline]
    fn add(&mut self, a: u32, b: u32) {
        self.row[a as usize] += 1;
        self.col[b as usize] += 1;
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

impl<K: PairKey> Shard<K> {
    /// Like [`Shard::add_line`], but count a pair only if `keep(a, b)` and
    /// without touching the marginals (the bounded path counts those in its
    /// sketch pass, over every pair).
    #[inline]
    pub fn add_line_filtered(
        &mut self,
        ids: &[u32],
        window: usize,
        shard: u32,
        n_shards: u32,
        keep: impl Fn(u32, u32) -> bool,
    ) {
        let n = ids.len();
        for i in 0..n {
            let a = ids[i];
            if a == OOV || a % n_shards != shard {
                continue;
            }
            let hi = (i + window + 1).min(n - 1);
            for &b in &ids[i + 1..=hi] {
                if b != OOV && keep(a, b) {
                    let c = self.map.entry(K::pack(a, b)).or_insert(0);
                    match c.checked_add(1) {
                        Some(v) => *c = v,
                        None => self.overflow = true,
                    }
                }
            }
        }
    }

    /// Approximate heap bytes held by the map (capacity x entry + control byte).
    pub fn table_bytes(&self) -> usize {
        self.map.capacity() * (std::mem::size_of::<(K, u32)>() + 1)
    }
}

pub struct PairTable<K> {
    pub shards: Vec<Shard<K>>,
}

/// COO pair counts, sorted by row then column, plus the unpruned marginals.
pub struct Coo {
    pub rows: Vec<i32>,
    pub cols: Vec<i32>,
    pub counts: Vec<i64>,
    /// Total in-vocabulary pairs per focus word (row sums of the full matrix).
    pub row_sums: Vec<i64>,
    /// Total in-vocabulary pairs per context word (column sums).
    pub col_sums: Vec<i64>,
}

impl<K: PairKey> PairTable<K> {
    pub fn new(n_shards: usize, vocab_size: usize) -> Self {
        Self {
            shards: (0..n_shards.max(1))
                .map(|_| Shard::new(vocab_size))
                .collect(),
        }
    }

    pub fn overflowed(&self) -> bool {
        self.shards.iter().any(|s| s.overflow)
    }

    /// Drain into sorted COO arrays, keeping pairs with count >= `min_count`.
    ///
    /// Shards are freed one by one as they are drained, and the buffer is
    /// sized to the kept pairs only, so pruning lowers the peak as well.
    pub fn into_coo(self, min_count: u32) -> Coo {
        let v_size = self.shards.first().map_or(0, |s| s.row.len());
        let (mut row_sums, mut col_sums) = (vec![0i64; v_size], vec![0i64; v_size]);
        for shard in &self.shards {
            for (t, &x) in row_sums.iter_mut().zip(&shard.row) {
                *t += x as i64;
            }
            for (t, &x) in col_sums.iter_mut().zip(&shard.col) {
                *t += x as i64;
            }
        }
        let kept: usize = self
            .shards
            .par_iter()
            .map(|s| s.map.values().filter(|&&c| c >= min_count).count())
            .sum();
        let mut v: Vec<(K, u32)> = Vec::with_capacity(kept);
        for shard in self.shards {
            v.extend(shard.map.into_iter().filter(|&(_, c)| c >= min_count));
        }
        v.par_sort_unstable_by_key(|&(k, _)| k);
        let mut coo = Coo {
            rows: Vec::with_capacity(v.len()),
            cols: Vec::with_capacity(v.len()),
            counts: Vec::with_capacity(v.len()),
            row_sums,
            col_sums,
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
        let mut whole = PairTable::<u32>::new(1, 4);
        whole.shards[0].add_line(&ids, 1, 0, 1);
        let mut split = PairTable::<u32>::new(2, 4);
        for (s, shard) in split.shards.iter_mut().enumerate() {
            shard.add_line(&ids, 1, s as u32, 2);
        }
        let (w, p) = (whole.into_coo(1), split.into_coo(1));
        assert_eq!((&w.row_sums, &w.col_sums), (&p.row_sums, &p.col_sums));
        assert_eq!((w.rows, w.cols, w.counts), (p.rows, p.cols, p.counts));
    }

    #[test]
    fn min_count_prunes() {
        let mut t = PairTable::<u64>::new(1, 3);
        t.shards[0].add_line(&[0, 1, 0, 1, 0, 2], 0, 0, 1);
        let coo = t.into_coo(2);
        // marginals count the pruned (0, 2) pair too
        assert_eq!(
            (coo.row_sums.clone(), coo.col_sums.clone()),
            (vec![3, 2, 0], vec![2, 2, 1])
        );
        assert_eq!(
            (coo.rows, coo.cols, coo.counts),
            (vec![0, 1], vec![1, 0], vec![2, 2])
        );
    }
}
