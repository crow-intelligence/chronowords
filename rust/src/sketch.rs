//! Count-Min Sketch with conservative update, used only as a *filter*.
//!
//! A Count-Min Sketch never under-estimates a count (Cormode & Muthukrishnan
//! 2005); conservative update (Estan & Varghese 2002) keeps that guarantee
//! while lowering the over-estimate. So "estimate < threshold" proves
//! "true count < threshold": the bounded-memory path uses a sketch to decide
//! which words or pairs *might* reach a threshold and then counts only those
//! exactly. Final counts are never taken from the sketch.
//!
//! The sketch is split into shards by key (the same partition the counting
//! workers use), so each shard is updated by one thread, sequentially, and
//! conservative update stays exact; the shards' sizes sum to the budget.

/// Rows per sketch.
pub const DEPTH: usize = 4;

#[inline]
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// One Count-Min Sketch of `DEPTH` rows by `width` saturating `u32` counters.
pub struct CountMin {
    width: usize,
    seeds: [u64; DEPTH],
    counters: Vec<u32>,
}

impl CountMin {
    pub fn new(width: usize, seed: u64) -> Self {
        let width = width.max(1);
        let mut seeds = [0u64; DEPTH];
        for (r, s) in seeds.iter_mut().enumerate() {
            *s = splitmix64(seed ^ (r as u64).wrapping_mul(0x632b_e59b_d9b4_e019));
        }
        Self {
            width,
            seeds,
            counters: vec![0; width * DEPTH],
        }
    }

    #[inline]
    fn slots(&self, key: u64) -> [usize; DEPTH] {
        let mut out = [0usize; DEPTH];
        for (r, o) in out.iter_mut().enumerate() {
            let h = splitmix64(key ^ self.seeds[r]);
            *o = r * self.width + ((h as u128 * self.width as u128) >> 64) as usize;
        }
        out
    }

    /// Conservative update: raise only the counters holding the current
    /// minimum. Saturates at `u32::MAX`, which still reads as "large".
    #[inline]
    pub fn add(&mut self, key: u64) {
        let slots = self.slots(key);
        let min = slots.iter().map(|&i| self.counters[i]).min().unwrap_or(0);
        if min == u32::MAX {
            return;
        }
        for &i in &slots {
            if self.counters[i] == min {
                self.counters[i] = min + 1;
            }
        }
    }

    /// Upper bound on the number of times `key` was added.
    #[inline]
    pub fn estimate(&self, key: u64) -> u32 {
        self.slots(key)
            .iter()
            .map(|&i| self.counters[i])
            .min()
            .unwrap_or(0)
    }

    pub fn bytes(&self) -> usize {
        self.counters.len() * std::mem::size_of::<u32>()
    }
}

/// Sketches partitioned by `shard_of(key) % n_shards`; total size = budget.
pub struct ShardedSketch {
    pub shards: Vec<CountMin>,
}

impl ShardedSketch {
    pub fn with_budget(bytes: usize, n_shards: usize, seed: u64) -> Self {
        let n = n_shards.max(1);
        let width = (bytes / (n * DEPTH * std::mem::size_of::<u32>())).max(1);
        Self {
            shards: (0..n)
                .map(|s| CountMin::new(width, seed.wrapping_add(s as u64)))
                .collect(),
        }
    }

    #[inline]
    pub fn estimate(&self, shard: usize, key: u64) -> u32 {
        self.shards[shard].estimate(key)
    }

    pub fn bytes(&self) -> usize {
        self.shards.iter().map(CountMin::bytes).sum()
    }
}

/// 64-bit hash of a token, the key under which it is sketched.
#[inline]
pub fn word_key(word: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    word.hash(&mut h);
    splitmix64(h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_underestimates_even_when_tiny() {
        // width 1: every key collides, estimates are large but never too small
        for width in [1usize, 2, 7, 64] {
            let mut cm = CountMin::new(width, 42);
            let mut truth = std::collections::HashMap::new();
            for i in 0..2_000u64 {
                let key = splitmix64(i) % 300;
                cm.add(key);
                *truth.entry(key).or_insert(0u32) += 1;
            }
            for (&k, &n) in &truth {
                assert!(
                    cm.estimate(k) >= n,
                    "width {width}: {} < {n}",
                    cm.estimate(k)
                );
            }
        }
    }

    #[test]
    fn wide_sketch_is_exact_for_few_keys() {
        let mut cm = CountMin::new(1 << 16, 7);
        for _ in 0..5 {
            cm.add(1);
        }
        cm.add(2);
        assert_eq!((cm.estimate(1), cm.estimate(2), cm.estimate(3)), (5, 1, 0));
    }

    #[test]
    fn budget_is_respected() {
        let s = ShardedSketch::with_budget(1 << 20, 8, 1);
        assert!(s.bytes() <= 1 << 20);
        assert!(s.bytes() > (1 << 20) - 8 * DEPTH * 4);
    }
}
