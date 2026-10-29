//! Bounded-memory exact counting: a sketch decides, exact counts follow.
//!
//! Four passes over the files instead of two:
//!
//! 1. **1a** sketch every token; **1b** count exactly only tokens whose
//!    sketch estimate is at least `min_count` (a superset of the true
//!    vocabulary candidates, since estimates never fall short).
//! 2. **2a** sketch every in-vocabulary pair and accumulate the exact row
//!    and column sums and the skip-gram total; **2b** count exactly only
//!    pairs whose estimate is at least `min_pair_count`.
//!
//! Every word and pair that clears its threshold therefore gets exactly the
//! count the unbounded path gives it, whatever the sketch size: a smaller
//! sketch only lets more false candidates into the exact table. Exceeding the
//! table budget is an error, not a silent fallback.

use crate::exact::{line_ids, vocab_index};
use crate::pairs::{Coo, PairKey, PairTable};
use crate::parallel::{chunks, for_each_chunk_line, ChunkIds, SHARDS_PER_THREAD};
use crate::sketch::{word_key, ShardedSketch};
use crate::source::SourceError;
use crate::tokenize::{n_skipgrams, Tokenizer};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Debug)]
pub enum BoundedError {
    Source(SourceError),
    Budget(String),
}

impl From<SourceError> for BoundedError {
    fn from(e: SourceError) -> Self {
        BoundedError::Source(e)
    }
}

/// What the bounded path did, for the training record.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub sketch_bytes: usize,
    /// Distinct keys counted exactly (sketch candidates, incl. false positives).
    pub candidates: usize,
    /// Peak bytes held by the exact table (approximate).
    pub table_bytes: usize,
}

fn n_shards() -> usize {
    rayon::current_num_threads() * SHARDS_PER_THREAD
}

fn word_table_bytes(m: &FxHashMap<String, u64>) -> usize {
    m.capacity() * (std::mem::size_of::<(String, u64)>() + 1)
        + m.keys().map(String::capacity).sum::<usize>()
}

/// Pass 1 under a budget: exact counts of every token that can reach
/// `min_count`, and the total number of kept tokens.
pub fn count_words(
    paths: &[String],
    tok: Tokenizer,
    chunk_bytes: u64,
    sketch_bytes: usize,
    table_budget: usize,
    min_count: u64,
) -> Result<(FxHashMap<String, u64>, u64, Stats), BoundedError> {
    let chunks = chunks(paths, chunk_bytes.max(1))?;
    let shards = n_shards();
    let mut sketch = ShardedSketch::with_budget(sketch_bytes, shards, 0x5eed_0001);
    let wave = (2 * rayon::current_num_threads()).max(2);
    let mut total = 0u64;

    // 1a: sketch all tokens
    for group in chunks.chunks(wave) {
        let keys: Vec<Vec<u64>> = group
            .par_iter()
            .map(|c| {
                let mut out = Vec::new();
                let mut buf = String::new();
                for_each_chunk_line(paths, c, |line| {
                    tok.for_each(line, &mut buf, |w| out.push(word_key(w)));
                })?;
                Ok(out)
            })
            .collect::<Result<_, SourceError>>()?;
        total += keys.iter().map(|k| k.len() as u64).sum::<u64>();
        sketch
            .shards
            .par_iter_mut()
            .enumerate()
            .for_each(|(s, cm)| {
                for chunk in &keys {
                    for &k in chunk {
                        if k as usize % shards == s {
                            cm.add(k);
                        }
                    }
                }
            });
    }

    // 1b: exact counts of candidates only
    let threshold = min_count.min(u32::MAX as u64) as u32;
    let sketch = &sketch;
    let counts = chunks
        .par_iter()
        .try_fold(FxHashMap::<String, u64>::default, |mut m, c| {
            let mut buf = String::new();
            for_each_chunk_line(paths, c, |line| {
                tok.for_each(line, &mut buf, |w| {
                    let k = word_key(w);
                    if sketch.estimate(k as usize % shards, k) >= threshold {
                        if let Some(n) = m.get_mut(w) {
                            *n += 1;
                        } else {
                            m.insert(w.to_owned(), 1);
                        }
                    }
                });
            })?;
            Ok::<_, SourceError>(m)
        })
        .try_reduce(FxHashMap::default, |mut a, mut b| {
            if a.len() < b.len() {
                std::mem::swap(&mut a, &mut b);
            }
            for (w, n) in b {
                *a.entry(w).or_insert(0) += n;
            }
            Ok(a)
        })?;
    let stats = Stats {
        sketch_bytes: sketch.bytes(),
        candidates: counts.len(),
        table_bytes: word_table_bytes(&counts),
    };
    if stats.table_bytes > table_budget {
        return Err(BoundedError::Budget(format!(
            "{} candidate words need about {} MB, over the {} MB table budget; \
             raise max_memory or min_count",
            stats.candidates,
            stats.table_bytes >> 20,
            table_budget >> 20
        )));
    }
    Ok((counts, total, stats))
}

#[inline]
fn pair_key(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

fn tokenise_wave(
    paths: &[String],
    group: &[crate::parallel::Chunk],
    tok: &Tokenizer,
    index: &FxHashMap<String, u32>,
    window: usize,
) -> Result<Vec<ChunkIds>, SourceError> {
    group
        .par_iter()
        .map(|c| {
            let mut out = ChunkIds {
                ids: Vec::new(),
                ends: Vec::new(),
                skipgrams: 0,
            };
            let mut buf = String::new();
            for_each_chunk_line(paths, c, |line| {
                let before = out.ids.len();
                line_ids(line, tok, index, &mut buf, &mut out.ids);
                out.skipgrams += n_skipgrams(out.ids.len() - before, window);
                out.ends.push(out.ids.len());
            })?;
            Ok(out)
        })
        .collect()
}

/// Pass 2 under a budget: exact counts of every pair that can reach
/// `min_pair_count`, exact marginals over all pairs, and the skip-gram total.
#[allow(clippy::too_many_arguments)]
pub fn count_pairs<K: PairKey>(
    paths: &[String],
    vocabulary: &[String],
    window: usize,
    tok: Tokenizer,
    chunk_bytes: u64,
    sketch_bytes: usize,
    table_budget: usize,
    min_pair_count: u32,
) -> Result<(Coo, u64, Stats), BoundedError> {
    let index = vocab_index(vocabulary);
    let chunks = chunks(paths, chunk_bytes.max(1))?;
    let shards = n_shards();
    let n = shards as u32;
    let v = vocabulary.len();
    let wave = (2 * rayon::current_num_threads()).max(2);
    let mut sketch = ShardedSketch::with_budget(sketch_bytes, shards, 0x5eed_0002);
    let mut total = 0u64;
    let (mut row_sums, mut col_sums) = (vec![0i64; v], vec![0i64; v]);

    // 2a: sketch every in-vocabulary pair; exact marginals
    for group in chunks.chunks(wave) {
        let ids = tokenise_wave(paths, group, &tok, &index, window)?;
        total += ids.iter().map(|c| c.skipgrams).sum::<u64>();
        let marginals: Vec<(Vec<i64>, Vec<i64>)> = sketch
            .shards
            .par_iter_mut()
            .enumerate()
            .map(|(s, cm)| {
                let (mut rs, mut cs) = (vec![0i64; v], vec![0i64; v]);
                for c in &ids {
                    let mut start = 0;
                    for &end in &c.ends {
                        let line = &c.ids[start..end];
                        start = end;
                        for i in 0..line.len() {
                            let a = line[i];
                            if a == crate::pairs::OOV || a % n != s as u32 {
                                continue;
                            }
                            let hi = (i + window + 1).min(line.len() - 1);
                            for &b in &line[i + 1..=hi] {
                                if b != crate::pairs::OOV {
                                    cm.add(pair_key(a, b));
                                    rs[a as usize] += 1;
                                    cs[b as usize] += 1;
                                }
                            }
                        }
                    }
                }
                (rs, cs)
            })
            .collect();
        for (rs, cs) in marginals {
            for (t, x) in row_sums.iter_mut().zip(rs) {
                *t += x;
            }
            for (t, x) in col_sums.iter_mut().zip(cs) {
                *t += x;
            }
        }
    }

    // 2b: exact counts of candidate pairs only
    let mut table = PairTable::<K>::new(shards, 0);
    let mut peak = 0usize;
    for group in chunks.chunks(wave) {
        let ids = tokenise_wave(paths, group, &tok, &index, window)?;
        let sk = &sketch;
        table
            .shards
            .par_iter_mut()
            .enumerate()
            .for_each(|(s, shard)| {
                let keep = |a: u32, b: u32| sk.estimate(s, pair_key(a, b)) >= min_pair_count;
                for c in &ids {
                    let mut start = 0;
                    for &end in &c.ends {
                        shard.add_line_filtered(&c.ids[start..end], window, s as u32, n, keep);
                        start = end;
                    }
                }
            });
        let bytes: usize = table.shards.iter().map(|s| s.table_bytes()).sum();
        peak = peak.max(bytes);
        if bytes > table_budget {
            let entries: usize = table.shards.iter().map(|s| s.map.len()).sum();
            return Err(BoundedError::Budget(format!(
                "{entries} candidate pairs need about {} MB, over the {} MB table \
                 budget; raise max_memory, min_count or ppmi_min_count",
                bytes >> 20,
                table_budget >> 20
            )));
        }
    }
    let stats = Stats {
        sketch_bytes: sketch.bytes(),
        candidates: table.shards.iter().map(|s| s.map.len()).sum(),
        table_bytes: peak,
    };
    drop(sketch);
    if table.overflowed() {
        return Err(BoundedError::Budget(
            "a pair count exceeded 2**32 - 1".to_string(),
        ));
    }
    let mut coo = table.into_coo(min_pair_count);
    coo.row_sums = row_sums;
    coo.col_sums = col_sums;
    Ok((coo, total, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::Kind;
    use std::io::Write;

    fn tmpfile(name: &str, text: &str) -> String {
        let dir = std::env::temp_dir().join(format!("cw-bounded-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::File::create(&p)
            .unwrap()
            .write_all(text.as_bytes())
            .unwrap();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn any_sketch_size_gives_exact_counts() {
        let text = "a b a c\nb a b\na a b c d\n".repeat(7);
        let p = tmpfile("t.txt", &text);
        let paths = std::slice::from_ref(&p);
        let tok = Tokenizer {
            kind: Kind::Whitespace,
            lowercase: false,
            min_len: 0,
        };
        let mut wc = crate::exact::WordCounter::new(tok);
        for line in text.lines() {
            wc.add_line(line);
        }
        let vocab: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let mut exact = crate::exact::PairCounter::<u32>::new(&vocab, 1, tok);
        for line in text.lines() {
            exact.add_line(line);
        }
        let exact_total = exact.total;
        let want = exact.table.into_coo(8);
        // from a single 16-byte sketch (everything collides) to a roomy one
        for sketch_bytes in [16usize, 64, 1024, 1 << 20] {
            let (got, total, _) = count_words(paths, tok, 5, sketch_bytes, 1 << 20, 7).unwrap();
            assert_eq!(total, wc.total);
            for (w, &n) in &wc.counts {
                if n >= 7 {
                    assert_eq!(got.get(w), Some(&n), "sketch {sketch_bytes}: {w}");
                }
            }
            let (coo, total, _) =
                count_pairs::<u32>(paths, &vocab, 1, tok, 5, sketch_bytes, 1 << 20, 8).unwrap();
            assert_eq!(total, exact_total);
            assert_eq!(
                (&coo.rows, &coo.cols, &coo.counts),
                (&want.rows, &want.cols, &want.counts)
            );
            assert_eq!(
                (&coo.row_sums, &coo.col_sums),
                (&want.row_sums, &want.col_sums)
            );
        }
    }

    #[test]
    fn table_budget_overrun_is_an_error() {
        let p = tmpfile("u.txt", &"a b c d e f g h\n".repeat(50));
        let tok = Tokenizer {
            kind: Kind::Whitespace,
            lowercase: false,
            min_len: 0,
        };
        let err = count_words(std::slice::from_ref(&p), tok, 1 << 20, 1 << 10, 8, 1);
        assert!(matches!(err, Err(BoundedError::Budget(_))));
    }
}
