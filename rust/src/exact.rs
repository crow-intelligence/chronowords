//! Exact two-pass counting and PPMI from exact counts.
//!
//! Pass 1 counts every token. The caller then fixes a vocabulary. Pass 2
//! counts directional skip-gram pairs between vocabulary words only, while
//! still advancing the window over out-of-vocabulary tokens (positions are
//! those of the length-filtered token stream, exactly as
//! `nltk.util.skipgrams` sees it). The skip-gram total counts *all* pairs,
//! in- or out-of-vocabulary, matching the sketch path's denominator.

use crate::tokenize::{n_skipgrams, tokens};
use rustc_hash::FxHashMap;

#[derive(Default)]
pub struct WordCounter {
    pub counts: FxHashMap<String, u64>,
    pub total: u64,
}

impl WordCounter {
    pub fn add_line(&mut self, line: &str, min_len: usize) {
        for w in tokens(line, min_len) {
            self.total += 1;
            if let Some(c) = self.counts.get_mut(w) {
                *c += 1;
            } else {
                self.counts.insert(w.to_owned(), 1);
            }
        }
    }

    /// `(word, count)` sorted by descending count, ties by word.
    pub fn sorted(self) -> Vec<(String, u64)> {
        let mut v: Vec<_> = self.counts.into_iter().collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }
}

pub struct PairCounter {
    index: FxHashMap<String, u32>,
    pub counts: FxHashMap<u64, u64>,
    pub total: u64,
    window: usize,
    min_len: usize,
    ids: Vec<Option<u32>>,
}

impl PairCounter {
    pub fn new(vocabulary: &[String], window: usize, min_len: usize) -> Self {
        let index = vocabulary
            .iter()
            .enumerate()
            .map(|(i, w)| (w.clone(), i as u32))
            .collect();
        Self {
            index,
            counts: FxHashMap::default(),
            total: 0,
            window,
            min_len,
            ids: Vec::new(),
        }
    }

    pub fn add_line(&mut self, line: &str) {
        self.ids.clear();
        for w in tokens(line, self.min_len) {
            self.ids.push(self.index.get(w).copied());
        }
        let n = self.ids.len();
        self.total += n_skipgrams(n, self.window);
        for i in 0..n {
            let Some(a) = self.ids[i] else { continue };
            let hi = (i + self.window + 1).min(n - 1);
            for j in (i + 1)..=hi {
                if let Some(b) = self.ids[j] {
                    *self
                        .counts
                        .entry(((a as u64) << 32) | b as u64)
                        .or_insert(0) += 1;
                }
            }
        }
    }

    /// COO `(rows, cols, counts)` sorted by row then column.
    pub fn into_coo(self) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let mut v: Vec<_> = self.counts.into_iter().collect();
        v.sort_unstable_by_key(|&(k, _)| k);
        let rows = v.iter().map(|&(k, _)| (k >> 32) as i64).collect();
        let cols = v.iter().map(|&(k, _)| (k & 0xffff_ffff) as i64).collect();
        let counts = v.iter().map(|&(_, c)| c as i64).collect();
        (rows, cols, counts)
    }
}

pub struct PpmiParams {
    pub word_total: f64,
    pub skip_total: f64,
    pub shift: f64,
    pub alpha: f64,
    /// Words and pairs need a count strictly greater than this.
    pub min_count: i64,
}

/// PPMI over exact counts, in O(nnz). Same formula as the sketch kernel.
pub fn ppmi_from_counts(
    rows: &[i64],
    cols: &[i64],
    pair_counts: &[i64],
    word_counts: &[i64],
    p: &PpmiParams,
) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
    const EPS: f64 = 1e-10;
    let log_shift = p.shift.ln();
    let prob = |c: i64| {
        if c > p.min_count {
            c as f64 / p.word_total
        } else {
            0.0
        }
    };
    let (mut data, mut out_r, mut out_c) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..rows.len() {
        let (i, j, pair) = (rows[k], cols[k], pair_counts[k]);
        if pair <= p.min_count {
            continue;
        }
        let (fi, fj) = (prob(word_counts[i as usize]), prob(word_counts[j as usize]));
        if fi == 0.0 || fj == 0.0 {
            continue;
        }
        let pa = (fi + EPS).powf(p.alpha);
        let pb = fj + EPS;
        let pab = (pair as f64 + EPS) / p.skip_total;
        if pab > pa * pb {
            let pmi = (pab / (pa * pb)).ln() - log_shift;
            if pmi > 0.0 {
                data.push(pmi);
                out_r.push(i);
                out_c.push(j);
            }
        }
    }
    (data, out_r, out_c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_counts_skip_oov_but_keep_positions() {
        let vocab = vec!["cat".to_string(), "dog".to_string()];
        let mut pc = PairCounter::new(&vocab, 0, 3);
        // window 0 -> adjacent pairs only; "big" is OOV, so cat-dog are not adjacent
        pc.add_line("cat big dog");
        pc.add_line("cat dog");
        assert_eq!(pc.total, 3);
        let (r, c, n) = pc.into_coo();
        assert_eq!((r, c, n), (vec![0], vec![1], vec![1]));
    }

    #[test]
    fn golden_cell() {
        // six "cat dog" lines: PPMI(cat, dog) = 1.75 ln 2
        let p = PpmiParams {
            word_total: 12.0,
            skip_total: 6.0,
            shift: 1.0,
            alpha: 0.75,
            min_count: 5,
        };
        let (d, r, c) = ppmi_from_counts(&[0], &[1], &[6], &[6, 6], &p);
        assert_eq!((r, c), (vec![0], vec![1]));
        assert!((d[0] - 1.75 * 2f64.ln()).abs() < 1e-8);
    }
}
