//! Exact two-pass counting and PPMI from exact counts.
//!
//! Pass 1 counts every token. The caller then fixes a vocabulary. Pass 2
//! counts directional skip-gram pairs between vocabulary words only, while
//! still advancing the window over out-of-vocabulary tokens (positions are
//! those of the length-filtered token stream, exactly as
//! `nltk.util.skipgrams` sees it). The skip-gram total counts *all* pairs,
//! in- or out-of-vocabulary.

use crate::pairs::{PairKey, PairTable, OOV};
use crate::tokenize::{n_skipgrams, Tokenizer};
use rustc_hash::FxHashMap;

pub struct WordCounter {
    pub counts: FxHashMap<String, u64>,
    pub total: u64,
    tokenizer: Tokenizer,
    buf: String,
}

impl WordCounter {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self {
            counts: FxHashMap::default(),
            total: 0,
            tokenizer,
            buf: String::new(),
        }
    }

    pub fn add_line(&mut self, line: &str) {
        let Self {
            counts,
            total,
            tokenizer,
            buf,
        } = self;
        tokenizer.for_each(line, buf, |w| {
            *total += 1;
            if let Some(c) = counts.get_mut(w) {
                *c += 1;
            } else {
                counts.insert(w.to_owned(), 1);
            }
        });
    }

    /// `(word, count)` sorted by descending count, ties by word.
    pub fn sorted(self) -> Vec<(String, u64)> {
        let mut v: Vec<_> = self.counts.into_iter().collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }
}

/// Sequential exact pair counting (used for Python iterables).
pub struct PairCounter<K> {
    index: FxHashMap<String, u32>,
    pub table: PairTable<K>,
    pub total: u64,
    window: usize,
    tokenizer: Tokenizer,
    buf: String,
    ids: Vec<u32>,
}

/// Vocabulary word -> id.
pub fn vocab_index(vocabulary: &[String]) -> FxHashMap<String, u32> {
    vocabulary
        .iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), i as u32))
        .collect()
}

/// Ids of the kept tokens of `line` (OOV for out-of-vocabulary tokens).
#[inline]
pub fn line_ids(
    line: &str,
    tokenizer: &Tokenizer,
    index: &FxHashMap<String, u32>,
    buf: &mut String,
    ids: &mut Vec<u32>,
) {
    tokenizer.for_each(line, buf, |w| {
        ids.push(index.get(w).copied().unwrap_or(OOV))
    });
}

impl<K: PairKey> PairCounter<K> {
    pub fn new(vocabulary: &[String], window: usize, tokenizer: Tokenizer) -> Self {
        Self {
            index: vocab_index(vocabulary),
            table: PairTable::new(1),
            total: 0,
            window,
            tokenizer,
            buf: String::new(),
            ids: Vec::new(),
        }
    }

    pub fn add_line(&mut self, line: &str) {
        self.ids.clear();
        line_ids(
            line,
            &self.tokenizer,
            &self.index,
            &mut self.buf,
            &mut self.ids,
        );
        self.total += n_skipgrams(self.ids.len(), self.window);
        self.table.shards[0].add_line(&self.ids, self.window, 0, 1);
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

/// PPMI over exact counts, in O(nnz).
pub fn ppmi_from_counts(
    rows: &[i32],
    cols: &[i32],
    pair_counts: &[i64],
    word_counts: &[i64],
    p: &PpmiParams,
) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
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
        let tok = Tokenizer {
            kind: crate::tokenize::Kind::Whitespace,
            lowercase: false,
            min_len: 3,
        };
        let mut pc = PairCounter::<u32>::new(&vocab, 0, tok);
        // window 0 -> adjacent pairs only; "big" is OOV, so cat-dog are not adjacent
        pc.add_line("cat big dog");
        pc.add_line("cat dog");
        assert_eq!(pc.total, 3);
        let coo = pc.table.into_coo(1);
        assert_eq!(
            (coo.rows, coo.cols, coo.counts),
            (vec![0], vec![1], vec![1])
        );
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
