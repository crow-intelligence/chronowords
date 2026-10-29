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
            table: PairTable::new(1, vocabulary.len()),
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
    /// PMI shift: `log(shift)` is subtracted (shifted PPMI). Must be > 0.
    pub shift: f64,
    /// Context-distribution smoothing exponent. Must be > 0; 1 disables it.
    pub alpha: f64,
    /// A pair needs a count strictly greater than this.
    pub min_count: i64,
}

/// PPMI with context-distribution smoothing, as in Levy, Goldberg & Dagan
/// (2015) and their `hyperwords` toolkit, in O(nnz).
///
/// With `n(w, c)` the pair count, `row(w)` and `col(c)` the row and column
/// sums of the *full* (unpruned) in-vocabulary co-occurrence matrix, and
/// `Z = sum over c' of col(c')^alpha`:
///
/// `PPMI(w, c) = max(0, log(n(w, c) * Z / (row(w) * col(c)^alpha)) - log(shift))`
///
/// i.e. `P(w) = row(w) / D`, `P_alpha(c) = col(c)^alpha / Z` and
/// `P(w, c) = n(w, c) / D`, where the total `D` cancels.
pub fn ppmi_from_counts(
    rows: &[i32],
    cols: &[i32],
    pair_counts: &[i64],
    row_sums: &[i64],
    col_sums: &[i64],
    p: &PpmiParams,
) -> (Vec<f64>, Vec<i32>, Vec<i32>) {
    let z: f64 = col_sums
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| (c as f64).powf(p.alpha))
        .sum();
    let (ln_z, ln_shift) = (z.ln(), p.shift.ln());
    let (mut data, mut out_r, mut out_c) = (Vec::new(), Vec::new(), Vec::new());
    for k in 0..rows.len() {
        let (i, j, n) = (rows[k], cols[k], pair_counts[k]);
        if n <= p.min_count {
            continue;
        }
        // n > 0 implies row(i) > 0 and col(j) > 0
        let (r, c) = (row_sums[i as usize] as f64, col_sums[j as usize] as f64);
        let pmi = (n as f64).ln() + ln_z - r.ln() - p.alpha * c.ln() - ln_shift;
        if pmi > 0.0 {
            data.push(pmi);
            out_r.push(i);
            out_c.push(j);
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
    fn golden_cells() {
        // pairs (cat,dog)=6, (fish,dog)=6, (cat,bird)=6; ids cat=0 dog=1 bird=2 fish=3
        // rows: cat 12, fish 6; cols: dog 12, bird 6
        let (r, c, n) = (vec![0, 0, 3], vec![1, 2, 1], vec![6, 6, 6]);
        let (rs, cs) = (vec![12, 0, 0, 6], vec![0, 12, 6, 0]);
        let p = PpmiParams {
            shift: 1.0,
            alpha: 0.75,
            min_count: 5,
        };
        let (d, rr, cc) = ppmi_from_counts(&r, &c, &n, &rs, &cs, &p);
        // (cat, dog): log(6 Z / (12 * 12^0.75)) < 0, dropped
        assert_eq!((rr, cc), (vec![0, 3], vec![2, 1]));
        // (cat, bird) = log((2^0.75 + 1) / 2); (fish, dog) = log(1 + 2^-0.75)
        assert!((d[0] - ((2f64.powf(0.75) + 1.0) / 2.0).ln()).abs() < 1e-12);
        assert!((d[1] - (1.0 + 2f64.powf(-0.75)).ln()).abs() < 1e-12);
    }
}
