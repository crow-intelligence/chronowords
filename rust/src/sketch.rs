//! Sketch-mode PPMI: a port of the Cython `PPMIComputer` kernel.
//!
//! Re-queries Count-Min Sketch tables (filled by the Python `CountMinSketch`)
//! for every vocabulary pair. O(V^2 * depth) hash lookups; kept as the
//! bounded-memory mode, not the default.

use crate::murmur::sketch_index;
use numpy::ndarray::ArrayView2;

pub const EPS: f64 = 1e-10;

fn min_count(table: &ArrayView2<i32>, key: &[u8], seeds: &[u32], width: usize) -> i32 {
    let mut best = i32::MAX;
    for (row, &seed) in seeds.iter().enumerate() {
        let c = table[[row, sketch_index(key, seed, width)]];
        if c < best {
            best = c;
        }
    }
    best
}

pub struct SketchInputs<'a> {
    pub skipgram_counts: ArrayView2<'a, i32>,
    pub word_counts: ArrayView2<'a, i32>,
    pub vocabulary: &'a [String],
    pub seeds: &'a [u32],
    pub width: usize,
    pub skip_total: f64,
    pub word_total: f64,
    pub shift: f64,
    pub alpha: f64,
    /// Words and pairs need a sketch count strictly greater than this.
    pub min_count: i32,
}

/// Raw marginal P(w) per vocabulary word, or 0.0 when its count is <= 5.
pub fn word_probabilities(inp: &SketchInputs) -> Vec<f64> {
    inp.vocabulary
        .iter()
        .map(|w| {
            let c = min_count(&inp.word_counts, w.as_bytes(), inp.seeds, inp.width);
            if c > inp.min_count {
                c as f64 / inp.word_total
            } else {
                0.0
            }
        })
        .collect()
}

/// COO entries `(data, rows, cols)` for focus words in `[start, end)`.
pub fn ppmi_rows(inp: &SketchInputs, start: usize, end: usize) -> (Vec<f64>, Vec<i64>, Vec<i64>) {
    let probs = word_probabilities(inp);
    let n = inp.vocabulary.len();
    let log_shift = inp.shift.ln();
    let (mut data, mut rows, mut cols) = (Vec::new(), Vec::new(), Vec::new());
    let mut key = Vec::<u8>::new();

    for i in start..end.min(n) {
        if probs[i] == 0.0 {
            continue;
        }
        // Context-distribution smoothing on the focus word only; intentional.
        let pa = (probs[i] + EPS).powf(inp.alpha);
        let wi = inp.vocabulary[i].as_bytes();
        for (j, &pj) in probs.iter().enumerate() {
            if pj == 0.0 {
                continue;
            }
            key.clear();
            key.extend_from_slice(wi);
            key.push(b'#');
            key.extend_from_slice(inp.vocabulary[j].as_bytes());
            let pair = min_count(&inp.skipgram_counts, &key, inp.seeds, inp.width);
            if pair <= inp.min_count {
                continue;
            }
            let pb = pj + EPS;
            let pab = (pair as f64 + EPS) / inp.skip_total;
            if pab > pa * pb {
                let pmi = (pab / (pa * pb)).ln() - log_shift;
                if pmi > 0.0 {
                    data.push(pmi);
                    rows.push(i as i64);
                    cols.push(j as i64);
                }
            }
        }
    }
    (data, rows, cols)
}
