//! Compiled core of chronowords: counting and PPMI kernels.
//!
//! Exposed to Python as `chronowords._core`. Python-facing wrappers live in
//! `chronowords.utils`; this module only takes and returns plain arrays.

mod exact;
mod murmur;
mod sketch;
mod source;
mod tokenize;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type Coo<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
);

fn coo<'py>(py: Python<'py>, (d, r, c): (Vec<f64>, Vec<i64>, Vec<i64>)) -> Coo<'py> {
    (d.into_pyarray(py), r.into_pyarray(py), c.into_pyarray(py))
}

fn require(cond: bool, msg: impl Into<String>) -> PyResult<()> {
    if cond {
        Ok(())
    } else {
        Err(PyValueError::new_err(msg.into()))
    }
}

/// Signed MurmurHash3 x86 32-bit, identical to ``mmh3.hash(key, seed)``.
#[pyfunction]
fn murmur3_32(key: &[u8], seed: u32) -> i32 {
    murmur::murmur3_32(key, seed) as i32
}

/// PPMI COO entries for focus words ``[start, end)`` from Count-Min Sketch tables.
#[pyfunction]
#[pyo3(signature = (skipgram_counts, word_counts, vocabulary, seeds, width,
                    skip_total, word_total, shift, alpha, start, end, min_count=5))]
#[allow(clippy::too_many_arguments)]
fn sketch_ppmi<'py>(
    py: Python<'py>,
    skipgram_counts: PyReadonlyArray2<'py, i32>,
    word_counts: PyReadonlyArray2<'py, i32>,
    vocabulary: Vec<String>,
    seeds: Vec<u32>,
    width: usize,
    skip_total: f64,
    word_total: f64,
    shift: f64,
    alpha: f64,
    start: usize,
    end: usize,
    min_count: i32,
) -> PyResult<Coo<'py>> {
    let (skip, word) = (skipgram_counts.as_array(), word_counts.as_array());
    require(width > 0, "width must be positive")?;
    require(!seeds.is_empty(), "seeds must not be empty")?;
    for (name, t) in [("skipgram_counts", &skip), ("word_counts", &word)] {
        require(
            t.shape() == [seeds.len(), width],
            format!(
                "{name} has shape {:?}; expected (len(seeds), width) = ({}, {width})",
                t.shape(),
                seeds.len()
            ),
        )?;
    }
    let inp = sketch::SketchInputs {
        skipgram_counts: skip,
        word_counts: word,
        vocabulary: &vocabulary,
        seeds: &seeds,
        width,
        skip_total,
        word_total,
        shift,
        alpha,
        min_count,
    };
    let out = py.detach(|| sketch::ppmi_rows(&inp, start, end));
    Ok(coo(py, out))
}

type WordCounts<'py> = (Vec<String>, Bound<'py, PyArray1<i64>>, u64);
type PairCounts<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    u64,
);

fn word_counts_out(py: Python<'_>, wc: exact::WordCounter) -> WordCounts<'_> {
    let total = wc.total;
    let (words, counts): (Vec<String>, Vec<i64>) =
        wc.sorted().into_iter().map(|(w, c)| (w, c as i64)).unzip();
    (words, counts.into_pyarray(py), total)
}

fn pair_counts_out(py: Python<'_>, pc: exact::PairCounter) -> PairCounts<'_> {
    let total = pc.total;
    let (r, c, n) = pc.into_coo();
    (
        r.into_pyarray(py),
        c.into_pyarray(py),
        n.into_pyarray(py),
        total,
    )
}

fn pair_counter(
    vocabulary: &[String],
    window: usize,
    min_len: usize,
) -> PyResult<exact::PairCounter> {
    let unique: rustc_hash::FxHashSet<&str> = vocabulary.iter().map(String::as_str).collect();
    require(
        unique.len() == vocabulary.len(),
        "vocabulary contains duplicates",
    )?;
    Ok(exact::PairCounter::new(vocabulary, window, min_len))
}

/// Exact token counts over a Python iterable of lines or token lists.
///
/// Returns ``(words, counts, total)`` with words sorted by descending count
/// (ties alphabetical) and ``total`` the number of kept tokens. If
/// ``spill_path`` is given, every item is also written there as one line, so
/// a single-use iterator can be counted a second time from disk.
#[pyfunction]
#[pyo3(signature = (corpus, min_word_length, spill_path=None))]
fn count_words<'py>(
    py: Python<'py>,
    corpus: &Bound<'py, PyAny>,
    min_word_length: usize,
    spill_path: Option<String>,
) -> PyResult<WordCounts<'py>> {
    let mut wc = exact::WordCounter::default();
    let mut spill = spill_path
        .as_deref()
        .map(source::Spill::create)
        .transpose()?;
    source::for_each_py_line(corpus, |line| {
        wc.add_line(line, min_word_length);
        if let Some(s) = spill.as_mut() {
            s.write_line(line)?;
        }
        Ok(())
    })?;
    if let Some(s) = spill {
        s.finish()?;
    }
    Ok(word_counts_out(py, wc))
}

/// Exact token counts over UTF-8 text files (universal newlines). Releases the GIL.
#[pyfunction]
fn count_words_files(
    py: Python<'_>,
    paths: Vec<String>,
    min_word_length: usize,
) -> PyResult<WordCounts<'_>> {
    let wc = py.detach(|| {
        let mut wc = exact::WordCounter::default();
        source::for_each_file_line(&paths, |line| wc.add_line(line, min_word_length)).map(|_| wc)
    })?;
    Ok(word_counts_out(py, wc))
}

/// Exact directional skip-gram counts between vocabulary words.
///
/// Returns ``(rows, cols, counts, total)``: COO pair counts indexed by
/// vocabulary position, sorted by row then column, and the total number of
/// skip-grams in the corpus (including pairs with out-of-vocabulary words).
#[pyfunction]
fn count_pairs<'py>(
    py: Python<'py>,
    corpus: &Bound<'py, PyAny>,
    vocabulary: Vec<String>,
    window: usize,
    min_word_length: usize,
) -> PyResult<PairCounts<'py>> {
    let mut pc = pair_counter(&vocabulary, window, min_word_length)?;
    source::for_each_py_line(corpus, |line| {
        pc.add_line(line);
        Ok(())
    })?;
    Ok(pair_counts_out(py, pc))
}

/// As ``count_pairs``, reading UTF-8 text files. Releases the GIL.
#[pyfunction]
fn count_pairs_files(
    py: Python<'_>,
    paths: Vec<String>,
    vocabulary: Vec<String>,
    window: usize,
    min_word_length: usize,
) -> PyResult<PairCounts<'_>> {
    let mut pc = pair_counter(&vocabulary, window, min_word_length)?;
    py.detach(|| source::for_each_file_line(&paths, |line| pc.add_line(line)))?;
    Ok(pair_counts_out(py, pc))
}

/// PPMI COO entries from exact pair and word counts, in O(nnz).
#[pyfunction]
#[pyo3(signature = (rows, cols, pair_counts, word_counts, word_total, skip_total,
                    shift=1.0, alpha=0.75, min_count=5))]
#[allow(clippy::too_many_arguments)]
fn ppmi_from_counts<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i64>,
    cols: PyReadonlyArray1<'py, i64>,
    pair_counts: PyReadonlyArray1<'py, i64>,
    word_counts: PyReadonlyArray1<'py, i64>,
    word_total: f64,
    skip_total: f64,
    shift: f64,
    alpha: f64,
    min_count: i64,
) -> PyResult<Coo<'py>> {
    let (r, c, n, w) = (
        rows.as_slice()?,
        cols.as_slice()?,
        pair_counts.as_slice()?,
        word_counts.as_slice()?,
    );
    require(
        r.len() == c.len() && c.len() == n.len(),
        "rows, cols and pair_counts must have equal length",
    )?;
    require(word_total > 0.0, "word_total must be positive")?;
    require(skip_total > 0.0, "skip_total must be positive")?;
    require(shift > 0.0, "shift must be positive")?;
    let v = w.len() as i64;
    require(
        r.iter().chain(c).all(|&i| (0..v).contains(&i)),
        "pair index out of range of word_counts",
    )?;
    let p = exact::PpmiParams {
        word_total,
        skip_total,
        shift,
        alpha,
        min_count,
    };
    let out = py.detach(|| exact::ppmi_from_counts(r, c, n, w, &p));
    Ok(coo(py, out))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(murmur3_32, m)?)?;
    m.add_function(wrap_pyfunction!(sketch_ppmi, m)?)?;
    m.add_function(wrap_pyfunction!(count_words, m)?)?;
    m.add_function(wrap_pyfunction!(count_words_files, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs_files, m)?)?;
    m.add_function(wrap_pyfunction!(ppmi_from_counts, m)?)?;
    Ok(())
}
