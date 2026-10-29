//! Compiled core of chronowords: counting and PPMI kernels.
//!
//! Exposed to Python as `chronowords._core`. Python-facing wrappers live in
//! `chronowords.utils`; this module only takes and returns plain arrays.

mod bounded;
mod exact;
mod pairs;
mod parallel;
mod sketch;
mod source;
mod tokenize;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyMemoryError, PyOverflowError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn require(cond: bool, msg: impl Into<String>) -> PyResult<()> {
    if cond {
        Ok(())
    } else {
        Err(PyValueError::new_err(msg.into()))
    }
}

type PpmiCoo<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
);
type WordCounts<'py> = (Vec<String>, Bound<'py, PyArray1<i64>>, u64);
/// `(rows, cols, counts, row_sums, col_sums, total)`.
type PairCounts<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
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

fn pair_counts_out<K: pairs::PairKey>(
    py: Python<'_>,
    table: pairs::PairTable<K>,
    total: u64,
    min_pair_count: u32,
) -> PyResult<PairCounts<'_>> {
    if table.overflowed() {
        return Err(PyOverflowError::new_err(
            "a pair count exceeded 2**32 - 1; the corpus is too large for 32-bit pair counts",
        ));
    }
    let coo = py.detach(|| table.into_coo(min_pair_count));
    Ok((
        coo.rows.into_pyarray(py),
        coo.cols.into_pyarray(py),
        coo.counts.into_pyarray(py),
        coo.row_sums.into_pyarray(py),
        coo.col_sums.into_pyarray(py),
        total,
    ))
}

fn make_tokenizer(name: &str, lowercase: bool, min_len: usize) -> PyResult<tokenize::Tokenizer> {
    let kind = tokenize::Kind::parse(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "tokenizer must be \"whitespace\" or \"unicode\", got {name:?}"
        ))
    })?;
    Ok(tokenize::Tokenizer {
        kind,
        lowercase,
        min_len,
    })
}

fn check_vocabulary(vocabulary: &[String]) -> PyResult<()> {
    let unique: rustc_hash::FxHashSet<&str> = vocabulary.iter().map(String::as_str).collect();
    require(
        unique.len() == vocabulary.len(),
        "vocabulary contains duplicates",
    )?;
    require(
        vocabulary.len() < pairs::OOV as usize,
        "vocabulary too large",
    )
}

/// Run `f` on a rayon pool of `n_threads` threads (0 = one per CPU core).
fn with_pool<T: Send>(n_threads: usize, f: impl FnOnce() -> T + Send) -> PyResult<T> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(pool.install(f))
}

/// Tokens of one line under the given tokenizer settings, exactly as the
/// counting functions see them.
#[pyfunction]
#[pyo3(name = "tokenize", signature = (line, min_word_length, tokenizer="whitespace", lowercase=false))]
fn tokenize_line(
    line: &str,
    min_word_length: usize,
    tokenizer: &str,
    lowercase: bool,
) -> PyResult<Vec<String>> {
    Ok(make_tokenizer(tokenizer, lowercase, min_word_length)?.tokens(line))
}

/// Exact token counts over a Python iterable of lines or token lists.
///
/// Returns ``(words, counts, total)`` with words sorted by descending count
/// (ties alphabetical) and ``total`` the number of kept tokens. If
/// ``spill_path`` is given, every item is also written there as one line, so
/// a single-use iterator can be counted a second time from disk.
#[pyfunction]
#[pyo3(signature = (corpus, min_word_length, spill_path=None, tokenizer="whitespace", lowercase=false))]
fn count_words<'py>(
    py: Python<'py>,
    corpus: &Bound<'py, PyAny>,
    min_word_length: usize,
    spill_path: Option<String>,
    tokenizer: &str,
    lowercase: bool,
) -> PyResult<WordCounts<'py>> {
    let tok = make_tokenizer(tokenizer, lowercase, min_word_length)?;
    let mut wc = exact::WordCounter::new(tok);
    let mut spill = spill_path
        .as_deref()
        .map(source::Spill::create)
        .transpose()?;
    source::for_each_py_line(corpus, tok.kind == tokenize::Kind::Whitespace, |line| {
        wc.add_line(line);
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

/// Exact token counts over UTF-8 text files (universal newlines), counted in
/// parallel on ``n_threads`` threads (0 = one per core). Releases the GIL.
#[pyfunction]
#[pyo3(signature = (paths, min_word_length, tokenizer="whitespace", lowercase=false, n_threads=0, chunk_bytes=parallel::CHUNK))]
fn count_words_files<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    min_word_length: usize,
    tokenizer: &str,
    lowercase: bool,
    n_threads: usize,
    chunk_bytes: u64,
) -> PyResult<WordCounts<'py>> {
    let tok = make_tokenizer(tokenizer, lowercase, min_word_length)?;
    let wc = py.detach(|| {
        with_pool(n_threads, || {
            parallel::count_words(&paths, tok, chunk_bytes)
        })
    })??;
    Ok(word_counts_out(py, wc))
}

fn seq_pairs<'py, K: pairs::PairKey>(
    py: Python<'py>,
    corpus: &Bound<'py, PyAny>,
    vocabulary: &[String],
    window: usize,
    tok: tokenize::Tokenizer,
    min_pair_count: u32,
) -> PyResult<PairCounts<'py>> {
    let mut pc = exact::PairCounter::<K>::new(vocabulary, window, tok);
    source::for_each_py_line(corpus, tok.kind == tokenize::Kind::Whitespace, |line| {
        pc.add_line(line);
        Ok(())
    })?;
    pair_counts_out(py, pc.table, pc.total, min_pair_count)
}

/// Exact directional skip-gram counts between vocabulary words.
///
/// Returns ``(rows, cols, counts, total)``: COO pair counts (int32 indices
/// into the vocabulary), sorted by row then column, keeping pairs observed at
/// least ``min_pair_count`` times; and the total number of skip-grams in the
/// corpus (including pairs with out-of-vocabulary words, and pruned pairs).
#[pyfunction]
#[pyo3(signature = (corpus, vocabulary, window, min_word_length, tokenizer="whitespace", lowercase=false, min_pair_count=1))]
#[allow(clippy::too_many_arguments)]
fn count_pairs<'py>(
    py: Python<'py>,
    corpus: &Bound<'py, PyAny>,
    vocabulary: Vec<String>,
    window: usize,
    min_word_length: usize,
    tokenizer: &str,
    lowercase: bool,
    min_pair_count: u32,
) -> PyResult<PairCounts<'py>> {
    let tok = make_tokenizer(tokenizer, lowercase, min_word_length)?;
    check_vocabulary(&vocabulary)?;
    if vocabulary.len() <= pairs::COMPACT_MAX_VOCAB {
        seq_pairs::<u32>(py, corpus, &vocabulary, window, tok, min_pair_count)
    } else {
        seq_pairs::<u64>(py, corpus, &vocabulary, window, tok, min_pair_count)
    }
}

struct PairJob<'a> {
    paths: &'a [String],
    vocabulary: &'a [String],
    window: usize,
    tok: tokenize::Tokenizer,
    min_pair_count: u32,
    n_threads: usize,
    chunk_bytes: u64,
}

fn par_pairs<'py, K: pairs::PairKey>(
    py: Python<'py>,
    job: PairJob<'_>,
) -> PyResult<PairCounts<'py>> {
    let (table, total) = py.detach(|| {
        with_pool(job.n_threads, || {
            parallel::count_pairs::<K>(
                job.paths,
                job.vocabulary,
                job.window,
                job.tok,
                rayon::current_num_threads() * parallel::SHARDS_PER_THREAD,
                job.chunk_bytes,
            )
        })
    })??;
    pair_counts_out(py, table, total, job.min_pair_count)
}

/// As ``count_pairs``, reading UTF-8 text files in parallel on ``n_threads``
/// threads (0 = one per core). Releases the GIL.
#[pyfunction]
#[pyo3(signature = (paths, vocabulary, window, min_word_length, tokenizer="whitespace", lowercase=false, min_pair_count=1, n_threads=0, chunk_bytes=parallel::CHUNK))]
#[allow(clippy::too_many_arguments)]
fn count_pairs_files<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    vocabulary: Vec<String>,
    window: usize,
    min_word_length: usize,
    tokenizer: &str,
    lowercase: bool,
    min_pair_count: u32,
    n_threads: usize,
    chunk_bytes: u64,
) -> PyResult<PairCounts<'py>> {
    let tok = make_tokenizer(tokenizer, lowercase, min_word_length)?;
    check_vocabulary(&vocabulary)?;
    let job = PairJob {
        paths: &paths,
        vocabulary: &vocabulary,
        window,
        tok,
        min_pair_count,
        n_threads,
        chunk_bytes,
    };
    if vocabulary.len() <= pairs::COMPACT_MAX_VOCAB {
        par_pairs::<u32>(py, job)
    } else {
        par_pairs::<u64>(py, job)
    }
}

impl From<bounded::BoundedError> for PyErr {
    fn from(e: bounded::BoundedError) -> PyErr {
        match e {
            bounded::BoundedError::Source(s) => s.into(),
            bounded::BoundedError::Budget(m) => PyMemoryError::new_err(m),
        }
    }
}

fn stats_dict<'py>(py: Python<'py>, st: &bounded::Stats) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("sketch_bytes", st.sketch_bytes)?;
    d.set_item("candidates", st.candidates)?;
    d.set_item("table_bytes", st.table_bytes)?;
    Ok(d)
}

/// Write a Python iterable of lines or token lists to ``path``, one item per
/// line, so a multi-pass reader can re-read it. Token lists are validated and
/// joined as in ``count_words``.
#[pyfunction]
#[pyo3(signature = (corpus, path, tokenizer="whitespace"))]
fn spill_corpus(corpus: &Bound<'_, PyAny>, path: &str, tokenizer: &str) -> PyResult<()> {
    let tok = make_tokenizer(tokenizer, false, 0)?;
    let mut spill = source::Spill::create(path)?;
    source::for_each_py_line(corpus, tok.kind == tokenize::Kind::Whitespace, |line| {
        spill.write_line(line)?;
        Ok(())
    })?;
    spill.finish()?;
    Ok(())
}

/// Exact counts of every token that can reach ``min_count``, under a memory
/// budget (sketch of ``sketch_bytes`` + exact table of at most
/// ``table_budget`` bytes). Returns ``(words, counts, total, stats)``.
#[pyfunction]
#[pyo3(signature = (paths, min_word_length, min_count, sketch_bytes, table_budget,
                    tokenizer="whitespace", lowercase=false, n_threads=0,
                    chunk_bytes=parallel::CHUNK))]
#[allow(clippy::too_many_arguments)]
fn count_words_bounded<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    min_word_length: usize,
    min_count: u64,
    sketch_bytes: usize,
    table_budget: usize,
    tokenizer: &str,
    lowercase: bool,
    n_threads: usize,
    chunk_bytes: u64,
) -> PyResult<BoundedWords<'py>> {
    let tok = make_tokenizer(tokenizer, lowercase, min_word_length)?;
    let (counts, total, st) = py.detach(|| {
        with_pool(n_threads, || {
            bounded::count_words(
                &paths,
                tok,
                chunk_bytes,
                sketch_bytes,
                table_budget,
                min_count,
            )
        })
    })??;
    let mut v: Vec<(String, u64)> = counts.into_iter().collect();
    v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let (words, counts): (Vec<String>, Vec<i64>) =
        v.into_iter().map(|(w, c)| (w, c as i64)).unzip();
    Ok((words, counts.into_pyarray(py), total, stats_dict(py, &st)?))
}

type BoundedWords<'py> = (
    Vec<String>,
    Bound<'py, PyArray1<i64>>,
    u64,
    Bound<'py, PyDict>,
);

type BoundedPairs<'py> = (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    u64,
    Bound<'py, PyDict>,
);

/// Exact counts of every pair that can reach ``min_pair_count``, exact
/// marginals and skip-gram total, under a memory budget. Returns
/// ``(rows, cols, counts, row_sums, col_sums, total, stats)``.
#[pyfunction]
#[pyo3(signature = (paths, vocabulary, window, min_word_length, min_pair_count,
                    sketch_bytes, table_budget, tokenizer="whitespace", lowercase=false,
                    n_threads=0, chunk_bytes=parallel::CHUNK))]
#[allow(clippy::too_many_arguments)]
fn count_pairs_bounded<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    vocabulary: Vec<String>,
    window: usize,
    min_word_length: usize,
    min_pair_count: u32,
    sketch_bytes: usize,
    table_budget: usize,
    tokenizer: &str,
    lowercase: bool,
    n_threads: usize,
    chunk_bytes: u64,
) -> PyResult<BoundedPairs<'py>> {
    let tok = make_tokenizer(tokenizer, lowercase, min_word_length)?;
    check_vocabulary(&vocabulary)?;
    let run = |compact: bool| {
        with_pool(n_threads, || {
            if compact {
                bounded::count_pairs::<u32>(
                    &paths,
                    &vocabulary,
                    window,
                    tok,
                    chunk_bytes,
                    sketch_bytes,
                    table_budget,
                    min_pair_count,
                )
            } else {
                bounded::count_pairs::<u64>(
                    &paths,
                    &vocabulary,
                    window,
                    tok,
                    chunk_bytes,
                    sketch_bytes,
                    table_budget,
                    min_pair_count,
                )
            }
        })
    };
    let compact = vocabulary.len() <= pairs::COMPACT_MAX_VOCAB;
    let (coo, total, st) = py.detach(|| run(compact))??;
    Ok((
        coo.rows.into_pyarray(py),
        coo.cols.into_pyarray(py),
        coo.counts.into_pyarray(py),
        coo.row_sums.into_pyarray(py),
        coo.col_sums.into_pyarray(py),
        total,
        stats_dict(py, &st)?,
    ))
}

/// PPMI with context-distribution smoothing (Levy, Goldberg & Dagan 2015),
/// from pair counts and the unpruned row/column sums, in O(nnz).
#[pyfunction]
#[pyo3(signature = (rows, cols, pair_counts, row_sums, col_sums,
                    shift=1.0, alpha=0.75, min_count=1))]
#[allow(clippy::too_many_arguments)]
fn ppmi_from_counts<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, i32>,
    cols: PyReadonlyArray1<'py, i32>,
    pair_counts: PyReadonlyArray1<'py, i64>,
    row_sums: PyReadonlyArray1<'py, i64>,
    col_sums: PyReadonlyArray1<'py, i64>,
    shift: f64,
    alpha: f64,
    min_count: i64,
) -> PyResult<PpmiCoo<'py>> {
    let (r, c, n, rs, cs) = (
        rows.as_slice()?,
        cols.as_slice()?,
        pair_counts.as_slice()?,
        row_sums.as_slice()?,
        col_sums.as_slice()?,
    );
    require(
        r.len() == c.len() && c.len() == n.len(),
        "rows, cols and pair_counts must have equal length",
    )?;
    require(
        rs.len() == cs.len(),
        "row_sums and col_sums must have equal length",
    )?;
    require(shift > 0.0, "shift must be positive")?;
    require(alpha > 0.0, "alpha must be positive")?;
    let v = rs.len() as i64;
    require(
        r.iter().chain(c).all(|&i| (0..v).contains(&(i as i64))),
        "pair index out of range of row_sums/col_sums",
    )?;
    require(
        (0..r.len()).all(|k| n[k] <= rs[r[k] as usize] && n[k] <= cs[c[k] as usize]),
        "a pair count exceeds its row or column sum",
    )?;
    let p = exact::PpmiParams {
        shift,
        alpha,
        min_count,
    };
    let (d, rr, cc) = py.detach(|| exact::ppmi_from_counts(r, c, n, rs, cs, &p));
    Ok((d.into_pyarray(py), rr.into_pyarray(py), cc.into_pyarray(py)))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_line, m)?)?;
    m.add_function(wrap_pyfunction!(count_words, m)?)?;
    m.add_function(wrap_pyfunction!(count_words_files, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs_files, m)?)?;
    m.add_function(wrap_pyfunction!(ppmi_from_counts, m)?)?;
    m.add_function(wrap_pyfunction!(spill_corpus, m)?)?;
    m.add_function(wrap_pyfunction!(count_words_bounded, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs_bounded, m)?)?;
    Ok(())
}
