//! Corpus line sources: files on disk and Python iterables.
//!
//! Files are read the way Python's text mode reads them with
//! `encoding="utf-8"`: strict UTF-8, universal newlines (`\n`, `\r\n` and a
//! lone `\r` all end a line). Python iterables yield either a `str` line or a
//! sequence of `str` tokens (pre-tokenised input), which is joined with
//! single spaces after checking that no token contains whitespace.

use crate::tokenize::is_python_space;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

#[derive(Debug)]
pub enum SourceError {
    Io(String),
    Utf8(String),
}

impl From<SourceError> for PyErr {
    fn from(e: SourceError) -> PyErr {
        match e {
            SourceError::Io(m) => pyo3::exceptions::PyOSError::new_err(m),
            SourceError::Utf8(m) => PyValueError::new_err(m),
        }
    }
}

/// Call `f` on every line of every file, in order.
pub fn for_each_file_line(paths: &[String], mut f: impl FnMut(&str)) -> Result<(), SourceError> {
    let mut buf = Vec::with_capacity(1 << 16);
    for path in paths {
        let file = File::open(path).map_err(|e| SourceError::Io(format!("{path}: {e}")))?;
        let mut reader = BufReader::with_capacity(1 << 20, file);
        let mut line_no = 0usize;
        loop {
            buf.clear();
            let n = reader
                .read_until(b'\n', &mut buf)
                .map_err(|e| SourceError::Io(format!("{path}: {e}")))?;
            if n == 0 {
                break;
            }
            // Universal newlines: a lone '\r' also ends a line. Splitting the
            // '\n'-terminated chunk on '\r' yields the same token streams;
            // empty pieces (e.g. from "\r\n") contain no tokens.
            for piece in buf.split(|&b| b == b'\r') {
                line_no += 1;
                let text = std::str::from_utf8(piece).map_err(|e| {
                    SourceError::Utf8(format!("{path}: invalid UTF-8 near line {line_no}: {e}"))
                })?;
                f(text);
            }
        }
    }
    Ok(())
}

/// One corpus item from Python as a line of text.
fn item_to_line(item: &Bound<'_, PyAny>, scratch: &mut String) -> PyResult<Option<PyBackedStr>> {
    if let Ok(s) = item.extract::<PyBackedStr>() {
        return Ok(Some(s));
    }
    let tokens = item.try_iter().map_err(|_| {
        PyTypeError::new_err(format!(
            "corpus items must be str or a sequence of str tokens, got {}",
            item.get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_default()
        ))
    })?;
    scratch.clear();
    for tok in tokens {
        let tok: PyBackedStr = tok?.extract()?;
        if tok.is_empty() || tok.chars().any(is_python_space) {
            return Err(PyValueError::new_err(format!(
                "pre-tokenised token {:?} is empty or contains whitespace",
                &*tok
            )));
        }
        if !scratch.is_empty() {
            scratch.push(' ');
        }
        scratch.push_str(&tok);
    }
    Ok(None)
}

/// Call `f` on every item of a Python iterable, as a line of text.
pub fn for_each_py_line(
    corpus: &Bound<'_, PyAny>,
    mut f: impl FnMut(&str) -> PyResult<()>,
) -> PyResult<()> {
    if corpus.extract::<PyBackedStr>().is_ok() {
        return Err(PyTypeError::new_err(
            "corpus is a single str; pass an iterable of lines, or a pathlib.Path for a file",
        ));
    }
    let mut scratch = String::new();
    for item in corpus.try_iter()? {
        match item_to_line(&item?, &mut scratch)? {
            Some(line) => f(&line)?,
            None => f(&scratch)?,
        }
    }
    Ok(())
}

/// Writes lines so that `for_each_file_line` reads back the same token
/// streams: '\n' and '\r' inside a line become spaces (both are whitespace
/// to `str.split()`, so tokens are unchanged) and each line ends in '\n'.
pub struct Spill {
    out: BufWriter<File>,
    path: String,
}

impl Spill {
    pub fn create(path: &str) -> Result<Self, SourceError> {
        let f = File::create(path).map_err(|e| SourceError::Io(format!("{path}: {e}")))?;
        Ok(Self {
            out: BufWriter::with_capacity(1 << 20, f),
            path: path.to_owned(),
        })
    }

    pub fn write_line(&mut self, line: &str) -> Result<(), SourceError> {
        let err = |e: std::io::Error| SourceError::Io(format!("{}: {e}", self.path));
        if line.contains(['\n', '\r']) {
            self.out
                .write_all(line.replace(['\n', '\r'], " ").as_bytes())
                .map_err(err)?;
        } else {
            self.out.write_all(line.as_bytes()).map_err(err)?;
        }
        self.out.write_all(b"\n").map_err(err)
    }

    pub fn finish(mut self) -> Result<(), SourceError> {
        self.out
            .flush()
            .map_err(|e| SourceError::Io(format!("{}: {e}", self.path)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines_of(bytes: &[u8]) -> Result<Vec<String>, SourceError> {
        let dir = std::env::temp_dir().join(format!("cw-src-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(format!(
            "{:x}.txt",
            bytes.len() * 7919 + bytes.first().copied().unwrap_or(0) as usize
        ));
        File::create(&p).unwrap().write_all(bytes).unwrap();
        let mut out = Vec::new();
        let r = for_each_file_line(&[p.to_string_lossy().into_owned()], |l| {
            out.push(l.to_owned())
        });
        std::fs::remove_file(&p).ok();
        r.map(|_| out)
    }

    #[test]
    fn universal_newlines() {
        let got = lines_of(b"a b\r\nc\rd\ne").unwrap();
        let non_empty: Vec<_> = got
            .iter()
            .map(|l| l.trim_end_matches('\n'))
            .filter(|l| !l.is_empty())
            .collect();
        assert_eq!(non_empty, ["a b", "c", "d", "e"]);
    }

    #[test]
    fn invalid_utf8_is_an_error() {
        match lines_of(b"ok\n\xff\xfe\n") {
            Err(SourceError::Utf8(m)) => assert!(m.contains("invalid UTF-8"), "{m}"),
            other => panic!("expected a UTF-8 error, got {other:?}"),
        }
    }
}
