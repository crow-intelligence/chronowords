//! Parallel exact counting over files.
//!
//! Files are cut into chunks that end just after a `'\n'`, so every line
//! (and every UTF-8 sequence, and every `"\r\n"`) lies in exactly one chunk.
//! Lines are independent contexts, so counts do not depend on how the corpus
//! is chunked or in which order chunks are processed.
//!
//! - Pass 1: each worker counts words of its chunks into a local map; maps
//!   are merged.
//! - Pass 2: chunks are processed in waves. First every chunk of the wave is
//!   tokenised to vocabulary ids in parallel; then each worker owns one shard
//!   of the pair table (pairs whose focus id is `shard` modulo the number of
//!   shards) and scans the whole wave, inserting only its own pairs. The pair
//!   table therefore exists once, not once per thread.

use crate::exact::{line_ids, vocab_index, WordCounter};
use crate::pairs::{PairKey, PairTable};
use crate::source::SourceError;
use crate::tokenize::{n_skipgrams, Tokenizer};
use rayon::prelude::*;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

/// Default target chunk size in bytes.
pub const CHUNK: u64 = 16 << 20;

/// Pair-table shards per worker thread. Each worker fills its shards one at a
/// time, so at most one shard per thread is resizing at any moment; more,
/// smaller shards keep that transient (old + new table) small.
pub const SHARDS_PER_THREAD: usize = 8;

#[derive(Clone, Debug)]
pub struct Chunk {
    path: usize,
    start: u64,
    len: u64,
}

fn io_err(path: &str, e: std::io::Error) -> SourceError {
    SourceError::Io(format!("{path}: {e}"))
}

/// Offset just past the first `'\n'` at or after `pos`, or the file size.
fn next_line_start(f: &mut File, pos: u64, size: u64, path: &str) -> Result<u64, SourceError> {
    f.seek(SeekFrom::Start(pos)).map_err(|e| io_err(path, e))?;
    let mut buf = [0u8; 1 << 16];
    let mut at = pos;
    while at < size {
        let n = f.read(&mut buf).map_err(|e| io_err(path, e))?;
        if n == 0 {
            break;
        }
        if let Some(i) = buf[..n].iter().position(|&b| b == b'\n') {
            return Ok(at + i as u64 + 1);
        }
        at += n as u64;
    }
    Ok(size)
}

pub fn chunks(paths: &[String], target: u64) -> Result<Vec<Chunk>, SourceError> {
    let mut out = Vec::new();
    for (pi, path) in paths.iter().enumerate() {
        let mut f = File::open(path).map_err(|e| io_err(path, e))?;
        let size = f.metadata().map_err(|e| io_err(path, e))?.len();
        let mut start = 0;
        while start < size {
            let end = if size - start <= target {
                size
            } else {
                next_line_start(&mut f, start + target, size, path)?
            };
            out.push(Chunk {
                path: pi,
                start,
                len: end - start,
            });
            start = end;
        }
    }
    Ok(out)
}

/// Read a chunk and call `f` on each of its lines (universal newlines, strict
/// UTF-8), exactly as `source::for_each_file_line` would.
pub(crate) fn for_each_chunk_line(
    paths: &[String],
    c: &Chunk,
    mut f: impl FnMut(&str),
) -> Result<(), SourceError> {
    let path = &paths[c.path];
    let mut file = File::open(path).map_err(|e| io_err(path, e))?;
    file.seek(SeekFrom::Start(c.start))
        .map_err(|e| io_err(path, e))?;
    let mut buf = vec![0u8; c.len as usize];
    file.read_exact(&mut buf).map_err(|e| io_err(path, e))?;
    let mut offset = c.start;
    for line in buf.split_inclusive(|&b| b == b'\n') {
        let mut piece_off = offset;
        for piece in line.split(|&b| b == b'\r') {
            let text = std::str::from_utf8(piece).map_err(|e| {
                SourceError::Utf8(format!(
                    "{path}: invalid UTF-8 at byte {}: {e}",
                    piece_off + e.valid_up_to() as u64
                ))
            })?;
            f(text);
            piece_off += piece.len() as u64 + 1;
        }
        offset += line.len() as u64;
    }
    Ok(())
}

/// Exact word counts over files, in parallel.
pub fn count_words(
    paths: &[String],
    tok: Tokenizer,
    chunk_bytes: u64,
) -> Result<WordCounter, SourceError> {
    let chunks = chunks(paths, chunk_bytes.max(1))?;
    chunks
        .par_iter()
        .try_fold(
            || WordCounter::new(tok),
            |mut wc, c| {
                for_each_chunk_line(paths, c, |line| wc.add_line(line))?;
                Ok(wc)
            },
        )
        .try_reduce(
            || WordCounter::new(tok),
            |mut a, mut b| {
                if a.counts.len() < b.counts.len() {
                    std::mem::swap(&mut a, &mut b);
                }
                a.total += b.total;
                for (w, c) in b.counts {
                    *a.counts.entry(w).or_insert(0) += c;
                }
                Ok(a)
            },
        )
}

/// Vocabulary ids of every line of one chunk, flattened, with line ends.
pub(crate) struct ChunkIds {
    pub(crate) ids: Vec<u32>,
    pub(crate) ends: Vec<usize>,
    pub(crate) skipgrams: u64,
}

/// Exact pair counts over files, in parallel. Returns the table and the
/// total number of skip-grams.
pub fn count_pairs<K: PairKey>(
    paths: &[String],
    vocabulary: &[String],
    window: usize,
    tok: Tokenizer,
    n_shards: usize,
    chunk_bytes: u64,
) -> Result<(PairTable<K>, u64), SourceError> {
    let index = vocab_index(vocabulary);
    let chunks = chunks(paths, chunk_bytes.max(1))?;
    let mut table = PairTable::<K>::new(n_shards, vocabulary.len());
    let n_shards = table.shards.len() as u32;
    let mut total = 0u64;
    let wave = (2 * rayon::current_num_threads()).max(2);

    for group in chunks.chunks(wave) {
        let ids: Vec<ChunkIds> = group
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
                    line_ids(line, &tok, &index, &mut buf, &mut out.ids);
                    out.skipgrams += n_skipgrams(out.ids.len() - before, window);
                    out.ends.push(out.ids.len());
                })?;
                Ok(out)
            })
            .collect::<Result<_, SourceError>>()?;
        total += ids.iter().map(|c| c.skipgrams).sum::<u64>();

        table
            .shards
            .par_iter_mut()
            .enumerate()
            .for_each(|(s, shard)| {
                for c in &ids {
                    let mut start = 0;
                    for &end in &c.ends {
                        shard.add_line(&c.ids[start..end], window, s as u32, n_shards);
                        start = end;
                    }
                }
            });
    }
    Ok((table, total))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmpfile(name: &str, bytes: &[u8]) -> String {
        let dir = std::env::temp_dir().join(format!("cw-par-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        File::create(&p).unwrap().write_all(bytes).unwrap();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn chunks_cover_the_file_at_line_starts() {
        let text = b"aa bb\r\ncc\ndd ee ff\n\ngg\rhh\nlast";
        let p = tmpfile("chunks.txt", text);
        for target in 1..=text.len() as u64 + 1 {
            let cs = chunks(std::slice::from_ref(&p), target).unwrap();
            let mut at = 0;
            for c in &cs {
                assert_eq!(c.start, at);
                assert!(c.start == 0 || text[c.start as usize - 1] == b'\n');
                at += c.len;
            }
            assert_eq!(at, text.len() as u64);
        }
    }

    #[test]
    fn parallel_word_counts_do_not_depend_on_chunking() {
        let p = tmpfile("words.txt", "a b a\r\nc a\rb\n\u{e9}t\u{e9} a\n".as_bytes());
        let tok = Tokenizer {
            kind: crate::tokenize::Kind::Whitespace,
            lowercase: false,
            min_len: 0,
        };
        let mut seq = WordCounter::new(tok);
        crate::source::for_each_file_line(std::slice::from_ref(&p), |l| seq.add_line(l)).unwrap();
        let par = count_words(std::slice::from_ref(&p), tok, 3).unwrap();
        assert_eq!(seq.total, par.total);
        assert_eq!(seq.sorted(), par.sorted());
    }
}
