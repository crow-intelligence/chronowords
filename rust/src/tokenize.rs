//! Whitespace tokenisation matching Python's `str.split()` and `len()`.

/// True where Python's `str.isspace()` is true.
///
/// Rust's `char::is_whitespace` is the Unicode `White_Space` property; Python
/// additionally treats the information separators U+001C..U+001F as
/// whitespace (bidi class B/S).
#[inline]
pub fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Tokens of `line` as `line.split()` yields them, keeping only those with at
/// least `min_len` code points (Python `len`, not UTF-8 bytes).
pub fn tokens(line: &str, min_len: usize) -> impl Iterator<Item = &str> {
    line.split(is_python_space)
        .filter(move |w| !w.is_empty() && (min_len == 0 || w.chars().nth(min_len - 1).is_some()))
}

/// Number of forward pairs `nltk.util.skipgrams(tokens, 2, window)` yields for
/// a line of `n` tokens: each position pairs with the next `window + 1`.
pub fn n_skipgrams(n: usize, window: usize) -> u64 {
    (0..n).map(|i| (n - 1 - i).min(window + 1) as u64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_python() {
        let got: Vec<_> = tokens("  the\tcat\u{1f}sat\u{a0}on \n", 0).collect();
        assert_eq!(got, ["the", "cat", "sat", "on"]);
    }

    #[test]
    fn length_is_code_points() {
        // "élő" is 3 code points but 5 UTF-8 bytes
        let got: Vec<_> = tokens("élő az ég", 3).collect();
        assert_eq!(got, ["élő"]);
    }

    #[test]
    fn skipgram_count_matches_nltk() {
        // nltk: 9 tokens, k=2 -> 21 pairs
        assert_eq!(n_skipgrams(9, 2), 21);
        assert_eq!(n_skipgrams(0, 5), 0);
        assert_eq!(n_skipgrams(1, 5), 0);
        assert_eq!(n_skipgrams(2, 5), 1);
    }
}
