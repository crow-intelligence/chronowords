//! Tokenisation: whitespace (Python `str.split()`) or Unicode words (UAX #29).

use unicode_segmentation::UnicodeSegmentation;

/// True where Python's `str.isspace()` is true.
///
/// Rust's `char::is_whitespace` is the Unicode `White_Space` property; Python
/// additionally treats the information separators U+001C..U+001F as
/// whitespace (bidi class B/S).
#[inline]
pub fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Exactly Python's `str.split()`.
    Whitespace,
    /// Unicode word boundaries (UAX #29), keeping only segments that contain
    /// a letter or digit: punctuation and whitespace are dropped, `can't`
    /// and `3.14` stay whole.
    Unicode,
}

impl Kind {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "whitespace" => Some(Kind::Whitespace),
            "unicode" => Some(Kind::Unicode),
            _ => None,
        }
    }
}

/// Splits lines into tokens, optionally lowercases them, then drops tokens
/// shorter than `min_len` code points (Python `len`, not UTF-8 bytes).
#[derive(Clone, Copy, Debug)]
pub struct Tokenizer {
    pub kind: Kind,
    pub lowercase: bool,
    pub min_len: usize,
}

#[inline]
fn long_enough(w: &str, min_len: usize) -> bool {
    !w.is_empty() && (min_len == 0 || w.chars().nth(min_len - 1).is_some())
}

impl Tokenizer {
    #[cfg(test)]
    pub fn whitespace(min_len: usize) -> Self {
        Self {
            kind: Kind::Whitespace,
            lowercase: false,
            min_len,
        }
    }

    /// Call `f` on each kept token of `line`. `buf` is scratch space reused
    /// across calls for lowercased tokens.
    pub fn for_each(&self, line: &str, buf: &mut String, mut f: impl FnMut(&str)) {
        let mut emit = |w: &str| {
            if self.lowercase && w.chars().any(char::is_uppercase) {
                buf.clear();
                if w.is_ascii() {
                    buf.push_str(w);
                    buf.make_ascii_lowercase();
                } else {
                    buf.push_str(&w.to_lowercase());
                }
                if long_enough(buf, self.min_len) {
                    f(buf);
                }
            } else if long_enough(w, self.min_len) {
                f(w);
            }
        };
        match self.kind {
            Kind::Whitespace => line.split(is_python_space).for_each(&mut emit),
            Kind::Unicode => line.unicode_words().for_each(&mut emit),
        }
    }

    /// Kept tokens of `line` as owned strings.
    pub fn tokens(&self, line: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut buf = String::new();
        self.for_each(line, &mut buf, |w| out.push(w.to_owned()));
        out
    }
}

/// Number of forward pairs `nltk.util.skipgrams(tokens, 2, window)` yields for
/// a line of `n` tokens: each position pairs with the next `window + 1`.
pub fn n_skipgrams(n: usize, window: usize) -> u64 {
    (0..n).map(|i| (n - 1 - i).min(window + 1) as u64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws(line: &str, min_len: usize) -> Vec<String> {
        Tokenizer::whitespace(min_len).tokens(line)
    }

    #[test]
    fn splits_like_python() {
        assert_eq!(
            ws("  the\tcat\u{1f}sat\u{a0}on \n", 0),
            ["the", "cat", "sat", "on"]
        );
    }

    #[test]
    fn length_is_code_points() {
        // "élő" is 3 code points but 5 UTF-8 bytes
        assert_eq!(ws("élő az ég", 3), ["élő"]);
    }

    #[test]
    fn unicode_words_drop_punctuation() {
        let t = Tokenizer {
            kind: Kind::Unicode,
            lowercase: true,
            min_len: 0,
        };
        assert_eq!(
            t.tokens("Hello, World! It's 3.14 — Élő\u{a0}ÁRVÍZTŰRŐ."),
            ["hello", "world", "it's", "3.14", "élő", "árvíztűrő"]
        );
    }

    #[test]
    fn lowercase_then_length_filter() {
        let t = Tokenizer {
            kind: Kind::Whitespace,
            lowercase: true,
            min_len: 3,
        };
        assert_eq!(t.tokens("The CAT is ÉDES"), ["the", "cat", "édes"]);
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
