//! MurmurHash3 x86 32-bit, bit-compatible with Python's `mmh3.hash`.
//!
//! The sketch-mode PPMI kernel re-queries Count-Min Sketch tables that were
//! filled by the Python `CountMinSketch` (which hashes with `mmh3`), so the
//! row indices must match exactly.

const C1: u32 = 0xcc9e_2d51;
const C2: u32 = 0x1b87_3593;

#[inline]
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// Unsigned MurmurHash3_x86_32 of `key` with `seed`.
pub fn murmur3_32(key: &[u8], seed: u32) -> u32 {
    let mut h = seed;
    let (blocks, tail) = key.as_chunks::<4>();
    for block in blocks {
        let mut k = u32::from_le_bytes(*block);
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    let mut k: u32 = 0;
    if tail.len() >= 3 {
        k ^= (tail[2] as u32) << 16;
    }
    if tail.len() >= 2 {
        k ^= (tail[1] as u32) << 8;
    }
    if !tail.is_empty() {
        k ^= tail[0] as u32;
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
    }
    h ^= key.len() as u32;
    fmix32(h)
}

/// Row index exactly as `mmh3.hash(key, seed) % width` computes it in Python:
/// signed 32-bit hash, floor modulo (always non-negative for positive width).
#[inline]
pub fn sketch_index(key: &[u8], seed: u32, width: usize) -> usize {
    let signed = murmur3_32(key, seed) as i32 as i64;
    signed.rem_euclid(width as i64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from `mmh3.hash(key, seed)` (signed) in Python.
    #[test]
    fn matches_mmh3_reference_vectors() {
        assert_eq!(murmur3_32(b"", 0) as i32, 0);
        assert_eq!(murmur3_32(b"foo", 0) as i32, -156_908_512);
        assert_eq!(murmur3_32(b"Hello, world!", 0) as i32, -1_070_186_941);
        assert_eq!(murmur3_32(b"cat#dog", 42) as i32, 1_718_930_470);
        assert_eq!(murmur3_32("été".as_bytes(), 123_456) as i32, -1_051_994_264);
        assert_eq!(murmur3_32(b"abcd", 1) as i32, -1_678_424_686);
        assert_eq!(murmur3_32(b"abcde", 999_999) as i32, 166_100_363);
    }

    #[test]
    fn index_is_in_range() {
        for seed in [0u32, 1, 42, 999_999] {
            for key in [&b"cat"[..], b"dog#cat", b"\xc3\xa9t\xc3\xa9"] {
                assert!(sketch_index(key, seed, 7) < 7);
            }
        }
    }
}
