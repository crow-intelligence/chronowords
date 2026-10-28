from collections.abc import Iterable
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

_Coo = tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]

def murmur3_32(key: bytes, seed: int) -> int: ...
def tokenize(
    line: str,
    min_word_length: int,
    tokenizer: str = "whitespace",
    lowercase: bool = False,
) -> list[str]: ...
def sketch_ppmi(
    skipgram_counts: NDArray[np.int32],
    word_counts: NDArray[np.int32],
    vocabulary: list[str],
    seeds: list[int],
    width: int,
    skip_total: float,
    word_total: float,
    shift: float,
    alpha: float,
    start: int,
    end: int,
    min_count: int = 5,
) -> _Coo: ...

_WordCounts = tuple[list[str], NDArray[np.int64], int]
_PairCounts = tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int64], int]

def count_words(
    corpus: Iterable[str] | Iterable[Sequence[str]],
    min_word_length: int,
    spill_path: str | None = None,
    tokenizer: str = "whitespace",
    lowercase: bool = False,
) -> _WordCounts: ...
def count_words_files(
    paths: list[str],
    min_word_length: int,
    tokenizer: str = "whitespace",
    lowercase: bool = False,
    n_threads: int = 0,
    chunk_bytes: int = ...,
) -> _WordCounts: ...
def count_pairs(
    corpus: Iterable[str] | Iterable[Sequence[str]],
    vocabulary: list[str],
    window: int,
    min_word_length: int,
    tokenizer: str = "whitespace",
    lowercase: bool = False,
    min_pair_count: int = 1,
) -> _PairCounts: ...
def count_pairs_files(
    paths: list[str],
    vocabulary: list[str],
    window: int,
    min_word_length: int,
    tokenizer: str = "whitespace",
    lowercase: bool = False,
    min_pair_count: int = 1,
    n_threads: int = 0,
    chunk_bytes: int = ...,
) -> _PairCounts: ...
def ppmi_from_counts(
    rows: NDArray[np.int32],
    cols: NDArray[np.int32],
    pair_counts: NDArray[np.int64],
    word_counts: NDArray[np.int64],
    word_total: float,
    skip_total: float,
    shift: float = 1.0,
    alpha: float = 0.75,
    min_count: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]: ...
