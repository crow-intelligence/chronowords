from collections.abc import Iterable
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

def tokenize(
    line: str,
    min_word_length: int,
    tokenizer: str = "whitespace",
    lowercase: bool = False,
) -> list[str]: ...

_WordCounts = tuple[list[str], NDArray[np.int64], int]
# Pair counts: COO rows, cols and counts, then row sums, column sums, total.
_PairCounts = tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    int,
]

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
    row_sums: NDArray[np.int64],
    col_sums: NDArray[np.int64],
    shift: float = 1.0,
    alpha: float = 0.75,
    min_count: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.int32]]: ...
def spill_corpus(
    corpus: Iterable[str] | Iterable[Sequence[str]],
    path: str,
    tokenizer: str = "whitespace",
) -> None: ...
def count_words_bounded(
    paths: list[str],
    min_word_length: int,
    min_count: int,
    sketch_bytes: int,
    table_budget: int,
    tokenizer: str = "whitespace",
    lowercase: bool = False,
    n_threads: int = 0,
    chunk_bytes: int = ...,
) -> tuple[list[str], NDArray[np.int64], int, dict[str, int]]: ...
def count_pairs_bounded(
    paths: list[str],
    vocabulary: list[str],
    window: int,
    min_word_length: int,
    min_pair_count: int,
    sketch_bytes: int,
    table_budget: int,
    tokenizer: str = "whitespace",
    lowercase: bool = False,
    n_threads: int = 0,
    chunk_bytes: int = ...,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    int,
    dict[str, int],
]: ...
