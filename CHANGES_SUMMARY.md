# Changes Summary — chronowords 0.4: a Rust core for gigabyte-scale corpora

Goal: make "analyse gigabytes of text on a laptop" true. The previous summary (PRs #6–#9,
all merged) is in git history at `a129b09`.

Six PRs, all green, **none merged**. They are stacked, so merge in this order:

| order | PR | branch | what |
|---|---|---|---|
| 1 | #10 | `fix-ppmi-marginals` | **Bug fix:** PPMI was empty / noise in every release since Jan 2025 |
| 2 | #14 | `phase0-benchmarks` | Benchmark harness + pure-Python baseline (Wikipedia hu/en) |
| 3 | #11 | `phase1-rust-core` | Rust core (PyO3 + maturin) replaces Cython; same numbers |
| 4 | #12 | `phase2-exact-default` | Exact two-pass counting by default; new vocabulary rule; `M_dense` removed |
| 5 | #13 | `phase3-scale` | Parallel compact counting, Unicode tokenizer, pair pruning |
| 6 | #15 | `phase4-summary` | Clean before/after benchmarks + this summary |

#14 and #11 are independent of each other; both build on #10. #15 merges #14 into the
#13 chain.

## Headline numbers

Same corpora, same slices, same 4-core / 25 GB laptop. "Before" is 0.3.x *with* the #10
fix, single-threaded; "after" is #13, end to end including SVD (`n_components=100`), on
4 threads. Full tables: `experiments/scaling/README.md`.

| corpus | slice | 0.3 (+#10) total | 0.4 total | speed-up | 0.3 peak RSS | 0.4 peak RSS | 0.3 vocabulary | 0.4 vocabulary |
|---|---|---|---|---|---|---|---|---|
| presidential | all (23.5 MB) | 2.2 min | 0.8 s | **155×** | 294 MB | 150 MB | 1,318 | 2,851 |
| huwiki | 10 MB | 42.5 s | 0.6 s | **68×** | 304 MB | 121 MB | 1,225 | 1,235 |
| huwiki | 100 MB | 6.6 min | 3.8 s | **103×** | 418 MB | 403 MB | 1,122 | 11,976 |
| huwiki | 1000 MB | 60.6 min | 33.1 s | **110×** | 738 MB | 2223 MB | 1,084 | 50,000 |
| huwiki | all (1.21 GB) | not run | 41.5 s | — | — | 2560 MB | — | 50,000 |
| enwiki | 10 MB | 54.3 s | 0.6 s | **92×** | 299 MB | 108 MB | 1,472 | 1,987 |
| enwiki | 100 MB | 8.3 min | 3.5 s | **140×** | 353 MB | 372 MB | 1,339 | 12,008 |
| enwiki | 1000 MB | 80.9 min | 33.0 s | **147×** | 492 MB | 2006 MB | 1,340 | 49,636 |
| enwiki | all (3.00 GB) | not run | 91.0 s | — | — | 3393 MB | — | 50,000 |

That works out to 68–155× faster end to end. The vocabulary is 46× larger at a gigabyte,
and memory is higher accordingly: 2.0–2.6 GB at 1 GB, 3.4 GB at 3 GB, against 0.5–0.7 GB
for the tiny 0.3 model. The 0.3 PPMI matrices were mostly sketch noise; the 0.4 ones are
exact.

## PR by PR

**#10 — PPMI marginal fix.** `PPMIComputer` read word counts from the *skip-gram* sketch,
where single words never occur. Every marginal was 0, so the PPMI matrix was empty, `svds`
failed, and `train` silently returned a dense SVD of zeros plus Gaussian noise. That held
for every bundled corpus. The PR fixes the lookup, logs warnings for an empty PPMI matrix
and for the noisy fallback, and adds a hand-computed golden test (1.75·ln 2) plus a
reference-parity test. All three regression tests fail on `main`.

**#14 — Phase 0 baseline.** Reproducible download, preparation and benchmark scripts, with
provenance manifests. It found three things: counting is >99% of runtime at a flat
~26k tokens/s (61 min per GB of Hungarian, 81 min per GB of English); the vocabulary rule
keeps only ~1,100 words and shrinks as the corpus grows; and 62–79% of PPMI cells are
nonzero, which is sketch-collision noise.

**#11 — Phase 1, Rust core.** maturin with an `abi3-py310` extension, so one wheel per
platform covers Python 3.10–3.13+. `PPMIComputer` is reimplemented in Rust, bit-compatible
with `mmh3`, and reproduces the frozen output of the fixed Cython kernel. Exact counting
primitives are added. CI gets a Rust job and a **build-only** wheels workflow (Linux
x86_64/aarch64, macOS arm64/x86_64, Windows). `experiments/sketch_accuracy`: at least 82% of
sketch PPMI entries on the 23.5 MB presidential corpus are collision artefacts.

**#12 — Phase 2, exact counting by default.**
- `train` accepts paths, lines, token lists, or a one-shot iterable (spilled to a temp
  file).
- New vocabulary rule: top `vocab_size=50_000` words with count ≥ `min_count=100`, ties
  alphabetical.
- `model.counts` and `model.training_record` (counts, parameters, corpus, totals) are
  added; `model.ppmi` replaces `M_dense`; the SVD is seeded, so training is deterministic.
- The runtime dependencies shrink to numpy, scipy and scikit-learn.

**#13 — Phase 3, scale.**
- Files are counted in parallel, in line-aligned chunks, into one sharded pair table
  (8 shards per thread).
- Pair keys are `u32` up to 65,536 words; a count that would overflow raises.
- `min_pair_count` pruning: `SVDAlgebra` prunes at its PPMI threshold, which cannot change
  PPMI.
- `tokenizer="unicode"` (UAX #29) and `lowercase=True` are added; `n_threads` is exposed.

## Needs your decision

1. **Releases.** #10 alone is a good **0.3.1** hotfix. #11–#13 change defaults and numbers
   and remove `M_dense`, so they are **0.4.0**. I bumped no versions and published nothing.
2. **Publishing.** `publish.yml` uploads an **sdist only**. With a Rust extension, anyone
   installing from that sdist needs `cargo`. Before 0.4.0, publishing should upload the
   `wheels.yml` artifacts. I didn't touch the publish workflows because they hold the PyPI
   secret. (The `deploy` job in `ci.yml` is unreachable: it triggers on pushes to `develop`
   but requires `main`.)
3. **Downstream: kenon.** `PMIEmbedder` trains `SVDAlgebra` on small corpora with default
   arguments. Under `min_count=100` that now raises "Empty vocabulary", and kenon's 2-line
   doctest will fail against 0.4. It needs to pass or expose `min_count`. I haven't touched
   kenon.
4. **Defaults that change the numbers** — all exposed and documented, please confirm:
   - `counting="exact"` (was a Count-Min Sketch)
   - `vocab_size=50_000`, `min_count=100` (was words above 0.01% of tokens)
   - `ppmi_min_count=5`, unchanged but now a parameter
   - `tokenizer="whitespace"`, `lowercase=False` (unchanged behaviour)
   - Unchanged: `window_size=5`, `min_word_length=3`, `alpha=0.75`, `shift=1.0`,
     `n_components=300`.
5. **A diachronic corpus for benchmarks.** UK Hansard's archive page says "All
   Parliamentary copyrights are reserved". The Open Parliament Licence page sits behind a
   Cloudflare challenge, so I couldn't verify it and downloaded nothing. `parlamonitor` is
   too small (~32 MB). Please pick or confirm a corpus.
6. **`examples/presidential_speeches.ipynb`.** Its code is updated for the new API
   (`min_count=20` per slice; topics from `model.ppmi`), but it is **not re-executed**, so
   its stored outputs are from 0.3.x, i.e. noise embeddings. Re-running it is blocked by two
   issues that predate this work: cell 1 runs `!pip install chronowords`, which would pull
   0.3.0 from PyPI, and the data path `../data/speeches_lemmatized.json` doesn't match
   `data/corpus/`.
7. **Read the Docs** now needs `build.tools.rust` (set to `"latest"` in #11). Please check
   the first RTD build.
8. **Python 3.13.** The abi3 wheel already covers it. Lifting `requires-python` and adding
   3.13 to the CI matrix is now a two-line change; I didn't make it because it's a support
   commitment.
9. **macOS CI runners.** Twice a macOS job was cancelled after 15 minutes with no steps run
   (no runner assigned). Re-running fixed it both times. If it keeps happening, consider
   trimming the macOS test matrix.

## Deliberately left alone, and why

- **Spill-to-disk / a `max_memory` budget** (planned for Phase 3). Measurements didn't call
  for it: distinct pairs level off, and 3 GB of English peaks well under 3 GB of RAM. A
  GloVe-style design (sorted runs on disk, then a k-way merge with pruning) is sketched in
  #13 if the target ever becomes ≫10 GB corpora or 8 GB machines.
- **A Rust conservative-update Count-Min Sketch / Space-Saving mode.** Sketch mode is still
  a Python counting loop. Accuracy data argues against it, so I didn't speed up a mode that
  should rarely be used.
- **Hugging Face `tokenizers`.** They produce subwords, which is the wrong unit for
  word-level semantic change. The Rust core offers `str.split()`-identical whitespace
  tokenisation, UAX #29 words, and pre-tokenised input instead. Lemmatisation stays upstream
  (spaCy).
- **SVD, NMF and Procrustes stay in scipy/scikit-learn.** They already run in LAPACK/BLAS;
  a Rust rewrite gains nothing.
- **Inherited PPMI semantics are kept exactly:**
  - the asymmetric `alpha` smoothing
  - the `> 5` thresholds (now the `ppmi_min_count` parameter)
  - the rule that PMI must be positive *before* the shift, which differs from
    `max(0, PMI − log shift)` only when `shift < 1`; documented, not changed
- **Carried over from the previous summary, still open:**
  - `CountMinSketch.estimate_error` ignores `confidence`
  - `ProcrustesAligner.get_word_similarity` returns `nan` on zero-norm vectors
  - `TopicModel._compute_topic_similarity` returns 0.0 on any error
  - the dense-SVD fallback adds unseeded noise (it now at least warns and is recorded in
    `training_record["svd"]`)
  - `CountMinSketch.get_heavy_hitters` orders count ties by set iteration order, which
    depends on the hash seed. `SVDAlgebra` no longer relies on it, but the public method
    is unchanged.
- **`main`**: never touched. No force-pushes, no PyPI, no CI secrets. `kenon` was not
  modified.

## How to verify locally

```bash
# Rust toolchain: https://rustup.rs (needed to build from source)
uv sync --all-groups
uv run pytest && uv run pytest --doctest-modules src
uv run ruff check . && uv run ruff format --check . && uv run ty check src tests
(cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test)
(cd docs && uv run make html)

# Reproduce the benchmarks (downloads ~3.3 GB, writes ~4.2 GB of text)
uv run python experiments/scaling/scripts/fetch_wikipedia.py hu
uv run python experiments/scaling/scripts/fetch_wikipedia.py en --shards 0-7
# then see experiments/scaling/README.md
```
