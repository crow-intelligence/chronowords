# Changes Summary — chronowords 0.4: a Rust core for gigabyte-scale corpora

Goal: make "analyse gigabytes of text on a laptop" true. The previous summary (PRs #6–#9,
all merged) is in git history at `a129b09`.

Fifteen PRs, **none merged**, in three batches. They are stacked, so merge in this order:

| order | PR | branch | what |
|---|---|---|---|
| 1 | #10 | `fix-ppmi-marginals` | **Bug fix:** PPMI was empty / noise in every release since Jan 2025 |
| 2 | #14 | `phase0-benchmarks` | Benchmark harness + pure-Python baseline (Wikipedia hu/en) |
| 3 | #11 | `phase1-rust-core` | Rust core (PyO3 + maturin) replaces Cython; same numbers |
| 4 | #12 | `phase2-exact-default` | Exact two-pass counting by default; new vocabulary rule; `M_dense` removed |
| 5 | #13 | `phase3-scale` | Parallel compact counting, Unicode tokenizer, pair pruning |
| 6 | #15 | `phase4-summary` | Clean before/after benchmarks + this summary |
| 7 | #16 | `python-313-315` | Python ≥ 3.13 (tested 3.13/3.14/3.15) and CI update, incl. wheel publishing |
| 8 | #17 | `remove-sketch-mode` | Count-Min Sketch mode removed; exact counting only |
| 9 | #18 | `icu-tokenizer` | `tokenizer="unicode"` uses ICU4X: Chinese, Japanese, Thai, … |
| 10 | #19 | `notebook-fix` | Example notebook fixed and re-executed with 0.4 |
| 11 | #20 | `hansard-diachronic` | Hansard 1919–2026 diachronic experiment |
| 12 | #21 | `eval-harness` | Evaluation harness (WordSim-353, SimLex-999) + baseline |
| 13 | #22 | `smoothing-published` | PPMI as published (Levy et al. 2015), defaults from evidence, words without vectors |
| 14 | #23 | `sketch-filter` | Optional `max_memory`: Count-Min Sketch as an exact-preserving filter |
| 15 | #24 | `rerun-experiments` | Notebook and Hansard re-run on #22's defaults + this summary |

#14 and #11 are independent of each other; both build on #10. #15 merges #14 into the
#13 chain. #16–#18 stack on #15. #19 and #20 each stack on #18 and are independent of
each other.

## Third batch (2026-10-29): smoothing chosen by evidence, bounded memory

You were uneasy about losing the Count-Min Sketch (your bounter-inspired
design), about laptop resources, and about how rare words are handled. We
agreed to measure quality first, then fix the smoothing, and to bring the
sketch back as a filter that cannot change the numbers.

- **#21 — evaluation harness.** Spearman ρ on WordSim-353 (CC BY 4.0) and
  SimLex-999, always with coverage, on 100 MB and 1 GB of English Wikipedia.
  MEN and the WordSim similarity/relatedness split were left out: their hosts
  are gone. **Finding:** dropping pairs seen 5 times or fewer was the largest
  quality loss in the defaults, and the source of the zero vectors.
- **#22 — PPMI as published, defaults from evidence.**
  - Our "context-distribution smoothing" did not match Levy, Goldberg & Dagan
    (2015): it was applied to the focus word and not normalised. It now
    follows the paper, with marginals from the full co-occurrence matrix
    (accumulated in Rust before pruning).
  - The **`ppmi_min_count` default goes from 5 to 1.** `alpha` stays 0.75
    and `eig` stays 0.5.
  - Result: **1 GB English goes from WordSim 0.55 / SimLex 0.19 to 0.68 /
    0.29**, and the zero vectors go from 2,140 to 0.
  - Words without a vector are now exactly zero, `has_vector()` is added, and
    `ProcrustesAligner.get_word_similarity` returns **`None` instead of
    `nan`**.
- **#23 — `max_memory`.**
  - A conservative-update Count-Min Sketch decides which words and pairs are
    counted exactly. Kept counts are **identical** to the unbounded path (a
    Hypothesis property); an overrun raises `MemoryError`.
  - **Honest result:** at the new default it cannot lower the peak, because
    the pairs seen twice or more are the model. With `ppmi_min_count=5` a 1 GB
    budget cut 1 GB English from 2.03 to 1.28 GB, at 2.3× the time.
- **#24 — re-runs on #22's defaults.**
  - Notebook: clean, with 251 alignment anchors in the earliest period
    instead of 94.
  - Hansard: the controls are quieter (shift 0.08), and the neighbours are
    cleaner. *gay* is clean from the 1980s; *climate* shows three senses
    (weather → "atmosphere, mood" → *Kyoto, warming*); *web* reaches
    *internet* in the 1990s; *terror* goes *horror, dread* → *IRA* →
    *al-Qaeda* → *Hamas*.
  - All 12 decades train in 7.2 minutes, at most 1.13 GB per decade.

## Second batch (2026-10-29), after your answers

You confirmed the defaults, asked for Python 3.13–3.15, a CI update, the Hansard
corpora, a working notebook, removal of the old counter mode, and ICU4X.

- **#16 — Python and CI.**
  - `requires-python = ">=3.13"`, with abi3-py313 wheels. The full suite passes locally
    on CPython 3.13.13, 3.14.10 and 3.15.2. The lock is upgraded to numpy 2.5 / scipy 1.18
    / scikit-learn 1.9, which all ship 3.15 wheels.
  - The notebook/data `examples` dependency group is skipped on 3.15, because spaCy and
    altair's `rpds-py` have no 3.15 builds yet.
  - CI tests 3.13–3.15 on Linux, 3.13 and 3.15 on macOS, and 3.15 on Windows. It pins the
    matrix interpreter with `UV_PYTHON`; before, uv could silently use another Python.
  - The duplicate `ci.yml` with its dead deploy job is removed.
  - **`publish.yml` now builds and uploads all wheels plus the sdist** on a release,
    using the existing `PYPI_TOKEN`.
- **#17 — sketch mode removed:** `counting=`, `cms_*`, `PPMIComputer`, `CountMinSketch`,
  the Rust sketch kernel and the murmur hash; +79 / −1,905 lines. The tests keep an
  nltk-anchored reference for skip-grams without depending on nltk.
- **#18 — ICU4X.**
  - Chinese: 我们 · 今天 · 北京 · 大学 instead of single characters. Thai: วัน · นี้ ·
    อากาศ instead of fragments.
  - On alphabetic text, tokens differ from the old UAX #29 tokenizer by 0.0002–0.033%,
    at the same speed. The wheel is 2.4 MB.
- **#19 — notebook.** Runs end to end on 0.4 with no errors and no warnings. Periods too
  small for 100 dimensions are skipped (2025–2049 has one speech).
- **#20 — Hansard.**
  - Data: every Commons debate from 1919 to Oct 2026 (TheyWorkForYou / ParlParse, Open
    Parliament Licence) — 15,543 sitting days, 5.5 GB of text, 738 M tokens, no parse
    failures.
  - Training: one model per decade, **4.5 minutes in total, ≤ 1.03 GB of memory per
    decade**.
  - Results: control words shift 0.10–0.17 on average; words with known changes shift
    0.4–0.8, and their neighbours show it. *cell*: prison → stem cells → gene therapy.
    *virus*: HIV → pandemic flu → covid. *climate*: soil → Kyoto → warming. *gay*,
    *mobile*, *web* change as expected.
  - Caveats are written up in `experiments/hansard/README.md`.

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
platform covers Python 3.10–3.13+ (floor raised to 3.13 in #16). `PPMIComputer` is reimplemented in Rust, bit-compatible
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

1. **Releases.** #10 alone is a good **0.3.1** hotfix; the rest is **0.4.0** (breaking:
   defaults and numbers, removed `M_dense`, sketch mode, `CountMinSketch` and
   `PPMIComputer`, Python ≥ 3.13, `nan` → `None` from Procrustes). I bumped no
   versions and published nothing. `publish.yml` is ready (#16).
2. **Is #23 (`max_memory`) worth merging?** It adds about 600 lines for an exact,
   fail-loudly memory cap that saves little at the default threshold. Without it,
   default peaks are 1.2 GB per Hansard decade, about 2.3 GB per GB of Wikipedia,
   and 3.6 GB at 3 GB. #24 does not depend on it.
3. **`alpha` = 0.75 vs 1.0.** No smoothing scored 0.012–0.015 ρ higher at 1 GB,
   which is within benchmark noise. I kept the published 0.75 because the
   benchmarks only cover frequent words and smoothing protects rare ones. Your
   call.
4. **Downstream: kenon.** `PMIEmbedder` uses the default `min_count=100`, which
   raises "Empty vocabulary" on its small corpora; kenon also needs Python ≥ 3.13.
   I haven't touched kenon.
5. **Read the Docs** needs `build.tools.rust` (#11) and Python 3.13 (#16). Please
   check the first RTD build.
6. **macOS CI runners** were cancelled twice with no runner assigned; re-runs
   fixed it both times.
7. **Tokenizer extras not chosen:** Snowball stemming and Unicode normalisation.
   Both are small, opt-in additions if wanted later.

Resolved since the second batch: Procrustes `nan` (#22, now `None`); smoothing
matching the literature (#22); a memory cap exists (#23).

## Deliberately left alone, and why

- **Spill-to-disk / a `max_memory` budget** (planned for Phase 3). Measurements didn't call
  for it: distinct pairs level off, and 3 GB of English peaks well under 3 GB of RAM. A
  GloVe-style design (sorted runs on disk, then a k-way merge with pruning) is sketched in
  #13 if the target ever becomes ≫10 GB corpora or 8 GB machines.
- **Count-Min Sketch:** removed entirely (#17), on your decision.
- **Hugging Face `tokenizers`.** They produce subwords, which is the wrong unit for
  word-level semantic change. The Rust core offers `str.split()`-identical whitespace
  tokenisation, ICU4X word segmentation (#18), and pre-tokenised input instead.
  charabia, lindera and jieba-rs were considered and not added: they bring large
  dictionaries into the core. Lemmatisation stays upstream
  (spaCy).
- **SVD, NMF and Procrustes stay in scipy/scikit-learn.** They already run in LAPACK/BLAS;
  a Rust rewrite gains nothing.
- **Inherited PPMI semantics are kept exactly:**
  - the asymmetric `alpha` smoothing
  - the `> 5` thresholds (now the `ppmi_min_count` parameter)
  - the rule that PMI must be positive *before* the shift, which differs from
    `max(0, PMI − log shift)` only when `shift < 1`; documented, not changed
- **Carried over from the previous summary, still open:**
  - `TopicModel._compute_topic_similarity` returns 0.0 on any error
  - the dense-SVD fallback adds unseeded noise (it now at least warns and is recorded in
    `training_record["svd"]`)
- **Hansard before 1919.** The parliament.uk Hansard archive (1803–2005) did not respond
  to requests, so the experiment starts in 1919 (TheyWorkForYou's coverage).
- **`main`**: never touched. No force-pushes, no PyPI. The CI secret is referenced
  unchanged. `kenon` was not modified.
- **Local tooling:** `uv` was upgraded from 0.11.7 to 0.13.6 to get Python 3.15.2
  (revert with `uv self update 0.11.7`). Python 3.13, 3.14 and 3.15 were installed with
  uv.

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
