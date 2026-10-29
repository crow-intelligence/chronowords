# Scaling baseline (Phase 0)

How far does `SVDAlgebra.train` get on gigabyte-scale corpora on a laptop-class
machine, and where do the time and memory go? These numbers are the yardstick
for the planned Rust core: every later change is measured against them, on the
same corpora, slices and parameters.

The baseline is measured on the code **with the PPMI marginal fix** (PR #10).
Without it the PPMI matrix is empty and the O(V²) kernel exits early, which would
make the PPMI phase look far cheaper than it is.

## Corpora

| label | source | tokenisation |
|-------|--------|--------------|
| `presidential` | bundled `data/corpus/speeches_*_lemmatized.txt`, all periods concatenated | spaCy lemmas, lowercased (as shipped) |
| `huwiki` | Hungarian Wikipedia, 2023-11-01 snapshot, all articles | see below |
| `enwiki` | English Wikipedia, 2023-11-01 snapshot, shards 0–7 of 41 | see below |

Wikipedia comes from the `wikimedia/wikipedia` parquet dataset, which is
already stripped of markup (CC BY-SA 4.0; provenance manifests in
`data/raw/wikipedia/`). `prepare_wikipedia.py` writes one paragraph per line.
It Unicode-lowercases, keeps maximal letter runs (`[^\W\d_]+`, so digits and
punctuation are dropped), does no lemmatisation, and drops paragraphs under 5
tokens. These are benchmark choices, not library defaults. They change vocabulary
and counts, so they are fixed here and must stay fixed across comparisons.

Slices are byte prefixes of the derived text (`--max-mb`), so a 100 MB run reads
exactly the first 100 MB of the file that a 1000 MB run reads.

## What is measured

`bench_train.py` measures the **0.3.x API** (Count-Min Sketch counting), so run it on a checkout of the fix branch (#10). It runs one `SVDAlgebra.train` per process with the library
defaults, except `n_components=100`. The other defaults are `window_size=5`,
`min_word_length=3`, `cms_width=1_000_000` and `cms_depth=5`. It records:

- **wall time per phase:**
  - `count`: the Python counting loop
  - `vocab_ppmi`: the heavy-hitter scan plus the O(V²) PPMI kernel
  - `svd`: inside `scipy.sparse.linalg.svds`
  - `post_svd`: mostly the dense `M_dense` copy
- **peak RSS:** in total and at each phase boundary
- **corpus and output sizes:** tokens, vocabulary size, PPMI nnz
- **reproducibility record:** parameters, git commit, machine

Results append to `results/baseline.jsonl`, one JSON object per run.

Machine: 4 cores, 25 GB RAM, Linux, Python 3.11. Training is single-threaded.

`run_baseline.sh` reproduces the whole series. The recorded runs were launched as
the same `bench_train.py` invocations in shell loops, one corpus at a time.

## Reproduce

```bash
uv sync --all-groups
uv run python experiments/scaling/scripts/fetch_wikipedia.py hu
uv run python experiments/scaling/scripts/fetch_wikipedia.py en --shards 0-7
uv run --group bench python experiments/scaling/scripts/prepare_wikipedia.py \
    data/raw/wikipedia/20231101.hu data/derived/huwiki.txt
uv run --group bench python experiments/scaling/scripts/prepare_wikipedia.py \
    data/raw/wikipedia/20231101.en data/derived/enwiki.txt --max-mb 3000
experiments/scaling/scripts/run_baseline.sh 10 100 1000
```

## Results

All runs finished with status `ok`. Run on code with the PPMI marginal fix (#10); all
runs single-threaded. Some runs shared the machine with Rust compilation (at most 2 jobs on
the other 3 cores), so treat the times as accurate to a few percent.

| corpus | slice | tokens | total | counting | vocab + PPMI | SVD | tokens/s (counting) | peak RSS | vocabulary | PPMI nnz |
|---|---|---|---|---|---|---|---|---|---|---|
| presidential | all (23.5 MB) | 3.2 M | 2.2 min | 2.1 min | 2.2 s | 0.4 s | 25,269 | 294 MB | 1,318 | 100,900 |
| huwiki | 10 MB | 1.0 M | 42.5 s | 39.2 s | 3.0 s | 0.4 s | 25,708 | 304 MB | 1,225 | 32,225 |
| huwiki | 100 MB | 10.2 M | 6.6 min | 6.5 min | 6.4 s | 1.1 s | 26,174 | 418 MB | 1,122 | 938,088 |
| huwiki | 1000 MB | 102.3 M | 60.6 min | 60.2 min | 23.5 s | 2.4 s | 28,317 | 738 MB | 1,084 | 931,510 |
| enwiki | 10 MB | 1.3 M | 54.3 s | 51.4 s | 2.7 s | 0.2 s | 25,743 | 299 MB | 1,472 | 52,181 |
| enwiki | 100 MB | 13.4 M | 8.3 min | 8.2 min | 4.1 s | 1.4 s | 27,356 | 353 MB | 1,339 | 1,078,769 |
| enwiki | 1000 MB | 135.0 M | 80.9 min | 80.7 min | 11.0 s | 1.4 s | 27,890 | 492 MB | 1,340 | 1,118,184 |

What the numbers say:

- **Counting is the whole cost, and it is linear.** The Python per-token loop runs at a flat
  ~25–28k tokens/s regardless of corpus or size: about 1 hour per gigabyte of Hungarian and
  1 h 20 min per gigabyte of English. Building PPMI and SVD together take under 30 s.
- **Memory is low because the vocabulary is tiny, not because the design is efficient.** The
  hidden rule "words above 0.01% of tokens" keeps about 1,100–1,500 words, and the count
  *shrinks* as the corpus grows (huwiki: 1,225 → 1,122 → 1,084). That is far too small for
  semantic-shift work on gigabyte corpora.
- **The PPMI matrix fills with sketch noise.** At 100 MB and above, 62–79% of all V × V cells
  are nonzero (huwiki 1 GB: 931,510 of 1,175,056; enwiki 1 GB: 1,118,184 of 1,795,600). With
  hundreds of millions of skip-grams spread over 1M sketch columns, collision counts alone
  clear the `> 5` threshold for most vocabulary pairs. See `experiments/sketch_accuracy` on
  the Rust branch (#11).

## Rust core (0.4, #11–#13)

Recorded by `scripts/bench_exact.py` into `results/exact.jsonl`: `SVDAlgebra.train(Path)` end
to end, including SVD, with library defaults except `n_components=100` (`counting="exact"`,
`vocab_size=50_000`, `min_count=100`). Counting runs on 4 threads; nothing else ran on the
machine. The corpora, slices and tokenisation match the baseline.

| corpus | slice | tokens | total | counting | PPMI | SVD | tokens/s (counting) | peak RSS | vocabulary | PPMI nnz |
|---|---|---|---|---|---|---|---|---|---|---|
| presidential | all (23.5 MB) | 3.2 M | 0.8 s | 0.5 s | 0.01 s | 0.3 s | 5,961,968 | 150 MB | 2,851 | 26,947 |
| huwiki | 10 MB | 1.0 M | 0.6 s | 0.4 s | 0.00 s | 0.3 s | 2,765,198 | 121 MB | 1,235 | 5,583 |
| huwiki | 100 MB | 10.2 M | 3.8 s | 2.4 s | 0.02 s | 1.5 s | 4,318,716 | 403 MB | 11,976 | 155,887 |
| huwiki | 1000 MB | 102.3 M | 33.1 s | 24.6 s | 0.26 s | 8.2 s | 4,156,517 | 2223 MB | 50,000 | 1,810,247 |
| huwiki | all (1.21 GB) | 123.9 M | 41.5 s | 31.4 s | 0.34 s | 9.7 s | 3,946,781 | 2560 MB | 50,000 | 2,108,142 |
| enwiki | 10 MB | 1.3 M | 0.6 s | 0.3 s | 0.00 s | 0.3 s | 4,725,339 | 108 MB | 1,987 | 11,808 |
| enwiki | 100 MB | 13.4 M | 3.5 s | 1.8 s | 0.04 s | 1.7 s | 7,507,637 | 372 MB | 12,008 | 176,169 |
| enwiki | 1000 MB | 135.0 M | 33.0 s | 25.1 s | 0.32 s | 7.6 s | 5,387,800 | 2006 MB | 49,636 | 1,340,171 |
| enwiki | all (3.00 GB) | 404.6 M | 91.0 s | 80.6 s | 0.77 s | 9.6 s | 5,023,208 | 3393 MB | 50,000 | 2,723,769 |

### Before and after

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

- **Speed:** counting runs at 2.7–7.5M tokens/s versus ~26k (small slices are dominated by fixed costs). A gigabyte of Wikipedia now
  takes about half a minute, a quarter of it SVD.
- **Memory:** higher than 0.3 at a gigabyte and above, because the model is real and
  much larger. The vocabulary goes from ~1,100 to 50,000 words (46×), and the PPMI holds
  observed associations instead of sketch noise. Peak RSS grows sub-linearly: 2.0 GB at 1 GB of English, 3.4 GB
  at 3 GB.
- **SVD** (`scipy.sparse.linalg.svds`, unchanged) is now a visible share of the time:
  8–10 s at 50,000 words.
- `results/exact_preliminary.jsonl` holds the counting-only measurements taken during
  development (shared machine, 3 threads). They are kept as the record behind the Phase 3
  design decisions.

## Bounded memory (`max_memory`, #23)

`scripts/run_memory.sh` and `scripts/run_memory_pmc5.sh` train the same models
with and without a counting budget (4 threads, `n_components=100`, otherwise
defaults). Records: `results/memory.jsonl`. "memory error" is the documented
outcome when the exact candidates do not fit: training stops with a
`MemoryError` naming the sizes, instead of degrading silently.

| corpus | ppmi_min_count | max_memory | outcome | time | peak memory | pair candidates / kept |
|---|---|---|---|---|---|---|
| enwiki-1GB | 1 | none | ok | 54 s | 2.25 GB | — |
| enwiki-1GB | 1 | 2GB | ok | 104 s | 2.27 GB | 44,985,830 / 36,305,545 |
| enwiki-1GB | 1 | 1GB | memory error | 51 s | 1.44 GB | — |
| enwiki-1GB | 1 | 500MB | memory error | 45 s | 0.82 GB | — |
| huwiki-1GB | 1 | none | ok | 59 s | 2.38 GB | — |
| huwiki-1GB | 1 | 1GB | memory error | 42 s | 1.45 GB | — |
| huwiki-1GB | 1 | 500MB | memory error | 35 s | 0.80 GB | — |
| enwiki-3GB | 1 | none | ok | 143 s | 3.56 GB | — |
| enwiki-3GB | 1 | 1GB | memory error | 144 s | 1.58 GB | — |
| hansard-1990s | 1 | none | ok | 29 s | 1.17 GB | — |
| hansard-1990s | 1 | 500MB | memory error | 33 s | 0.84 GB | — |
| hansard-1990s | 1 | 250MB | memory error | 29 s | 0.69 GB | — |
| enwiki-1GB | 5 | none | ok | 37 s | 2.03 GB | — |
| enwiki-1GB | 5 | 1GB | ok | 85 s | 1.28 GB | 12,470,735 / 10,843,501 |
| enwiki-1GB | 5 | 500MB | memory error | 57 s | 0.83 GB | — |
| enwiki-3GB | 5 | none | ok | 112 s | 3.33 GB | — |
| enwiki-3GB | 5 | 1GB | memory error | 158 s | 1.53 GB | — |
| hansard-1990s | 5 | none | ok | 19 s | 1.15 GB | — |
| hansard-1990s | 5 | 250MB | memory error | 27 s | 0.53 GB | — |

What this shows:

- **The guarantee holds.** Every successful bounded run produced the same PPMI
  matrix as the unbounded run (identical nnz; equality is also a Hypothesis
  property in the test suite).
- **At the default `ppmi_min_count=1` a budget cannot save memory.** The filter
  can only rule out pairs seen once, and the pairs seen twice or more (36 M at
  1 GB of English) are what the model keeps. They are the source of the quality
  gain measured in `experiments/evaluation`. A 2 GB budget peaked at 2.27 GB,
  the same as unbounded, and took twice as long.
- **With a higher threshold it helps moderately.** With `ppmi_min_count=5`, a
  1 GB budget cut the 1 GB-English peak from 2.03 to 1.28 GB (−37%) at 2.3× the
  time.
- **Peaks happen during counting**, not in PPMI or SVD, so the budget acts on
  the right phase. Without a budget the default peaks are 1.2 GB (one Hansard
  decade), 2.3–2.4 GB (1 GB of Wikipedia) and 3.6 GB (3 GB).
