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

`bench_train.py` runs one `SVDAlgebra.train` per process with the library
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
