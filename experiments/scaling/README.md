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

RESULTS_PLACEHOLDER
