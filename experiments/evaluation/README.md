# Evaluation: word-similarity benchmarks

How good are the embeddings, and which settings help? This harness trains
`SVDAlgebra` on English Wikipedia and scores it against human similarity
judgements. It exists so that changes to the method (smoothing, thresholds,
vocabulary) are decided by evidence rather than by argument.

## Benchmarks

| benchmark | pairs | what it rates | licence | in this repository |
|---|---|---|---|---|
| [WordSim-353](https://gabrilovich.com/resources/data/wordsim353/wordsim353.html) (Finkelstein et al. 2002) | 353 | similarity *or* relatedness (*cup–coffee* scores high) | CC BY 4.0 | manifest only |
| [SimLex-999](https://fh295.github.io/simlex.html) (Hill, Reichart & Korhonen 2015) | 999 | similarity only (*cup–coffee* scores low) | not stated; not redistributed | manifest only |

MEN (Bruni et al. 2014) was planned but left out: its official page no longer
resolves and the mirrors state no licence. The WordSim-353
similarity/relatedness split (Agirre et al. 2009) is also left out: its only
host now redirects to an unrelated site.

`scripts/fetch_benchmarks.py` downloads both archives, untouched, into
`data/raw/eval/` (gitignored) and writes the committed manifests (URL,
SHA-256, licence). `scripts/evaluate.py` reads the pairs straight from the
archives.

## Method

- **Score:** Spearman ρ between human ratings and cosine similarity, **over
  the covered pairs only**: those where both words are in the vocabulary and
  have a non-zero vector. Coverage is reported next to every ρ, and missing
  pairs are never imputed. ρ values with very different coverage are not like
  for like.
- **Corpus:** English Wikipedia (2023-11-01 dump), prepared as in
  `experiments/scaling` (lowercased, letters only), as 100 MB and 1 GB prefixes.
- **Model:** `n_components=300`; otherwise library defaults unless a column
  says otherwise. Each run is one process; every record in `results/` carries
  the full parameters and `training_record`.
- **Reference:** the 0.3 code with the #10 PPMI fix (Count-Min Sketch counting,
  the 0.01%-of-tokens vocabulary rule), on 100 MB only: it takes 12 minutes
  there and about 80 at 1 GB.

## Baseline: current code (#18), before any smoothing change

| corpus | code | min_count | vocab_size | ppmi_min_count | vocabulary | zero vectors | WordSim-353 ρ (pairs) | SimLex-999 ρ (pairs) | time | peak memory |
|---|---|---|---|---|---|---|---|---|---|---|
| 100 MB | 0.4 | 100 | 50000 | 5 | 12,008 | 916 | **0.4331** (270/353) | **0.1159** (666/999) | 33 s | 0.4 GB |
| 100 MB | 0.4 | 20 | 50000 | 5 | 33,637 | 13,547 | **0.4656** (286/353) | **0.1143** (753/999) | 46 s | 0.8 GB |
| 100 MB | 0.4 | 100 | 50000 | 1 | 12,008 | 0 | **0.5946** (283/353) | **0.2132** (728/999) | 30 s | 0.5 GB |
| 100 MB | 0.4 | 20 | 50000 | 1 | 33,637 | 123 | **0.5584** (335/353) | **0.1769** (960/999) | 63 s | 0.9 GB |
| 100 MB | 0.4 | 100 | 50000 | 0 | 12,008 | 0 | **0.6252** (283/353) | **0.2536** (728/999) | 49 s | 0.7 GB |
| 100 MB | 0.4 | 20 | 50000 | 0 | 33,637 | 3 | **0.6297** (335/353) | **0.25** (962/999) | 92 s | 1.2 GB |
| 1 GB | 0.4 | 100 | 50000 | 5 | 49,636 | 2,140 | **0.5521** (349/353) | **0.193** (973/999) | 104 s | 2.3 GB |
| 1 GB | 0.4 | 20 | 50000 | 5 | 50,000 | 2,210 | **0.5526** (349/353) | **0.1968** (973/999) | 107 s | 2.2 GB |
| 1 GB | 0.4 | 100 | 50000 | 1 | 49,636 | 0 | **0.6241** (351/353) | **0.2333** (979/999) | 108 s | 2.0 GB |
| 1 GB | 0.4 | 20 | 50000 | 1 | 50,000 | 0 | **0.6253** (351/353) | **0.2333** (979/999) | 122 s | 2.0 GB |
| 1 GB | 0.4 | 100 | 50000 | 0 | 49,636 | 0 | **0.6502** (351/353) | **0.2756** (979/999) | 126 s | 3.1 GB |
| 1 GB | 0.4 | 20 | 50000 | 0 | 50,000 | 0 | **0.6483** (351/353) | **0.2756** (979/999) | 129 s | 3.1 GB |
| 1 GB | 0.4 | 20 | 100000 | 5 | 100,000 | 20,360 | **0.5652** (349/353) | **0.1767** (977/999) | 142 s | 2.9 GB |
| 1 GB | 0.4 | 20 | 100000 | 1 | 100,000 | 55 | **0.6254** (352/353) | **0.2151** (988/999) | 176 s | 3.0 GB |
| 100 MB | 0.3 | 0.01% rule | — | 5 | 1,339 | 0 | **0.3471** (65/353) | **-0.1077** (82/999) | 748 s | 0.4 GB |

### What it shows

1. **Discarding rare pairs costs the most.** `ppmi_min_count=5`, the default,
   drops every pair seen 5 times or fewer. Keeping them raises SimLex ρ from
   0.12 to 0.25 at 100 MB and from 0.19 to 0.28 at 1 GB, raises WordSim ρ from
   0.43 to 0.63 and from 0.55 to 0.65, and removes the zero vectors (2,140 at
   1 GB). The price is memory: 3.1 GB instead of 2.0–2.3 GB at 1 GB.
2. **A bigger vocabulary buys coverage, not quality.** `min_count=20` at 100 MB
   covers 335/353 and 962/999 pairs instead of 283 and 728, with similar ρ. At
   1 GB, a 100k vocabulary under the default threshold leaves 20,360 words
   without a vector.
3. **The 0.3 model covers too little to be scored meaningfully.** With 1,339
   words it covers 65/353 and 82/999 pairs; SimLex ρ is −0.11 on those.

## Reproduce

```bash
uv run python experiments/evaluation/scripts/fetch_benchmarks.py
experiments/evaluation/scripts/run_grid.sh base
```
