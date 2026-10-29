# Sketch vs exact PPMI

Does the Count-Min Sketch path produce the same PPMI matrix as exact counting?
At the library defaults (`cms_width=1_000_000`, `cms_depth=5`), it does not,
even on a 23.5 MB corpus.

`scripts/compare.py` builds one vocabulary (the `SVDAlgebra.train` rule: words
above 0.01% of tokens, `min_word_length=3`), then computes PPMI from sketches
filled exactly as `SVDAlgebra.train` fills them, and from exact counts
(`chronowords._core`). Each sketch entry is classified by its true pair count.
Records append to `results/compare.jsonl`.

## Result: bundled presidential speeches (all periods, 3.2M tokens, 18.3M skip-grams)

| | count |
|---|---|
| vocabulary | 1,318 |
| exact PPMI nonzeros | 12,924 |
| sketch PPMI nonzeros | 100,900 |
| sketch entries whose pair never occurred | 36,662 (36%) |
| sketch entries whose pair occurred 1–5 times (below the `> 5` threshold) | 46,034 (46%) |
| exact entries missing from the sketch | 0 |
| shared entries: sketch − exact PPMI | median +0.112, max +1.153 nats |

At least 82% of the sketch's nonzeros come from collisions alone, and every
true entry is inflated. The cause is load: 18.3M skip-grams in 1M counters per
row is about 18 phantom counts per cell, well above the threshold of 5. The
error grows linearly with corpus size at fixed width, so the gigabyte corpora
the package targets are worse.

## Reproduce

The sketch mode was removed from chronowords after this experiment, on the
strength of these numbers. The comparison script is in git history: check out
commit `289e902` (branch `phase4-summary`) and run

```bash
uv run python experiments/sketch_accuracy/scripts/compare.py data/corpus/speeches_*_lemmatized.txt --label presidential
```
