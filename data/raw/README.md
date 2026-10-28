# data/raw

Untouched downloads, kept as the provenance record. The payloads are large and
re-fetchable, so they are **gitignored**; each one has a committed
`*.manifest.json` beside it with URL, size, SHA-256, licence and fetch time.

| path | source | licence | fetched by |
|------|--------|---------|------------|
| `wikipedia/20231101.hu/` | [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), Hungarian, all 4 shards | CC BY-SA 4.0 / GFDL | `experiments/scaling/scripts/fetch_wikipedia.py hu` |
| `wikipedia/20231101.en/` | same, English, shards 0–7 of 41 | CC BY-SA 4.0 / GFDL | `experiments/scaling/scripts/fetch_wikipedia.py en --shards 0-7` |

Attribution: Wikipedia text © Wikipedia contributors, CC BY-SA 4.0. Derived
corpora built from it (`data/derived/`, gitignored) inherit that licence.

Derived text is rebuilt with
`experiments/scaling/scripts/prepare_wikipedia.py`; see
`experiments/scaling/README.md` for the tokenisation applied.
