"""Download word-similarity benchmarks, untouched, with provenance manifests.

Each archive is saved as downloaded under ``data/raw/eval/`` (gitignored) and a
``<name>.manifest.json`` (committed) records URL, size, SHA-256, licence and
fetch time. ``evaluate.py`` reads the pairs straight from the archives.

- **WordSim-353** (Finkelstein et al. 2002): 353 pairs rated for similarity or
  relatedness; CC BY 4.0.
- **SimLex-999** (Hill, Reichart & Korhonen 2015): 999 pairs rated for
  similarity only. No licence is stated on the distribution page, so the data
  is not committed anywhere in this repository.

MEN (Bruni et al. 2014) is not included: its official page no longer resolves
and the mirrors carry no licence.

Usage::

    uv run python experiments/evaluation/scripts/fetch_benchmarks.py
"""

import hashlib
import json
import urllib.request
from datetime import UTC
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw" / "eval"

BENCHMARKS = {
    "wordsim353": {
        "url": "https://gabrilovich.com/resources/data/wordsim353/wordsim353.zip",
        "page": "https://gabrilovich.com/resources/data/wordsim353/wordsim353.html",
        "licence": "CC BY 4.0",
        "citation": "Finkelstein et al. (2002). Placing search in context: the concept "
        "revisited. ACM TOIS 20(1):116-131.",
    },
    "simlex999": {
        "url": "https://fh295.github.io/SimLex-999.zip",
        "page": "https://fh295.github.io/simlex.html",
        "licence": "not stated on the distribution page; not redistributed",
        "citation": "Hill, Reichart & Korhonen (2015). SimLex-999: Evaluating semantic "
        "models with (genuine) similarity estimation. Computational Linguistics "
        "41(4):665-695.",
    },
}


def main() -> None:
    """Fetch every benchmark archive and write its manifest."""
    RAW.mkdir(parents=True, exist_ok=True)
    for name, meta in BENCHMARKS.items():
        target = RAW / f"{name}.zip"
        with urllib.request.urlopen(meta["url"]) as resp:
            payload = resp.read()
        target.write_bytes(payload)
        manifest = {
            **meta,
            "file": target.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        target.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(f"{name}: {len(payload):,} bytes, sha256 {manifest['sha256'][:12]}...")


if __name__ == "__main__":
    main()
