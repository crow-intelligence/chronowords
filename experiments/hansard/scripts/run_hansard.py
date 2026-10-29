"""Diachronic run on House of Commons debates: one model per decade, then shifts.

``train`` trains ``SVDAlgebra`` on one decade (raw text, ``tokenizer="unicode"``,
``lowercase=True``, otherwise library defaults) in this process, records time,
peak RSS and the training record to ``results/train.jsonl``, and saves the
model under ``data/derived/hansard/models/<decade>`` (gitignored). Run one
process per decade so that peak RSS is per decade.

``analyse`` loads the saved models and writes ``results/shifts.json``:

- for each consecutive pair of decades, the Procrustes alignment summary and
  the shift (1 - aligned cosine similarity) of each target word present in
  both, skipping words whose embedding is all zero (shift undefined);
- each target word's nearest neighbours in every decade.

Usage::

    uv run python experiments/hansard/scripts/run_hansard.py train 1950s
    uv run python experiments/hansard/scripts/run_hansard.py analyse
"""

import argparse
import json
import math
import os
import platform
import resource
import sys
import time
from datetime import UTC
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from chronowords.algebra.svd import SVDAlgebra
from chronowords.alignment.procrustes import ProcrustesAligner


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data" / "derived" / "hansard"
MODELS = DATA / "models"
RESULTS = Path(__file__).resolve().parents[1] / "results"

# Words with well-known meaning or usage changes in 20th-century English, plus
# stable controls ("house", "water").
TARGETS = [
    "gay",
    "wireless",
    "broadcast",
    "computer",
    "web",
    "mobile",
    "cell",
    "network",
    "virus",
    "green",
    "environment",
    "climate",
    "terror",
    "immigration",
    "europe",
    "empire",
    "commonwealth",
    "house",
    "water",
]
PARAMS = {"tokenizer": "unicode", "lowercase": True}


def peak_rss_mb() -> float:
    """Process peak resident set size so far, in MB (Linux reports KiB)."""
    kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kib * (1 if sys.platform == "darwin" else 1024) / 1e6


def train(decade: str) -> None:
    """Train, time and save one decade's model."""
    path = DATA / f"commons_{decade}.txt"
    model = SVDAlgebra(**PARAMS)
    t0 = time.perf_counter()
    model.train(path)
    seconds = time.perf_counter() - t0
    model.save_model(MODELS / decade)
    rec = model.training_record or {}
    row = {
        "decade": decade,
        "bytes": path.stat().st_size,
        "seconds": round(seconds, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "word_total": rec.get("word_total"),
        "vocab_size": rec.get("vocab_size"),
        "ppmi_nnz": rec.get("ppmi_nnz"),
        "svd": rec.get("svd"),
        "training_record": rec,
        "env": {
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        },
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "train.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(
        f"{decade}: {row['word_total']:,} tokens, vocab {row['vocab_size']:,}, "
        f"{seconds:.1f} s, peak {row['peak_rss_mb']:.0f} MB"
    )


def shift(aligner, word, src, tgt) -> float | None:
    """1 - aligned cosine similarity, or None when undefined (zero vector)."""
    with np.errstate(invalid="ignore"):
        sim = aligner.get_word_similarity(word, src, tgt)
    if sim is None or not math.isfinite(sim):
        return None
    return round(1.0 - sim, 4)


def analyse() -> None:
    """Align consecutive decades and record shifts and neighbours."""
    decades = sorted(p.name for p in MODELS.iterdir() if p.is_dir())
    models = {}
    for d in decades:
        m = SVDAlgebra()
        m.load_model(MODELS / d)
        models[d] = m

    transitions = []
    for a, b in pairwise(decades):
        ma, mb = models[a], models[b]
        aligner = ProcrustesAligner()
        metrics = aligner.fit(
            ma.embeddings, mb.embeddings, ma.vocabulary, mb.vocabulary
        )
        transitions.append(
            {
                "from": a,
                "to": b,
                "aligned_words": metrics.num_aligned_words,
                "average_cosine_similarity": round(
                    float(metrics.average_cosine_similarity), 4
                ),
                "shifts": {
                    w: s
                    for w in TARGETS
                    if (s := shift(aligner, w, ma.embeddings, mb.embeddings))
                    is not None
                },
            }
        )

    neighbours = {
        w: {
            d: [h.word for h in models[d].most_similar(w, n=8)]
            for d in decades
            if w in models[d].vocabulary
        }
        for w in TARGETS
    }
    out = {
        "decades": decades,
        "params": PARAMS,
        "targets": TARGETS,
        "transitions": transitions,
        "neighbours": neighbours,
    }
    (RESULTS / "shifts.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {RESULTS / 'shifts.json'}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("decade", help="e.g. 1950s")
    sub.add_parser("analyse")
    args = parser.parse_args()
    if args.cmd == "train":
        train(args.decade)
    else:
        analyse()


if __name__ == "__main__":
    main()
