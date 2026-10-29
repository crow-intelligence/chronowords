"""Benchmark ``SVDAlgebra.train`` (exact counting, file input) end to end.

Companion to ``bench_train.py`` (the pure-Python baseline): same corpora, same
byte prefixes, same phase markers, but the corpus is passed as a
``pathlib.Path`` so the compiled core reads it directly on ``--threads``
threads. A prefix is materialised once as ``<stem>_<N>mb.txt`` next to the
corpus, using the same rule as ``bench_train.read_prefix`` (whole lines until
at least N MB). One JSON line, including ``SVDAlgebra.training_record``, is
appended to ``--out``.

Usage::

    uv run python experiments/scaling/scripts/bench_exact.py
        data/derived/huwiki.txt --max-mb 1000 --label huwiki --threads 4
        --out experiments/scaling/results/exact.jsonl

(one command, wrapped here)
"""

import argparse
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import chronowords.algebra.svd as svd_module
from chronowords.algebra.svd import SVDAlgebra


REPO = Path(__file__).resolve().parents[3]


def peak_rss_mb() -> float:
    """Process peak resident set size so far, in MB (Linux reports KiB)."""
    kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kib * (1 if sys.platform == "darwin" else 1024) / 1e6


def prefix_file(corpus: Path, max_mb: float | None) -> Path:
    """Materialise (once) the first ``max_mb`` MB of whole lines of ``corpus``."""
    if max_mb is None:
        return corpus
    target = corpus.with_name(f"{corpus.stem}_{max_mb:g}mb.txt")
    if target.exists():
        return target
    limit, seen = int(max_mb * 1e6), 0
    tmp = target.with_suffix(".part")
    with corpus.open("rb") as src, tmp.open("wb") as dst:
        for raw in src:
            dst.write(raw)
            seen += len(raw)
            if seen >= limit:
                break
    tmp.rename(target)
    return target


class Marks(logging.Handler):
    """Timestamp the phase log lines emitted by ``SVDAlgebra.train``."""

    names: ClassVar[dict[str, str]] = {
        "Counting words and skipgrams...": "count_start",
        "Building vocabulary...": "count_end",
    }

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.at: dict[str, tuple[float, float]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        """Record time and peak RSS at a known phase boundary."""
        name = self.names.get(record.getMessage())
        if name:
            self.at[name] = (time.perf_counter(), peak_rss_mb())


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-mb", type=float)
    parser.add_argument("--threads", type=int, default=None, help="default: all cores")
    parser.add_argument("--n-components", type=int, default=100, help="default: 100")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    path = prefix_file(args.corpus, args.max_mb)
    marks = Marks()
    logger = logging.getLogger(svd_module.__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(marks)
    real_svds = svd_module.svds

    def timed_svds(matrix, *a, **kw):
        marks.at["svd_start"] = (time.perf_counter(), peak_rss_mb())
        try:
            return real_svds(matrix, *a, **kw)
        finally:
            marks.at["svd_end"] = (time.perf_counter(), peak_rss_mb())

    svd_module.svds = timed_svds

    model = SVDAlgebra(n_components=args.n_components, n_threads=args.threads)
    t0 = time.perf_counter()
    model.train(path)
    t1 = time.perf_counter()

    def span(a: str, b: str) -> float:
        return round(marks.at[b][0] - marks.at[a][0], 3)

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
    ).stdout.strip()
    rec = model.training_record or {}
    record = {
        "label": args.label,
        "corpus": str(path),
        "max_mb": args.max_mb,
        "bytes": path.stat().st_size,
        "threads": args.threads or os.cpu_count(),
        "seconds_total": round(t1 - t0, 3),
        "seconds_by_phase": {
            "count": span("count_start", "count_end"),
            "ppmi": span("count_end", "svd_start"),
            "svd": span("svd_start", "svd_end"),
        },
        "tokens_per_second": round(
            rec["word_total"] / span("count_start", "count_end")
        ),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "peak_rss_mb_by_phase_end": {k: round(v[1], 1) for k, v in marks.at.items()},
        "training_record": rec,
        "env": {
            "git_commit": commit,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
