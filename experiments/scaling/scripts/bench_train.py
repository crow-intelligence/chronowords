"""Benchmark ``SVDAlgebra.train`` on a byte-bounded prefix of a text corpus.

Run one configuration per process: peak RSS is a process-lifetime high-water
mark, so a second run in the same process would inherit the first one's peak.

Phases are timed without modifying the library:

- ``count``: from the "Counting words and skipgrams..." log line to the
  "Building vocabulary..." line (the per-token Python/CMS loop).
- ``vocab_ppmi``: from "Building vocabulary..." to the first ``svds`` call
  (heavy-hitter scan + the O(V^2) PPMI kernel).
- ``svd``: wall time inside ``scipy.sparse.linalg.svds``.
- ``post_svd``: from ``svds`` returning to the end of ``train`` (includes the
  dense ``M.toarray()`` copy kept for backward compatibility).

Peak RSS (``ru_maxrss``) is sampled at each phase boundary, so the record shows
which phase drove memory. One JSON line is appended to ``--out``.

Usage::

    uv run python experiments/scaling/scripts/bench_train.py
        data/derived/huwiki.txt --max-mb 10 --label huwiki
        --out experiments/scaling/results/baseline.jsonl

(one command, wrapped here)
"""

import argparse
import json
import logging
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import datetime
from datetime import timezone
from pathlib import Path

import scipy.sparse.linalg

import chronowords.algebra.svd as svd_module
from chronowords.algebra.svd import SVDAlgebra


REPO = Path(__file__).resolve().parents[3]
PHASE_MARKERS = {
    "Counting words and skipgrams...": "count_start",
    "Building vocabulary...": "vocab_start",
}


class BenchTimeoutError(Exception):
    """Raised by SIGALRM when a run exceeds ``--timeout``."""


def peak_rss_mb() -> float:
    """Process peak resident set size so far, in MB (Linux reports KiB)."""
    kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return kib * scale / 1e6


class PhaseRecorder(logging.Handler):
    """Record a timestamp and peak RSS when a known log message is emitted."""

    def __init__(self, marks: dict[str, tuple[float, float]]) -> None:
        super().__init__(level=logging.INFO)
        self.marks = marks

    def emit(self, record: logging.LogRecord) -> None:
        """Stamp the phase boundary named by ``record``'s message, if any."""
        name = PHASE_MARKERS.get(record.getMessage())
        if name:
            self.marks[name] = (time.perf_counter(), peak_rss_mb())


def read_prefix(path: Path, max_bytes: int | None) -> Iterator[str]:
    """Yield lines of ``path`` until ``max_bytes`` bytes have been yielded."""
    seen = 0
    with path.open("rb") as fh:
        for raw in fh:
            yield raw.decode("utf-8")
            seen += len(raw)
            if max_bytes is not None and seen >= max_bytes:
                return


def corpus_stats(path: Path, max_bytes: int | None, min_len: int) -> dict:
    """Count bytes, lines and kept tokens of the prefix (untimed pre-pass)."""
    n_bytes = n_lines = n_tokens = 0
    for line in read_prefix(path, max_bytes):
        n_bytes += len(line.encode("utf-8"))
        n_lines += 1
        n_tokens += sum(1 for w in line.split() if len(w) >= min_len)
    return {"bytes": n_bytes, "lines": n_lines, "tokens": n_tokens}


def git_commit() -> str | None:
    """Short commit hash of the working tree, or None outside git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--label", required=True, help="corpus name for the record")
    parser.add_argument("--max-mb", type=float, help="corpus prefix size; default all")
    parser.add_argument("--n-components", type=int, default=100, help="default: 100")
    parser.add_argument("--window-size", type=int, default=5, help="default: 5")
    parser.add_argument("--min-word-length", type=int, default=3, help="default: 3")
    parser.add_argument("--cms-width", type=int, default=1_000_000)
    parser.add_argument("--cms-depth", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=0, help="seconds; 0 = none")
    parser.add_argument("--out", type=Path, required=True, help="JSONL to append to")
    args = parser.parse_args()

    max_bytes = int(args.max_mb * 1e6) if args.max_mb else None
    stats = corpus_stats(args.corpus, max_bytes, args.min_word_length)

    marks: dict[str, tuple[float, float]] = {}
    logger = logging.getLogger(svd_module.__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(PhaseRecorder(marks))

    real_svds = scipy.sparse.linalg.svds
    ppmi_shape: dict[str, object] = {}

    def timed_svds(matrix, *a, **kw):
        marks["svd_start"] = (time.perf_counter(), peak_rss_mb())
        ppmi_shape["shape"] = list(matrix.shape)
        ppmi_shape["nnz"] = int(matrix.nnz)
        try:
            return real_svds(matrix, *a, **kw)
        finally:
            marks["svd_end"] = (time.perf_counter(), peak_rss_mb())

    svd_module.svds = timed_svds

    def on_alarm(signum, frame):
        raise BenchTimeoutError

    if args.timeout:
        signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(args.timeout)

    model = SVDAlgebra(
        n_components=args.n_components,
        window_size=args.window_size,
        min_word_length=args.min_word_length,
        cms_width=args.cms_width,
        cms_depth=args.cms_depth,
    )
    rss_before = peak_rss_mb()
    status = "ok"
    error = None
    t0 = time.perf_counter()
    try:
        model.train(read_prefix(args.corpus, max_bytes))
    except BenchTimeoutError:
        status = "timeout"
    except MemoryError as exc:
        status, error = "oom", repr(exc)
    except Exception as exc:
        status, error = "error", repr(exc)
    t_end = time.perf_counter()
    signal.alarm(0)

    def at(name: str) -> float | None:
        return marks[name][0] if name in marks else None

    def span(a: float | None, b: float | None) -> float | None:
        return round(b - a, 3) if a is not None and b is not None else None

    count_start, vocab_start = at("count_start"), at("vocab_start")
    svd_start, svd_end = at("svd_start"), at("svd_end")
    phases = {
        "count": span(count_start, vocab_start or (t_end if count_start else None)),
        "vocab_ppmi": span(vocab_start, svd_start or (t_end if vocab_start else None)),
        "svd": span(svd_start, svd_end),
        "post_svd": span(svd_end, t_end if svd_end else None),
    }
    total = round(t_end - t0, 3)

    record = {
        "label": args.label,
        "status": status,
        "error": error,
        "corpus": str(args.corpus),
        "max_mb": args.max_mb,
        **stats,
        "params": {
            "n_components": args.n_components,
            "window_size": args.window_size,
            "min_word_length": args.min_word_length,
            "cms_width": args.cms_width,
            "cms_depth": args.cms_depth,
            "timeout": args.timeout,
        },
        "seconds_total": total,
        "seconds_by_phase": phases,
        "tokens_per_second": round(stats["tokens"] / phases["count"])
        if status == "ok" and phases["count"]
        else None,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "rss_before_train_mb": round(rss_before, 1),
        "peak_rss_mb_by_phase_end": {
            k: round(v[1], 1) for k, v in sorted(marks.items(), key=lambda x: x[1][0])
        },
        "vocab_size": len(model.vocabulary),
        "ppmi": ppmi_shape or None,
        "embedding_shape": list(model.embeddings.shape)
        if model.embeddings is not None
        else None,
        "env": {
            "git_commit": git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "ram_gb": round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1
            ),
        },
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
