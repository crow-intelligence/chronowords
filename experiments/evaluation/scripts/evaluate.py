"""Train an SVDAlgebra model and score it on word-similarity benchmarks.

For each benchmark (``fetch_benchmarks.py``), the score is Spearman's rho between
human ratings and model cosine similarity, computed **only over pairs where
both words have a non-zero vector**. The number of such pairs (coverage) is
reported beside rho, and missing pairs are never imputed. Comparing rho between
models with very different coverage is therefore not like for like; read both
numbers.

Pair words are lowercased; the benchmark corpora used here are lowercased too.
WordSim-353 includes identity pairs (e.g. tiger-tiger); they are kept, as is
standard.

Works with the 0.4 API and, for a reference row, with chronowords 0.3 + the
PPMI fix (detected by ``cms_width`` in the constructor; the corpus is then
streamed as lines and every 0.4-only parameter is ignored and recorded as
such).

Usage::

    uv run python experiments/evaluation/scripts/evaluate.py
        data/derived/enwiki_100mb.txt --label base --param ppmi_min_count=1

(one command, wrapped here; ``--param`` may repeat)
"""

import argparse
import inspect
import json
import platform
import resource
import subprocess
import sys
import time
import zipfile
from datetime import UTC
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from chronowords.algebra.svd import SVDAlgebra


REPO = Path(__file__).resolve().parents[3]
RAW = REPO / "data" / "raw" / "eval"
RESULTS = Path(__file__).resolve().parents[1] / "results" / "evaluation.jsonl"


def load_pairs() -> dict[str, list[tuple[str, str, float]]]:
    """Benchmark pairs read straight from the untouched archives."""
    out = {}
    with zipfile.ZipFile(RAW / "wordsim353.zip") as z:
        text = z.read("combined.tab").decode("utf-8")
    rows = [line.split("\t") for line in text.splitlines()[1:] if line.strip()]
    out["wordsim353"] = [(a.lower(), b.lower(), float(s)) for a, b, s in rows]
    with zipfile.ZipFile(RAW / "simlex999.zip") as z:
        text = z.read("SimLex-999/SimLex-999.txt").decode("utf-8")
    rows = [line.split("\t") for line in text.splitlines()[1:] if line.strip()]
    out["simlex999"] = [(r[0].lower(), r[1].lower(), float(r[3])) for r in rows]
    return out


def parse_value(raw: str):
    """Interpret a --param value as JSON when possible, else as a string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def peak_rss_mb() -> float:
    """Process peak resident set size so far, in MB (Linux reports KiB)."""
    kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kib * (1 if sys.platform == "darwin" else 1024) / 1e6


def train(corpus: Path, params: dict) -> tuple[SVDAlgebra, dict, bool]:
    """Train with the installed chronowords; returns (model, params used, legacy)."""
    accepted = inspect.signature(SVDAlgebra.__init__).parameters
    legacy = "cms_width" in accepted
    used = {k: v for k, v in params.items() if k in accepted}
    model = SVDAlgebra(**used)
    if legacy:
        with corpus.open(encoding="utf-8") as fh:
            model.train(line for line in fh)
    else:
        model.train(corpus)
    return model, used, legacy


def score(model: SVDAlgebra, pairs: list[tuple[str, str, float]]) -> dict:
    """Spearman rho over pairs where both words have a non-zero vector."""
    emb = model.embeddings
    assert emb is not None
    index = {w: i for i, w in enumerate(model.vocabulary)}
    norms = np.linalg.norm(emb, axis=1)
    human, cosine = [], []
    missing_word, zero_vector = 0, 0
    for a, b, s in pairs:
        ia, ib = index.get(a), index.get(b)
        if ia is None or ib is None:
            missing_word += 1
            continue
        if norms[ia] < 1e-10 or norms[ib] < 1e-10:
            zero_vector += 1
            continue
        human.append(s)
        cosine.append(float(emb[ia] @ emb[ib] / (norms[ia] * norms[ib])))
    rho = float(spearmanr(human, cosine).statistic) if len(human) > 2 else None
    return {
        "pairs": len(pairs),
        "covered": len(human),
        "coverage": round(len(human) / len(pairs), 4),
        "missing_word": missing_word,
        "zero_vector": zero_vector,
        "spearman": None if rho is None else round(rho, 4),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()

    params = {"n_components": 300}
    for item in args.param:
        key, _, value = item.partition("=")
        params[key] = parse_value(value)

    benchmarks = load_pairs()
    t0 = time.perf_counter()
    model, used, legacy = train(args.corpus, params)
    seconds = time.perf_counter() - t0

    norms = np.linalg.norm(model.embeddings, axis=1)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    ).stdout.strip()
    record = {
        "label": args.label,
        "corpus": str(args.corpus),
        "corpus_bytes": args.corpus.stat().st_size,
        "chronowords_api": "0.3" if legacy else "0.4",
        "params_requested": params,
        "params_used": used,
        "params_ignored": sorted(set(params) - set(used)),
        "vocab_size": len(model.vocabulary),
        "zero_vectors": int((norms < 1e-10).sum()),
        "seconds": round(seconds, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "scores": {name: score(model, pairs) for name, pairs in benchmarks.items()},
        "training_record": getattr(model, "training_record", None),
        "env": {"python": platform.python_version(), "git_commit": commit},
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    s = record["scores"]
    print(
        f"{args.label}: vocab {record['vocab_size']:,}  zero {record['zero_vectors']:,}  "
        f"WS353 rho={s['wordsim353']['spearman']} ({s['wordsim353']['covered']}/353)  "
        f"SimLex rho={s['simlex999']['spearman']} ({s['simlex999']['covered']}/999)  "
        f"{seconds:.0f}s {record['peak_rss_mb']:.0f}MB"
    )


if __name__ == "__main__":
    main()
