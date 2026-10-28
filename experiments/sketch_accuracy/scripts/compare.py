"""Compare sketch-mode PPMI with exact-count PPMI on the same corpus.

Builds the vocabulary exactly as ``SVDAlgebra.train`` does (heavy hitters above
0.01% of tokens, ``min_word_length=3``), then computes PPMI twice:

- **sketch**: Count-Min Sketches filled exactly as ``SVDAlgebra.train`` fills
  them (``nltk.util.skipgrams`` window 5, width/depth as given), read back by
  ``PPMIComputer``;
- **exact**: ``_core.count_words`` / ``count_pairs`` / ``ppmi_from_counts``.

Every sketch entry is classified against the exact pair count. One JSON record
(counts, parameters, corpus) is written to ``--out``.

Usage::

    uv run python experiments/sketch_accuracy/scripts/compare.py
        data/corpus/speeches_*_lemmatized.txt --label presidential

(one command, wrapped here)
"""

import argparse
import json
from pathlib import Path

import numpy as np
from nltk.util import skipgrams

from chronowords import _core
from chronowords.utils.count_skipgrams import PPMIComputer
from chronowords.utils.probabilistic_counter import CountMinSketch


MIN_LEN, WINDOW, HEAVY = 3, 5, 0.0001


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cms-width", type=int, default=1_000_000)
    parser.add_argument("--cms-depth", type=int, default=5)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "compare.jsonl",
    )
    args = parser.parse_args()

    lines = [ln for f in sorted(args.files) for ln in f.read_text().splitlines()]

    words, wcounts, word_total = _core.count_words(lines, MIN_LEN)
    vocab = [
        w for w, c in zip(words, wcounts, strict=True) if c > int(word_total * HEAVY)
    ]
    rows, cols, pcounts, skip_total = _core.count_pairs(lines, vocab, WINDOW, MIN_LEN)
    true = dict(
        zip(
            zip(rows.tolist(), cols.tolist(), strict=True),
            pcounts.tolist(),
            strict=True,
        )
    )
    d, r, c = _core.ppmi_from_counts(
        rows, cols, pcounts, wcounts[: len(vocab)], float(word_total), float(skip_total)
    )
    exact = dict(zip(zip(r.tolist(), c.tolist(), strict=True), d.tolist(), strict=True))

    wc = CountMinSketch(args.cms_width, args.cms_depth)
    sc = CountMinSketch(args.cms_width, args.cms_depth, track_keys=False)
    for line in lines:
        toks = [w for w in line.split() if len(w) >= MIN_LEN]
        for w in toks:
            wc.update(w)
        for a, b in skipgrams(toks, 2, WINDOW):
            sc.update(f"{a}#{b}")
    m = (
        PPMIComputer(
            sc.counts,
            wc.counts,
            vocab,
            wc.seeds,
            wc.width,
            float(sc.total),
            float(wc.total),
        )
        .compute_ppmi_matrix_with_sketch()
        .tocoo()
    )
    sketch = dict(
        zip(
            zip(m.row.tolist(), m.col.tolist(), strict=True),
            m.data.tolist(),
            strict=True,
        )
    )

    shared = sketch.keys() & exact.keys()
    diffs = (
        np.array([sketch[k] - exact[k] for k in shared]) if shared else np.array([0.0])
    )
    record = {
        "label": args.label,
        "files": [str(f) for f in sorted(args.files)],
        "params": {
            "min_word_length": MIN_LEN,
            "window_size": WINDOW,
            "heavy_hitter_threshold": HEAVY,
            "cms_width": args.cms_width,
            "cms_depth": args.cms_depth,
            "shift": 1.0,
            "alpha": 0.75,
            "min_count": 5,
        },
        "word_total": int(word_total),
        "skip_total": int(skip_total),
        "vocab_size": len(vocab),
        "exact_pairs_observed": len(true),
        "exact_ppmi_nnz": len(exact),
        "sketch_ppmi_nnz": len(sketch),
        "sketch_entries_pair_never_observed": sum(
            1 for k in sketch if true.get(k, 0) == 0
        ),
        "sketch_entries_pair_count_1_to_5": sum(
            1 for k in sketch if 0 < true.get(k, 0) <= 5
        ),
        "exact_entries_missing_from_sketch": len(exact.keys() - sketch.keys()),
        "shared_entries": len(shared),
        "shared_sketch_minus_exact_median": float(np.median(diffs)),
        "shared_sketch_minus_exact_max": float(diffs.max()),
        "mean_overcount_per_sketch_cell": sc.total / args.cms_width,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
