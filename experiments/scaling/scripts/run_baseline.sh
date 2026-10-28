#!/usr/bin/env bash
# Phase 0 baseline: time and memory of SVDAlgebra.train at growing corpus sizes.
#
# Each configuration runs in its own process (peak RSS is per process).
# Results are appended to experiments/scaling/results/baseline.jsonl.
#
# Usage: experiments/scaling/scripts/run_baseline.sh [SIZES_MB...]
#   default sizes: 10 100 1000
set -euo pipefail

cd "$(dirname "$0")/../../.."
OUT=experiments/scaling/results/baseline.jsonl
TIMEOUT=${TIMEOUT:-14400}  # 4 h per run
if [[ $# -gt 0 ]]; then SIZES=("$@"); else SIZES=(10 100 1000); fi

bench() {
    uv run python experiments/scaling/scripts/bench_train.py "$@" \
        --timeout "$TIMEOUT" --out "$OUT" > /dev/null
    tail -n 1 "$OUT" | python3 -c '
import json, sys
r = json.load(sys.stdin)
print(f"{r[\"label\"]:>12} {r[\"max_mb\"] or 0:>7} MB  {r[\"status\"]:>7}  "
      f"{r[\"seconds_total\"]:>9.1f} s  peak {r[\"peak_rss_mb\"]:>8.1f} MB  "
      f"vocab {r[\"vocab_size\"]:>6}  phases {r[\"seconds_by_phase\"]}")'
}

# Bundled lemmatised presidential speeches, all periods concatenated (~23.5 MB).
if [[ ! -f data/derived/presidential.txt ]]; then
    mkdir -p data/derived
    cat data/corpus/speeches_*_lemmatized.txt > data/derived/presidential.txt
fi
bench data/derived/presidential.txt --label presidential

for mb in "${SIZES[@]}"; do
    for corpus in huwiki enwiki; do
        bench "data/derived/${corpus}.txt" --label "$corpus" --max-mb "$mb"
    done
done
