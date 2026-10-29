#!/usr/bin/env bash
# max_memory benchmark: same corpora with no budget and with shrinking budgets.
# Results append to experiments/scaling/results/memory.jsonl.
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=experiments/scaling/results/memory.jsonl
B="uv run python experiments/scaling/scripts/bench_exact.py"
run() {
    $B "$@" --out "$OUT" > /dev/null
    uv run python experiments/scaling/scripts/show_memory.py "$OUT"
}
for mm in none 2GB 1GB 500MB; do
    if [[ $mm == none ]]; then run data/derived/enwiki_1000mb.txt --label enwiki-1GB
    else run data/derived/enwiki_1000mb.txt --label enwiki-1GB --max-memory $mm; fi
done
for mm in none 1GB 500MB; do
    if [[ $mm == none ]]; then run data/derived/huwiki_1000mb.txt --label huwiki-1GB
    else run data/derived/huwiki_1000mb.txt --label huwiki-1GB --max-memory $mm; fi
done
run data/derived/enwiki.txt --label enwiki-3GB
run data/derived/enwiki.txt --label enwiki-3GB --max-memory 1GB
for mm in none 500MB 250MB; do
    if [[ $mm == none ]]; then run data/derived/hansard/commons_1990s.txt --label hansard-1990s
    else run data/derived/hansard/commons_1990s.txt --label hansard-1990s --max-memory $mm; fi
done
