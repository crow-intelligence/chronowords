#!/usr/bin/env bash
# max_memory with a higher pair threshold (ppmi_min_count=5), where the
# sketch filter can rule out most pairs. Appends to results/memory.jsonl.
set -euo pipefail
cd "$(dirname "$0")/../../.."
OUT=experiments/scaling/results/memory.jsonl
B="uv run python experiments/scaling/scripts/bench_exact.py --ppmi-min-count 5"
run() {
    $B "$@" --out "$OUT" > /dev/null
    uv run python experiments/scaling/scripts/show_memory.py "$OUT"
}
for mm in none 1GB 500MB; do
    if [[ $mm == none ]]; then run data/derived/enwiki_1000mb.txt --label enwiki-1GB-pmc5
    else run data/derived/enwiki_1000mb.txt --label enwiki-1GB-pmc5 --max-memory $mm; fi
done
run data/derived/enwiki.txt --label enwiki-3GB-pmc5
run data/derived/enwiki.txt --label enwiki-3GB-pmc5 --max-memory 1GB
for mm in none 250MB; do
    if [[ $mm == none ]]; then run data/derived/hansard/commons_1990s.txt --label hansard-1990s-pmc5
    else run data/derived/hansard/commons_1990s.txt --label hansard-1990s-pmc5 --max-memory $mm; fi
done
