#!/usr/bin/env bash
# Baseline evaluation grid on the English Wikipedia slices (see README).
# Usage: experiments/evaluation/scripts/run_grid.sh [label-prefix]
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=${1:-base}
EVAL="uv run python experiments/evaluation/scripts/evaluate.py"
for corpus in data/derived/enwiki_100mb.txt data/derived/enwiki_1000mb.txt; do
    size=$(basename "$corpus" .txt | sed 's/enwiki_//')
    for pmc in 5 1 0; do
        $EVAL "$corpus" --label "$P/$size/min_count=100/ppmi_min_count=$pmc" \
            --param ppmi_min_count=$pmc
        $EVAL "$corpus" --label "$P/$size/min_count=20/ppmi_min_count=$pmc" \
            --param min_count=20 --param ppmi_min_count=$pmc
    done
    if [[ $size == 1000mb ]]; then
        for pmc in 5 1; do
            $EVAL "$corpus" --label "$P/$size/min_count=20/vocab=100k/ppmi_min_count=$pmc" \
                --param min_count=20 --param vocab_size=100000 --param ppmi_min_count=$pmc
        done
    fi
done
