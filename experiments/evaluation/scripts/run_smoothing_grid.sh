#!/usr/bin/env bash
# Smoothing sweep for the published PPMI (alpha, ppmi_min_count, eig).
# Usage: experiments/evaluation/scripts/run_smoothing_grid.sh [label-prefix]
set -euo pipefail
cd "$(dirname "$0")/../../.."
P=${1:-published}
EVAL="uv run python experiments/evaluation/scripts/evaluate.py"
C100=data/derived/enwiki_100mb.txt
C1G=data/derived/enwiki_1000mb.txt
for alpha in 1.0 0.75; do
    for pmc in 0 1 2 5; do
        $EVAL $C100 --label "$P/100mb/alpha=$alpha/ppmi_min_count=$pmc" \
            --param alpha=$alpha --param ppmi_min_count=$pmc
    done
done
for eig in 0.0 1.0; do
    $EVAL $C100 --label "$P/100mb/alpha=0.75/ppmi_min_count=1/eig=$eig" \
        --param ppmi_min_count=1 --param eig=$eig
done
for alpha in 1.0 0.75; do
    for pmc in 0 1 2; do
        $EVAL $C1G --label "$P/1000mb/alpha=$alpha/ppmi_min_count=$pmc" \
            --param alpha=$alpha --param ppmi_min_count=$pmc
    done
done
for eig in 0.0 1.0; do
    $EVAL $C1G --label "$P/1000mb/alpha=0.75/ppmi_min_count=1/eig=$eig" \
        --param ppmi_min_count=1 --param eig=$eig
done
