#!/usr/bin/env bash

set -euo pipefail

export MPLBACKEND=Agg

echo "========================================"
echo "Bandit Platform v1.0 Reproduction"
echo "========================================"

echo
echo "[1/7] Cleaning generated outputs..."
rm -rf results figures

echo
echo "[2/7] Running automated tests..."
python -m pytest -q

echo
echo "[3/7] Running easy-gap benchmark..."
python run.py \
    --config configs/easy_gap.json

echo
echo "[4/7] Running hard-gap benchmark..."
python run.py \
    --config configs/hard_gap.json

echo
echo "[5/7] Running horizon benchmark..."
python run.py \
    --config configs/different_horizon.json

echo
echo "[6/7] Generating benchmark figures..."
python plots/plot_benchmarks.py

echo
echo "[7/7] Generating benchmark summary..."
python scripts/summarize_benchmarks.py

echo
echo "Checking generated outputs..."

test -f \
    results/easy_gap/config_snapshot.json

test -f \
    results/hard_gap/config_snapshot.json

test -f \
    results/different_horizon/config_snapshot.json

test -f figures/easy_gap_regret.png
test -f figures/hard_gap_regret.png
test -f figures/horizon_comparison.png
test -f figures/action_frequency.png
test -f reports/benchmark_summary.csv

easy_gap_count=$(
    find results/easy_gap \
        -maxdepth 1 \
        -name '*.csv' \
        | wc -l
)

hard_gap_count=$(
    find results/hard_gap \
        -maxdepth 1 \
        -name '*.csv' \
        | wc -l
)

horizon_count=$(
    find results/different_horizon \
        -maxdepth 1 \
        -name '*.csv' \
        | wc -l
)

test "$easy_gap_count" -eq 40
test "$hard_gap_count" -eq 40
test "$horizon_count" -eq 120

echo
echo "========================================"
echo "Reproduction completed successfully."
echo "========================================"
echo "Easy-gap runs:       $easy_gap_count"
echo "Hard-gap runs:       $hard_gap_count"
echo "Horizon runs:        $horizon_count"
echo "Figures:             figures/"


