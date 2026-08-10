#!/usr/bin/env bash

set -euo pipefail

export MPLBACKEND=Agg

echo "========================================"
echo "Bandit Platform v1.0 Reproduction"
echo "========================================"

echo
echo "[1/6] Cleaning generated outputs..."
rm -rf results figures

echo
echo "[2/6] Running automated tests..."
python -m pytest -q

echo
echo "[3/6] Running easy-gap benchmark..."
python run.py \
    --config configs/easy_gap.json

echo
echo "[4/6] Running hard-gap benchmark..."
python run.py \
    --config configs/hard_gap.json

echo
echo "[5/6] Running horizon benchmark..."
python run.py \
    --config configs/different_horizon.json

echo
echo "[6/6] Generating benchmark figures..."
python plots/plot_benchmarks.py

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

