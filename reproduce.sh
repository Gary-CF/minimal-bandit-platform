#!/usr/bin/env bash

set -e

echo "Cleaning old outputs..."
rm -rf results figures

echo "Running bandit experiments..."
python run.py --config configs/basic.json

echo "Plotting mean regret curves..."
python plots/plot_regret.py

echo "Reproduction completed."
echo "CSV results: results/"
echo "Regret figure: figures/regret_curve.png"

