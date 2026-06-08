#!/usr/bin/env bash
set -euo pipefail

# Run all non-LSTM baseline experiments (nbeats + tcn across all datasets)

echo "=== ETTh1 ==="
python train_baseline_script.py experiments=training/etth1_nbeats_baseline
python train_baseline_script.py experiments=training/etth1_tcn_baseline

echo "=== ETTm2 ==="
python train_baseline_script.py experiments=training/ettm2_nbeats_baseline
python train_baseline_script.py experiments=training/ettm2_tcn_baseline

echo "=== Exchange Rate ==="
python train_baseline_script.py experiments=training/exchange_rate_nbeats_baseline
python train_baseline_script.py experiments=training/exchange_rate_tcn_baseline

echo "=== Weather ==="
python train_baseline_script.py experiments=training/weather_nbeats_baseline
python train_baseline_script.py experiments=training/weather_tcn_baseline

echo "=== All baseline runs complete ==="
