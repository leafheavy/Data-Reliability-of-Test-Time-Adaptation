#!/usr/bin/env bash
set -euo pipefail
python run_experiment.py \
  --dataset cifar10_c \
  --data-root /Dataset/yezhong \
  --output-dir ./outputs_debug \
  --source-stats-path ./outputs_debug/source_stats \
  --source-split train \
  --target-split test \
  --eval-split test \
  --batch-size 8 \
  --train-epochs 5 \
  --opt-steps 2 \
  --max-batches 1 \
  "$@"