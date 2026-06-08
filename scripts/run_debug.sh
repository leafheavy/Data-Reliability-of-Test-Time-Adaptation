#!/usr/bin/env bash
set -euo pipefail
python run_experiment.py \
  --phase structure \
  --dataset cifar10_c \
  --data-root /Dataset/yezhong \
  --corruption-source synthetic \
  --corruptions gaussian_noise,brightness \
  --severities 1 \
  --output-dir ./outputs_debug \
  --source-stats-path ./outputs_debug/source_stats \
  --source-split train \
  --target-split test \
  --eval-split test \
  --batch-size 8 \
  --train-epochs 5 \
  --max-batches 1 \
  --epsilon-bootstrap 2 \
  --max-descriptor-items 512 \
  "$@"
