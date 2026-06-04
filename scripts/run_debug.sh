#!/usr/bin/env bash
set -euo pipefail
python run_experiment.py --dataset cifar10_c --data-root /Dataset/yezhong --output-dir ./outputs_debug --batch-size 8 --opt-steps 2 --max-batches 1 "$@"