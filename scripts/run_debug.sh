#!/usr/bin/env bash
set -euo pipefail --source-split trai
python run_experiment.py --dataset cifar10_c --data-root /Dataset/yezhong --output-dir ./outputs_debug --source-stats-path ./outputs_debug/source_stats --batch-size 8 --opt-steps 2 --max-batches 1n --target-split test "$@"