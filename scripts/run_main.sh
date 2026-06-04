#!/usr/bin/env bash
set -euo pipefail
python run_experiment.py --dataset imagenet_c --data-root /Dataset/yezhong --output-dir ./outputs "$@"