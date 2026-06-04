#!/usr/bin/env bash
set -euo pipefail
python data/precompute_stats.py --dataset imagenet_c --data-root /Dataset/yezhong --model-name resnet50 --output ./outputs/source_stats "$@"