#!/usr/bin/env bash
set -euo pipefail
python3 src/demos/operator_construction_demo.py --depth 24 --dps 180 --radius 4 --samples 2048 --out-prefix outputs/operator_depth24
