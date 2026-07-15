#!/usr/bin/env bash
set -euo pipefail
python3 src/demos/generator_construction_demo.py --depth 20 --time 1 --dps 140 --radius 4 --samples 1024 --out-prefix outputs/generator_depth20
