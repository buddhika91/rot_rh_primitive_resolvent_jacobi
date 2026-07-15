#!/usr/bin/env bash
set -euo pipefail
python3 src/rot_rh_resolvent_gue_frontier_scan_v2.py \
  --nodes-file outputs/stable_depth52_nodes.csv \
  --depths 40,44,48,52 \
  --relative-tolerances 1e-7,3e-7,1e-6,3e-6,1e-5 \
  --absolute-tolerances 1e-7,3e-7,1e-6,3e-6,1e-5 \
  --trim-fractions 0,0.05,0.10 \
  --unfold-methods rvm,polynomial,local --poly-degree 5 \
  --minimum-spacings-list 8,12,16,20 --bootstrap-trials 500 \
  --out-prefix outputs/gue_frontier
