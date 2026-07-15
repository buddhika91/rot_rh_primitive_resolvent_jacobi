#!/usr/bin/env bash
set -euo pipefail
python3 src/rot_rh_primitive_resolvent_zero_preserving_gue_audit.py \
  --dps 220 --radius 4 --samples 4096 \
  --depths 20,24,28,32,36,40 --zero-reference-count 40 \
  --stable-relative-tolerance 1e-6 --stable-absolute-tolerance 1e-6 \
  --bulk-trim-fraction 0.10 --unfold-methods rvm,polynomial,local \
  --poly-degree 5 --minimum-spacings 12 \
  --out-prefix outputs/stable_depth40
