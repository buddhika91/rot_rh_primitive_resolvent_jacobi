#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs
python3 src/rot_rh_resolvent_preregistered_gue_validation_fast_v2.py \
  --dps 60 --radius 3 --samples 96 \
  --calibration-depths 4,6 --evaluation-depths 8 \
  --zero-reference-count 8 \
  --stable-relative-tolerance 1e-2 --stable-absolute-tolerance 1e-2 \
  --trim-fraction 0 --unfold-methods rvm,local \
  --minimum-spacings 2 --minimum-winning-methods 1 \
  --minimum-bootstrap-win 0 --maximum-zero-relative-rmse 10 \
  --bootstrap-trials 10 --cache-prefix outputs/smoke_cache \
  --out-prefix outputs/smoke
