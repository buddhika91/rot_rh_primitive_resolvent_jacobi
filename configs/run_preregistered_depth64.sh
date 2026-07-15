#!/usr/bin/env bash
set -euo pipefail
python3 src/rot_rh_resolvent_preregistered_gue_validation_fast_v2.py \
  --dps 400 --radius 4 --samples 8192 \
  --calibration-depths 48,52 --evaluation-depths 56,60,64 \
  --zero-reference-count 64 \
  --stable-relative-tolerance 1e-6 --stable-absolute-tolerance 1e-6 \
  --trim-fraction 0 --unfold-methods rvm,polynomial,local --poly-degree 5 \
  --minimum-spacings 28 --minimum-winning-methods 2 \
  --ratio-min 0.55 --ratio-max 0.65 --minimum-bootstrap-win 0.80 \
  --maximum-zero-relative-rmse 1e-8 --bootstrap-trials 500 \
  --cache-prefix outputs/depth64_cache \
  --out-prefix outputs/preregistered_depth64
