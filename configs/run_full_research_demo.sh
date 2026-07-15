#!/usr/bin/env bash
set -euo pipefail
bash configs/run_operator_demo.sh
bash configs/run_generator_demo.sh
bash configs/run_zero_alignment_demo.sh
bash configs/run_controls_demo.sh
python3 src/demos/feature_audit.py
bash configs/run_all_plots.sh
