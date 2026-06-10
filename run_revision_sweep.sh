#!/usr/bin/env bash
# run_revision_sweep.sh
# =====================
# Focused sweep for the major revision. Fills results.csv with the numbers needed
# for the augmentation table, confidence intervals and combined-noise comparison.
#
# Addresses:
#   Reviewer 1 #5  -> 20 random seeds (mean / std / 95% CI)
#   Reviewer 2 (F) -> exact p-values + CIs come from statistical_analysis.py afterwards
#   Reviewer 2 (R) -> combined noise distributions
#
# Design choice: the per-distribution SNR *line plots* (Fig 8/9) can stay at the
# original 3 seeds; tight CIs are only needed at the focal SNR=30 augmentation
# comparison, so we run 20 seeds THERE (single + combined distributions) + baseline.
#
# Runtime: ~20 seeds x 9 conditions x ~10-30 s  ~=  30-90 min on an 8-core laptop.
#
# Usage:
#   cd 02_experiments
#   chmod +x run_revision_sweep.sh
#   ./run_revision_sweep.sh

set -e
cd "$(dirname "$0")"

SEEDS=(42 7 123 2024 88 314 1 999 55 7777 13 21 34 101 202 303 404 505 606 808)
SINGLE=(gaussian student_t pink hetero)
COMBINED=(gaussian_pink gaussian_student_t gaussian_hetero pink_student_t gaussian_pink_student_t)
SNR=30

echo "=== Clean baseline (20 seeds) ==="
for s in "${SEEDS[@]}"; do
    echo "[baseline seed=$s]"
    python3 run_experiment.py baseline 0 "$s"
done

echo "=== Single-distribution noise at SNR=$SNR (20 seeds) ==="
for noise in "${SINGLE[@]}"; do
    for s in "${SEEDS[@]}"; do
        echo "[$noise SNR=$SNR seed=$s]"
        python3 run_experiment.py "$noise" "$SNR" "$s"
    done
done

echo "=== Combined-distribution noise at SNR=$SNR (20 seeds) ==="
for noise in "${COMBINED[@]}"; do
    for s in "${SEEDS[@]}"; do
        echo "[$noise SNR=$SNR seed=$s]"
        python3 run_experiment.py "$noise" "$SNR" "$s"
    done
done

echo
echo "=== Exact-p / CI / effect-size statistics ==="
python3 ../03_analysis/statistical_analysis.py

echo
echo "Done. Outputs:"
echo "  results.csv               (20-seed augmentation + combined noise)"
echo "  augmentation_stats.csv    (exact p-values, 95% CIs, effect sizes)"
