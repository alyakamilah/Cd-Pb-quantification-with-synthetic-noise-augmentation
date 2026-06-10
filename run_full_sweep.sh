#!/usr/bin/env bash
# run_full_sweep.sh
# =================
# Runs the complete experimental grid used in the paper.
#
# Total configurations: 73
#   - 1 baseline (clean)  x 3 seeds  = 3 runs
#   - 4 noise types x 6 SNR levels x 3 seeds = 72 runs
# Plus optional: multi-classifier comparison (32 runs).
#
# Approximate runtime: 30-60 minutes depending on hardware.
# Each run takes ~10-30 seconds (feature extraction + RF CV).
#
# Usage:
#   chmod +x run_full_sweep.sh
#   ./run_full_sweep.sh

set -e
cd "$(dirname "$0")"

NOISE_TYPES=(gaussian student_t pink hetero)
SNR_LEVELS=(5 10 15 20 25 30)
SEEDS=(42 7 123)

echo "==================================================================="
echo "Full SNR sweep + clean baseline"
echo "==================================================================="

# Clean baseline (3 seeds)
for seed in "${SEEDS[@]}"; do
    echo "[BASELINE seed=$seed]"
    python3 run_experiment.py baseline 0 "$seed"
done

# Full grid: 4 noise types x 6 SNR levels x 3 seeds = 72 runs
for noise in "${NOISE_TYPES[@]}"; do
    for snr in "${SNR_LEVELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "[$noise SNR=$snr seed=$seed]"
            python3 run_experiment.py "$noise" "$snr" "$seed"
        done
    done
done

echo
echo "Computing statistical analysis..."
python3 statistical_analysis.py

echo
echo "==================================================================="
echo "Multi-classifier confirmation (clean + Gaussian SNR=30)"
echo "==================================================================="

MODELS=(RF XGBoost SVR MLP)
METALS=(Cd Pb)

for model in "${MODELS[@]}"; do
    for metal in "${METALS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # Clean baseline
            echo "[$model $metal baseline seed=$seed]"
            python3 run_multiclassifier.py "$model" baseline 0 "$metal" "$seed"
            # Gaussian SNR=30
            echo "[$model $metal gaussian SNR=30 seed=$seed]"
            python3 run_multiclassifier.py "$model" gaussian 30 "$metal" "$seed"
        done
    done
done

echo
echo "All experiments complete."
echo "  Main results:           results.csv"
echo "  Statistical analysis:   augmentation_stats.csv"
echo "  Multi-classifier:       multi_classifier_results.csv"
