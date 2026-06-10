"""
analyze_noise_injection.py
==========================
Statistical analysis of the synthetic noise injection experiment.

Computes the central statistical findings of the paper (Section 2.9, Table 1):
    - One-sample t-tests of MAPE_mid against clean baseline
    - Tests are ONE-SIDED for Cd2+ (H1: noise reduces MAPE)
    - Tests are TWO-SIDED for Pb2+ (direction not preregistered)

Reads:  results.csv (from run_noise_injection_experiment.py)
Writes: augmentation_stats.csv

Usage:
    python3 analyze_noise_injection.py
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp


RESULTS_CSV = "../02_experiments/results.csv"
STATS_CSV = "augmentation_stats.csv"

# Noise distributions tested
NOISE_TYPES = ['gaussian', 'student_t', 'pink', 'hetero']

# Focal SNR for the augmentation comparison (Table 1, Fig 10)
FOCAL_SNR = 30


def main():
    print(f"Reading {RESULTS_CSV}...")
    df = pd.read_csv(RESULTS_CSV)

    # Deduplicate in case of re-runs
    df = df.drop_duplicates(
        subset=['method', 'noise_type', 'snr_db', 'seed', 'metal']
    )
    print(f"  -> {len(df)} unique rows")
    print()

    # Compute clean baseline per metal (mean across seeds)
    baseline = {}
    for metal in ['Cd', 'Pb']:
        sub = df[(df['noise_type'] == 'baseline') & (df['metal'] == metal)]
        baseline[metal] = float(sub['MAPE_mid'].mean())

    print("Clean baseline MAPE_mid:")
    for metal, val in baseline.items():
        print(f"  {metal}2+: {val:.2f}%")
    print()

    # One-sample t-test per (metal, noise_type) at focal SNR
    rows = []
    print(f"One-sample t-tests at SNR = {FOCAL_SNR} dB:")
    print(f"  {'Metal':6s} {'Noise':12s} {'MAPE_mid':>10s} {'Delta':>8s} "
          f"{'t-stat':>8s} {'p-value':>10s} {'alt':>10s}")
    print("  " + "-" * 70)

    for metal in ['Cd', 'Pb']:
        alt = 'less' if metal == 'Cd' else 'two-sided'
        bl = baseline[metal]

        for noise in NOISE_TYPES:
            sub = df[
                (df['noise_type'] == noise)
                & (df['snr_db'] == FOCAL_SNR)
                & (df['metal'] == metal)
            ]
            if len(sub) < 2:
                continue

            mapes = sub['MAPE_mid'].values
            t_stat, p_val = ttest_1samp(mapes, bl, alternative=alt)

            mean_mape = float(np.mean(mapes))
            std_mape = float(np.std(mapes, ddof=1))
            delta_pct = (mean_mape - bl) / bl * 100

            # Significance marker
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "n.s."

            print(f"  {metal:6s} {noise:12s} "
                  f"{mean_mape:>8.2f}±{std_mape:.2f} "
                  f"{delta_pct:>+7.1f}% "
                  f"{t_stat:>8.3f} {p_val:>10.5f} {alt:>10s} {sig}")

            rows.append({
                'metal': metal,
                'noise_type': noise,
                'snr_db': FOCAL_SNR,
                'mean_MAPE_mid': mean_mape,
                'std_MAPE_mid': std_mape,
                'baseline_MAPE_mid': bl,
                'delta_pct': delta_pct,
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'alternative': alt,
                'significance': sig,
                'n_seeds': len(sub),
            })

    df_stats = pd.DataFrame(rows)
    df_stats.to_csv(STATS_CSV, index=False)
    print()
    print(f"Saved {STATS_CSV}")

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY: Asymmetric noise-augmentation effect")
    print("=" * 70)

    cd_sig = df_stats[(df_stats['metal'] == 'Cd')
                      & (df_stats['p_value'] < 0.05)]
    pb_sig = df_stats[(df_stats['metal'] == 'Pb')
                      & (df_stats['p_value'] < 0.05)]

    print(f"  Cd2+: {len(cd_sig)}/4 noise distributions significant "
          f"(mean reduction: "
          f"{df_stats[df_stats['metal'] == 'Cd']['delta_pct'].mean():.1f}%)")
    print(f"  Pb2+: {len(pb_sig)}/4 noise distributions significant "
          f"(mean change:    "
          f"{df_stats[df_stats['metal'] == 'Pb']['delta_pct'].mean():+.1f}%)")


if __name__ == '__main__':
    main()
