"""
statistical_analysis.py  (REVISED for major revision)
======================================================
Statistical analysis of the noise-augmentation effect (paper Section 2.9 / Table 1).

WHAT CHANGED vs the original (addresses Reviewer 2 "statistical significance
incomplete" and Reviewer 1 #5):
  * PAIRED design. The clean baseline and each noise condition are compared
    seed-by-seed (paired), instead of testing per-seed noise values against a single
    baseline point. The same seed drives both runs, so the difference is a within-seed
    paired quantity.
  * EXACT p-values. Reports the exact Wilcoxon signed-rank p-value (non-parametric)
    alongside the paired t-test p-value. No more "p < 0.05".
  * 95% CONFIDENCE INTERVALS on the effect (absolute MAPE change and percentage
    change), via Student-t and a percentile bootstrap.
  * EFFECT SIZES. Cohen's d_z (paired) and the matched-pairs rank-biserial correlation.

Reads : results.csv  (produced by run_experiment.py / run_revision_sweep.sh)
Writes: augmentation_stats.csv      (one row per metal x noise condition)

Run AFTER you have many seeds in results.csv (Reviewer 1 #5 asks for >=10-20):
    ./run_revision_sweep.sh
    python3 ../03_analysis/statistical_analysis.py
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, t as t_dist

RESULTS_CSV = "results.csv"
OUTPUT_CSV = "augmentation_stats.csv"
NOISE_TYPES = ["gaussian", "student_t", "pink", "hetero",
               "gaussian_pink", "gaussian_student_t", "gaussian_hetero",
               "pink_student_t", "gaussian_pink_student_t"]
TEST_SNR = 30
N_BOOT = 10000
BOOT_SEED = 20260602


def _ci_t(x, alpha=0.05):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(n)
    h = t_dist.ppf(1 - alpha / 2, n - 1) * se
    return (m - h, m + h)


def _ci_bootstrap(x, alpha=0.05, n_boot=N_BOOT, seed=BOOT_SEED):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return (float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


def _cohen_dz(diff):
    diff = np.asarray(diff, float); sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else np.nan


def _rank_biserial(diff):
    diff = np.asarray(diff, float); nz = diff[diff != 0]
    if len(nz) == 0:
        return 0.0
    ranks = pd.Series(np.abs(nz)).rank().values
    total = ranks.sum()
    return float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / total)


def main():
    df = pd.read_csv(RESULTS_CSV)
    df = df.drop_duplicates(subset=["method", "noise_type", "snr_db", "seed", "metal"])
    rows = []
    for metal in ["Cd", "Pb"]:
        alt = "less" if metal == "Cd" else "two-sided"
        base = (df[(df.noise_type == "baseline") & (df.metal == metal)]
                .set_index("seed")["MAPE_mid"])
        for noise in NOISE_TYPES:
            cond = (df[(df.noise_type == noise) & (df.snr_db == TEST_SNR) &
                       (df.metal == metal)].set_index("seed")["MAPE_mid"])
            seeds = sorted(set(base.index) & set(cond.index))
            if len(seeds) < 2:
                continue
            b = base.loc[seeds].values; c = cond.loc[seeds].values
            diff = c - b; pct = diff / b * 100.0
            t_stat, p_t = ttest_rel(c, b, alternative=alt)
            try:
                w_stat, p_w = wilcoxon(c, b, alternative=alt, zero_method="wilcox", mode="exact")
            except ValueError:
                w_stat, p_w = wilcoxon(c, b, alternative=alt, zero_method="wilcox")
            ci_abs_t = _ci_t(diff); ci_abs_b = _ci_bootstrap(diff); ci_pct_t = _ci_t(pct)
            rows.append({
                "metal": metal, "noise_type": noise, "snr_db": TEST_SNR, "n_seeds": len(seeds),
                "baseline_MAPE_mid": float(b.mean()),
                "baseline_CI_lo": _ci_t(b)[0], "baseline_CI_hi": _ci_t(b)[1],
                "noise_MAPE_mid": float(c.mean()),
                "noise_CI_lo": _ci_t(c)[0], "noise_CI_hi": _ci_t(c)[1],
                "delta_abs": float(diff.mean()),
                "delta_abs_CI_lo_t": ci_abs_t[0], "delta_abs_CI_hi_t": ci_abs_t[1],
                "delta_abs_CI_lo_boot": ci_abs_b[0], "delta_abs_CI_hi_boot": ci_abs_b[1],
                "delta_pct": float(pct.mean()),
                "delta_pct_CI_lo": ci_pct_t[0], "delta_pct_CI_hi": ci_pct_t[1],
                "t_statistic": float(t_stat), "p_value_ttest": float(p_t),
                "wilcoxon_stat": float(w_stat), "p_value_wilcoxon_exact": float(p_w),
                "cohen_dz": _cohen_dz(diff), "rank_biserial": _rank_biserial(diff),
                "alternative": alt,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 50)
    show = ["metal", "noise_type", "n_seeds", "baseline_MAPE_mid", "noise_MAPE_mid",
            "delta_pct", "delta_pct_CI_lo", "delta_pct_CI_hi",
            "p_value_wilcoxon_exact", "p_value_ttest", "cohen_dz"]
    print("\nAugmentation effect at SNR = %d dB (paired across %s seeds):" %
          (TEST_SNR, out["n_seeds"].max() if len(out) else 0))
    if len(out):
        print(out[show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nSaved {OUTPUT_CSV}.")


if __name__ == "__main__":
    main()
