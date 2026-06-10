"""
leakage_safe_augmentation.py   (NEW - major revision, Reviewer 1 #6)
====================================================================
Makes the no-leakage argument airtight by comparing THREE protocols under an
identical, strictly leakage-safe cross-validation, across many seeds:

  A. clean -> clean        Train and test both on clean features (the baseline).
  B. noisy -> noisy        Train and test both on noise-injected features.
                           This is the paper's protocol; noise reflects the
                           acquisition condition, applied per-file independently.
  C. noisy(train) -> clean(test)   STRICT augmentation: noise is injected ONLY into
                           the training folds; the test fold stays CLEAN. This is the
                           protocol a reviewer worried might be missing.

In every protocol:
  * the split is at the FILE level (each file is one sample -> no scan leaks across
    folds),
  * the StandardScaler is fit on the training fold ONLY and then applied to the test
    fold,
  * the Random Forest is fit on the training fold ONLY.

There is no augment-before-split duplication (each file yields exactly one feature
vector, so an augmented copy of a training sample can never appear in the test fold).
If the asymmetric effect (Cd improves, Pb does not) persists in protocol C, it cannot
be a leakage artefact.

Reuses the paper pipeline by importing run_experiment.py.

Reads : ../01_data_preparation/data_cache.pkl
Writes: leakage_safe_augmentation.csv        (per metal/protocol/seed: MAPE_mid, R2)
        leakage_safe_stats.csv                (paired B-vs-A and C-vs-A summary)

Usage:
    cd 03_analysis
    python3 leakage_safe_augmentation.py            # 20 seeds, Gaussian @ SNR 30
    python3 leakage_safe_augmentation.py 10 gaussian 30
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from scipy.stats import wilcoxon, t as t_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_experiments"))
import run_experiment as rx  # noqa: E402

DATA_CACHE = "../01_data_preparation/data_cache.pkl"
OUT_LONG = "leakage_safe_augmentation.csv"
OUT_STATS = "leakage_safe_stats.csv"
DEFAULT_SEEDS = [42, 7, 123, 2024, 88, 314, 1, 999, 55, 7777,
                 13, 21, 34, 101, 202, 303, 404, 505, 606, 808]
KEY = ["metal", "conc", "file"]


def _mid_mape(y, yp):
    m = (y >= 20) & (y < 100)
    return float(np.mean(np.abs(y[m] - yp[m]) / y[m]) * 100) if m.sum() else np.nan


def aligned_features(data, noise, snr, seed):
    """Build clean and noisy feature tables aligned row-for-row by file."""
    clean = rx.build_features(data, rx.NOISE_FNS["baseline"], 0, seed).set_index(KEY)
    noisy = rx.build_features(data, rx.NOISE_FNS[noise], snr, seed).set_index(KEY)
    common = clean.index.intersection(noisy.index)
    return clean.loc[common].reset_index(), noisy.loc[common].reset_index()


def run_protocols(clean_df, noisy_df, seed):
    feat = [c for c in clean_df.columns if c not in KEY]
    rows = []
    for metal in ["Cd", "Pb"]:
        cm = clean_df[clean_df.metal == metal].reset_index(drop=True)
        nm = noisy_df[noisy_df.metal == metal].reset_index(drop=True)
        assert (cm["file"].values == nm["file"].values).all(), "row alignment broken"
        Xc, Xn = cm[feat].values, nm[feat].values
        y = cm["conc"].values
        bins = np.digitize(y, rx.BIN_EDGES) - 1
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        preds = {p: np.zeros_like(y, float) for p in ("A_clean", "B_noisy", "C_trainonly")}
        for tr, te in skf.split(Xc, bins):
            # A: clean -> clean
            scA = StandardScaler().fit(Xc[tr])
            preds["A_clean"][te] = (RandomForestRegressor(random_state=seed, **rx.RF_PARAMS)
                                    .fit(scA.transform(Xc[tr]), y[tr])
                                    .predict(scA.transform(Xc[te])))
            # B: noisy -> noisy
            scB = StandardScaler().fit(Xn[tr])
            preds["B_noisy"][te] = (RandomForestRegressor(random_state=seed, **rx.RF_PARAMS)
                                    .fit(scB.transform(Xn[tr]), y[tr])
                                    .predict(scB.transform(Xn[te])))
            # C: noisy(train) -> clean(test)   [scaler fit on noisy train only]
            scC = StandardScaler().fit(Xn[tr])
            preds["C_trainonly"][te] = (RandomForestRegressor(random_state=seed, **rx.RF_PARAMS)
                                        .fit(scC.transform(Xn[tr]), y[tr])
                                        .predict(scC.transform(Xc[te])))
        for proto, yp in preds.items():
            yp = np.clip(yp, 0.1, None)
            ss_res = np.sum((y - yp) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
            rows.append(dict(metal=metal, protocol=proto, seed=seed,
                             MAPE_mid=_mid_mape(y, yp), R2=float(1 - ss_res / ss_tot)))
    return rows


def _ci_t(x, alpha=0.05):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(n)
    h = t_dist.ppf(1 - alpha / 2, n - 1) * se
    return (x.mean() - h, x.mean() + h)


def main():
    seeds = DEFAULT_SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else DEFAULT_SEEDS
    noise = sys.argv[2] if len(sys.argv) > 2 else "gaussian"
    snr = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    with open(DATA_CACHE, "rb") as f:
        data = pickle.load(f)

    rows = []
    for seed in seeds:
        cdf, ndf = aligned_features(data, noise, snr, seed)
        rows += run_protocols(cdf, ndf, seed)
        print(f"  seed={seed} done")
    long = pd.DataFrame(rows)
    long.to_csv(OUT_LONG, index=False)

    print("\nMAPE_mid by protocol (mean +/- 95% CI over seeds):")
    stat_rows = []
    for metal in ["Cd", "Pb"]:
        print(f"\n  {metal}:")
        piv = long[long.metal == metal].pivot(index="seed", columns="protocol", values="MAPE_mid")
        for proto in ["A_clean", "B_noisy", "C_trainonly"]:
            v = piv[proto].values
            lo, hi = _ci_t(v)
            print(f"    {proto:>12}: {v.mean():6.2f}%  (95% CI {lo:5.2f}, {hi:5.2f})")
        # paired tests vs clean baseline
        base = piv["A_clean"].values
        for proto in ["B_noisy", "C_trainonly"]:
            cond = piv[proto].values
            diff = cond - base
            alt = "less" if metal == "Cd" else "two-sided"
            try:
                w, p = wilcoxon(cond, base, alternative=alt, zero_method="wilcox", mode="exact")
            except ValueError:
                w, p = wilcoxon(cond, base, alternative=alt, zero_method="wilcox")
            lo, hi = _ci_t(diff / base * 100)
            stat_rows.append(dict(metal=metal, comparison=f"{proto}_vs_A_clean",
                                  delta_pct=float((diff / base * 100).mean()),
                                  delta_pct_CI_lo=lo, delta_pct_CI_hi=hi,
                                  wilcoxon_p_exact=float(p), n_seeds=len(diff)))
    pd.DataFrame(stat_rows).to_csv(OUT_STATS, index=False)
    print("\nPaired comparisons vs clean baseline:")
    print(pd.DataFrame(stat_rows).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\nSaved {OUT_LONG} and {OUT_STATS}.")


if __name__ == "__main__":
    main()
