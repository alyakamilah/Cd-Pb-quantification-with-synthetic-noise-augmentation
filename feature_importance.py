"""
feature_importance.py   (NEW - major revision, Reviewer 1 #10)
==============================================================
Tests the paper's claim that noise injection makes the dispersion features
(standard-deviation and coefficient-of-variation aggregates) MORE informative.

STRONG option: three complementary importance measures, before vs after augmentation
  1. Permutation importance   (model-agnostic, computed on a held-out split; scored
     with MAPE so it speaks the paper's metric)
  2. Random-Forest built-in   (mean impurity decrease)
  3. SHAP (TreeExplainer)     (optional; skipped automatically if `shap` is absent)

The 17 features fall into three groups by their aggregation suffix:
    *_mean  (central tendency)   *_std  (dispersion)   *_cv  (relative dispersion)
We report the IMPORTANCE SHARE of each group, clean vs augmented, averaged over seeds
with 95% confidence intervals. The expected, claim-supporting result is that the
combined std+cv share rises after augmentation.

Reuses the exact paper pipeline by importing run_experiment.py (no reimplementation).

Reads : ../01_data_preparation/data_cache.pkl
Writes: feature_importance.csv            (per metal/condition/seed/feature, all 3 measures)
        feature_importance_grouped.csv    (per metal/condition/group, share mean + 95% CI)

Usage:
    cd 03_analysis
    python3 feature_importance.py            # default: 10 seeds, Gaussian @ SNR 30
    python3 feature_importance.py 20 gaussian 30
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# import the paper pipeline (feature extraction, noise fns, RF params) -- no duplication
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_experiments"))
import run_experiment as rx  # noqa: E402

try:
    import shap                       # optional
    _HAVE_SHAP = True
except Exception:
    _HAVE_SHAP = False

DATA_CACHE = "../01_data_preparation/data_cache.pkl"
OUT_LONG = "feature_importance.csv"
OUT_GROUPED = "feature_importance_grouped.csv"
DEFAULT_SEEDS = [42, 7, 123, 2024, 88, 314, 1, 999, 55, 7777,
                 13, 21, 34, 101, 202, 303, 404, 505, 606, 808]


def group_of(col):
    """Map a feature name (e.g. 'amp_std') to its aggregation group."""
    return col.rsplit("_", 1)[1]          # mean | std | cv


def importances_for(df_features, seed, metal):
    """Fit RF on a stratified train split; return per-feature permutation, built-in
    and (optional) SHAP importances evaluated on the held-out test split."""
    feat_cols = [c for c in df_features.columns if c not in ("metal", "conc", "file")]
    sub = df_features[df_features.metal == metal]
    X = sub[feat_cols].values
    y = sub["conc"].values
    bins = np.digitize(y, rx.BIN_EDGES) - 1
    Xtr, Xte, ytr, yte, _, _ = train_test_split(
        X, y, bins, test_size=0.30, random_state=seed, stratify=bins)

    sc = StandardScaler().fit(Xtr)             # fit on train only (no leakage)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    rf = RandomForestRegressor(random_state=seed, **rx.RF_PARAMS).fit(Xtr_s, ytr)

    perm = permutation_importance(
        rf, Xte_s, yte, n_repeats=10, random_state=seed,
        scoring="neg_mean_absolute_percentage_error", n_jobs=-1).importances_mean
    builtin = rf.feature_importances_

    if _HAVE_SHAP:
        expl = shap.TreeExplainer(rf)
        sv = expl.shap_values(Xte_s, check_additivity=False)
        shap_imp = np.abs(sv).mean(axis=0)
    else:
        shap_imp = np.full(len(feat_cols), np.nan)

    return pd.DataFrame({
        "feature": feat_cols, "group": [group_of(c) for c in feat_cols],
        "perm": np.clip(perm, 0, None), "builtin": builtin, "shap": shap_imp,
    })


def _share_ci(values, alpha=0.05):
    v = np.asarray(values, float)
    n = len(v)
    if n < 2:
        return (v.mean() if n else np.nan, np.nan, np.nan)
    from scipy.stats import t as t_dist
    m = v.mean(); se = v.std(ddof=1) / np.sqrt(n)
    h = t_dist.ppf(1 - alpha / 2, n - 1) * se
    return (m, m - h, m + h)


def main():
    seeds = DEFAULT_SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else DEFAULT_SEEDS[:10]
    noise = sys.argv[2] if len(sys.argv) > 2 else "gaussian"
    snr = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    if not _HAVE_SHAP:
        print("NOTE: `shap` not installed -> SHAP column will be NaN. "
              "Install with `pip install shap` to enable it.")

    with open(DATA_CACHE, "rb") as f:
        data = pickle.load(f)

    long_rows = []
    for seed in seeds:
        df_clean = rx.build_features(data, rx.NOISE_FNS["baseline"], 0, seed)
        df_aug = rx.build_features(data, rx.NOISE_FNS[noise], snr, seed)
        for metal in ["Cd", "Pb"]:
            for cond, dff in [("clean", df_clean), ("augmented", df_aug)]:
                imp = importances_for(dff, seed, metal)
                imp.insert(0, "seed", seed); imp.insert(0, "condition", cond)
                imp.insert(0, "metal", metal)
                long_rows.append(imp)
        print(f"  seed={seed} done")
    long = pd.concat(long_rows, ignore_index=True)
    long.to_csv(OUT_LONG, index=False)

    # group shares per (metal, condition, seed), then mean +/- CI across seeds
    grouped_rows = []
    for (metal, cond, seed), g in long.groupby(["metal", "condition", "seed"]):
        for measure in ["perm", "builtin", "shap"]:
            tot = g[measure].clip(lower=0).sum()
            if not np.isfinite(tot) or tot <= 0:
                continue
            for grp in ["mean", "std", "cv"]:
                share = g.loc[g.group == grp, measure].clip(lower=0).sum() / tot
                grouped_rows.append(dict(metal=metal, condition=cond, seed=seed,
                                         measure=measure, group=grp, share=share))
    gdf = pd.DataFrame(grouped_rows)

    summary = []
    for (metal, cond, measure, grp), gg in gdf.groupby(["metal", "condition", "measure", "group"]):
        m, lo, hi = _share_ci(gg["share"].values)
        summary.append(dict(metal=metal, condition=cond, measure=measure, group=grp,
                            share_mean=m, share_CI_lo=lo, share_CI_hi=hi, n_seeds=len(gg)))
    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT_GROUPED, index=False)

    # Headline table: permutation-importance share, clean vs augmented, std+cv combined
    print("\nPermutation-importance share by feature group (mean over seeds):")
    perm = sdf[sdf.measure == "perm"]
    for metal in ["Cd", "Pb"]:
        print(f"\n  {metal}:")
        piv = perm[perm.metal == metal].pivot(index="group", columns="condition",
                                              values="share_mean")
        for grp in ["mean", "std", "cv"]:
            if grp in piv.index:
                cl = piv.loc[grp].get("clean", np.nan)
                au = piv.loc[grp].get("augmented", np.nan)
                print(f"    {grp:>5}:  clean={cl:6.3f}  ->  augmented={au:6.3f}")
        disp_cl = perm[(perm.metal == metal) & (perm.condition == "clean") &
                       (perm.group.isin(["std", "cv"]))]["share_mean"].sum()
        disp_au = perm[(perm.metal == metal) & (perm.condition == "augmented") &
                       (perm.group.isin(["std", "cv"]))]["share_mean"].sum()
        print(f"    std+cv combined:  clean={disp_cl:6.3f}  ->  augmented={disp_au:6.3f}")
    print(f"\nSaved {OUT_LONG} and {OUT_GROUPED}.")


if __name__ == "__main__":
    main()
