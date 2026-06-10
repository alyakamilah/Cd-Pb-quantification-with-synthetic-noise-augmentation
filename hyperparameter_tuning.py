"""
hyperparameter_tuning.py   (NEW - major revision, Reviewer 1 #4)
================================================================
Answers "why no hyperparameter tuning?" with the STRONG option: a proper
GridSearchCV, run inside a leakage-safe NESTED cross-validation (the inner loop
selects hyperparameters on training folds only; the outer loop measures the tuned
score on untouched test folds). This simultaneously reinforces the no-leakage
argument (Reviewer 1 #6).

It reports three things the rebuttal needs:
  1. Default vs tuned MAPE_mid (nested, unbiased) for Cd and Pb -- shows how much,
     if anything, tuning buys over the paper's fixed settings.
  2. The hyperparameters most frequently selected across outer folds/seeds.
  3. That the asymmetric augmentation effect (Cd improves, Pb does not) PERSISTS
     under tuned models -- i.e. the finding is not an artefact of one RF setting.

A one-at-a-time SENSITIVITY table (n_estimators / max_depth / min_samples_leaf) is
also written as supporting evidence of robustness.

Reuses the paper pipeline by importing run_experiment.py.

Reads : ../01_data_preparation/data_cache.pkl
Writes: hyperparameter_tuning.csv         (nested default-vs-tuned, per metal/condition/seed)
        hyperparameter_sensitivity.csv     (one-at-a-time MAPE_mid grid)

Usage:
    cd 02_experiments
    python3 hyperparameter_tuning.py            # default: 5 seeds
    python3 hyperparameter_tuning.py 10
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

import run_experiment as rx                       # same folder; reuse pipeline

DATA_CACHE = "../01_data_preparation/data_cache.pkl"
OUT_TUNE = "hyperparameter_tuning.csv"
OUT_SENS = "hyperparameter_sensitivity.csv"
DEFAULT_SEEDS = [42, 7, 123, 2024, 88, 314, 1, 999, 55, 7777]

# GridSearchCV space (Reviewer asked for n_estimators / max_depth / min_samples_split).
# On a normal multi-core machine keep N_JOBS=-1 (default). In constrained sandboxes
# where joblib worker spawning is slow, set HP_NJOBS=1. HP_FAST=1 uses a tiny grid
# (for a quick end-to-end check only -- use the full grid for the paper).
N_JOBS = int(os.environ.get("HP_NJOBS", "-1"))
if os.environ.get("HP_FAST") == "1":
    PARAM_GRID = {"n_estimators": [100], "max_depth": [10, None],
                  "min_samples_split": [2], "min_samples_leaf": [1, 2]}
else:
    PARAM_GRID = {
        "n_estimators": [100, 200, 400],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2],
    }
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)


def _mid_mape(y, yp):
    m = (y >= 20) & (y < 100)
    return float(np.mean(np.abs(y[m] - yp[m]) / y[m]) * 100) if m.sum() else np.nan


def nested_cv(df_features, seed, metal):
    """Nested CV: inner GridSearchCV picks params on train; outer measures tuned and
    default MAPE_mid on held-out folds. Returns (tuned_mid, default_mid, best_params_list)."""
    feat_cols = [c for c in df_features.columns if c not in ("metal", "conc", "file")]
    sub = df_features[df_features.metal == metal]
    X = sub[feat_cols].values
    y = sub["conc"].values
    bins = np.digitize(y, rx.BIN_EDGES) - 1

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    yp_tuned = np.zeros_like(y, float)
    yp_default = np.zeros_like(y, float)
    best_params = []
    for tr, te in outer.split(X, bins):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        gs = GridSearchCV(
            RandomForestRegressor(random_state=seed, n_jobs=1),
            PARAM_GRID, scoring=mape_scorer, cv=inner.split(Xtr, bins[tr]), n_jobs=N_JOBS)
        gs.fit(Xtr, y[tr])
        best_params.append(gs.best_params_)
        yp_tuned[te] = gs.best_estimator_.predict(Xte)
        # default (paper) model on the same fold for a matched comparison
        rfd = RandomForestRegressor(random_state=seed, **rx.RF_PARAMS).fit(Xtr, y[tr])
        yp_default[te] = rfd.predict(Xte)
    yp_tuned = np.clip(yp_tuned, 0.1, None)
    yp_default = np.clip(yp_default, 0.1, None)
    return _mid_mape(y, yp_tuned), _mid_mape(y, yp_default), best_params


def sensitivity(df_features, seed):
    """One-at-a-time hyperparameter sensitivity via the paper's own evaluate_rf
    (temporarily overriding RF_PARAMS so the exact CV code is reused)."""
    base = dict(rx.RF_PARAMS)
    grid = {"n_estimators": [50, 100, 200, 400],
            "max_depth": [5, 10, 20, None],
            "min_samples_leaf": [1, 2, 4]}
    rows = []
    saved = rx.RF_PARAMS
    try:
        for param, values in grid.items():
            for v in values:
                params = dict(base); params[param] = v
                rx.RF_PARAMS = params
                for r in rx.evaluate_rf(df_features, seed):
                    rows.append(dict(seed=seed, param=param, value=str(v),
                                     metal=r["metal"], MAPE_mid=r["MAPE_mid"], R2=r["R2"]))
    finally:
        rx.RF_PARAMS = saved
    return rows


def main():
    seeds = DEFAULT_SEEDS[:int(sys.argv[1])] if len(sys.argv) > 1 else DEFAULT_SEEDS[:5]
    with open(DATA_CACHE, "rb") as f:
        data = pickle.load(f)

    tune_rows, sens_rows, all_best = [], [], []
    for seed in seeds:
        df_clean = rx.build_features(data, rx.NOISE_FNS["baseline"], 0, seed)
        df_aug = rx.build_features(data, rx.NOISE_FNS["gaussian"], 30, seed)
        for cond, dff in [("clean", df_clean), ("augmented", df_aug)]:
            for metal in ["Cd", "Pb"]:
                tuned, default, bps = nested_cv(dff, seed, metal)
                tune_rows.append(dict(metal=metal, condition=cond, seed=seed,
                                      default_MAPE_mid=default, tuned_MAPE_mid=tuned,
                                      improvement_pct=(tuned - default) / default * 100))
                all_best += [(metal, tuple(sorted(p.items()))) for p in bps]
                print(f"    seed={seed} {cond} {metal}: default={default:.2f}% tuned={tuned:.2f}%")
        sens_rows += sensitivity(df_clean, seed)     # sensitivity reported on clean data
        print(f"  seed={seed} done")

    tdf = pd.DataFrame(tune_rows); tdf.to_csv(OUT_TUNE, index=False)
    pd.DataFrame(sens_rows).to_csv(OUT_SENS, index=False)

    print("\nNested-CV default vs tuned MAPE_mid (mean over seeds):")
    agg = tdf.groupby(["metal", "condition"]).agg(
        default=("default_MAPE_mid", "mean"), tuned=("tuned_MAPE_mid", "mean"),
        impr=("improvement_pct", "mean")).reset_index()
    print(agg.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nDoes the augmentation effect persist under tuning?")
    for metal in ["Cd", "Pb"]:
        cl = agg[(agg.metal == metal) & (agg.condition == "clean")]["tuned"].values
        au = agg[(agg.metal == metal) & (agg.condition == "augmented")]["tuned"].values
        if len(cl) and len(au):
            d = (au[0] - cl[0]) / cl[0] * 100
            print(f"  {metal}: tuned clean={cl[0]:.2f}%  tuned augmented={au[0]:.2f}%  "
                  f"-> delta={d:+.1f}%")

    print("\nMost frequently selected hyperparameters (across outer folds & seeds):")
    for metal in ["Cd", "Pb"]:
        top = Counter([bp for (m, bp) in all_best if m == metal]).most_common(1)
        if top:
            print(f"  {metal}: {dict(top[0][0])}  (selected {top[0][1]}x)")
    print(f"\nSaved {OUT_TUNE} and {OUT_SENS}.")


if __name__ == "__main__":
    main()
