"""
run_multiclassifier.py
======================
Multi-classifier confirmation experiment (Section 3.5 of paper).

Purpose:
    Tests whether the asymmetric noise-augmentation effect is specific to
    Random Forest or generalises across model classes. Evaluates four
    regressors operating on the same multi-scan feature representation
    under matched conditions (clean baseline + Gaussian noise at SNR=30 dB).

Models compared:
    - RF       : Random Forest (replicates paper's primary finding)
    - XGBoost  : Gradient-boosted trees
    - SVR      : epsilon-insensitive Support Vector Regression with RBF kernel
    - MLP      : Multilayer Perceptron (2 hidden layers, 64-32 units)

Configuration (via command-line args):
    model:      RF | XGBoost | SVR | MLP
    noise_type: baseline | gaussian
    snr_db:     30 (only used if noise_type=gaussian; ignored for baseline)
    metal:      Cd | Pb
    seed:       integer (42, 7, 123 used in paper)

Usage:
    python3 run_multiclassifier.py XGBoost gaussian 30 Cd 42
    python3 run_multiclassifier.py SVR baseline 0 Pb 7

Output:
    Appends one row to multi_classifier_results.csv.

Reuses feature extraction from run_experiment.py to ensure consistency.
"""

import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# Reuse feature extraction logic
from run_experiment import (
    NOISE_FNS, build_features, BIN_EDGES, RF_PARAMS,
    DATA_CACHE,
)

warnings.filterwarnings('ignore')

RESULTS_CSV = "multi_classifier_results.csv"


# ------------------------------------------------------------------
# Model configurations
# ------------------------------------------------------------------
def make_model(name, seed):
    """Construct regressor with paper hyperparameters.

    Hyperparameters chosen for fair comparison:
    - Tree-based models: matched depth and ensemble size
    - SVR: RBF kernel with auto gamma
    - MLP: moderate depth with early stopping
    """
    if name == 'RF':
        return RandomForestRegressor(random_state=seed, **RF_PARAMS)
    elif name == 'XGBoost':
        return XGBRegressor(
            n_estimators=100, max_depth=10, learning_rate=0.1,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
    elif name == 'SVR':
        return SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
    elif name == 'MLP':
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def evaluate_model(df_features, model_name, metal, seed):
    """Train one model on one metal with stratified 5-fold CV."""
    feat_cols = [c for c in df_features.columns
                 if c not in ('metal', 'conc', 'file')]
    sub = df_features[df_features['metal'] == metal].copy()
    if len(sub) < 30:
        return None

    X = sub[feat_cols].values
    y = sub['conc'].values
    bins = np.digitize(y, BIN_EDGES) - 1

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred = np.zeros_like(y, dtype=float)
    for tr, te in skf.split(X, bins):
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr])
        Xte = sc.transform(X[te])
        m = make_model(model_name, seed)
        m.fit(Xtr, y[tr])
        y_pred[te] = m.predict(Xte)
    y_pred = np.clip(y_pred, 0.1, None)

    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred) * 100
    rel_err = np.abs(y - y_pred) / y * 100
    mid_mask = (y >= 20) & (y < 100)
    high_mask = y >= 100

    return {
        'R2': r2,
        'MAPE': mape,
        'MAPE_mid': float(np.mean(rel_err[mid_mask])) if mid_mask.sum() > 0 else np.nan,
        'MAPE_high': float(np.mean(rel_err[high_mask])) if high_mask.sum() > 0 else np.nan,
    }


def main():
    if len(sys.argv) != 6:
        print("Usage: python3 run_multiclassifier.py <model> <noise_type> "
              "<snr_db> <metal> <seed>")
        print("  model:      RF | XGBoost | SVR | MLP")
        print("  noise_type: baseline | gaussian")
        print("  snr_db:     integer (use 0 for baseline)")
        print("  metal:      Cd | Pb")
        print("  seed:       integer")
        sys.exit(1)

    model_name = sys.argv[1]
    noise_type = sys.argv[2]
    snr_db = int(sys.argv[3])
    metal = sys.argv[4]
    seed = int(sys.argv[5])

    print(f"Loading data...")
    with open(DATA_CACHE, 'rb') as f:
        data = pickle.load(f)

    print(f"Building features (noise={noise_type}, snr={snr_db}, seed={seed})...")
    t0 = time.time()
    df = build_features(data, NOISE_FNS[noise_type], snr_db, seed)
    print(f"  {len(df)} feature rows ({time.time()-t0:.1f}s)")

    print(f"Evaluating {model_name} on {metal}...")
    metrics = evaluate_model(df, model_name, metal, seed)
    if metrics is None:
        print(f"  ERROR: insufficient data for {metal}")
        sys.exit(1)

    print(f"  R²={metrics['R2']:.3f}  MAPE={metrics['MAPE']:.1f}%  "
          f"mid={metrics['MAPE_mid']:.1f}%")

    row = pd.DataFrame([{
        'model': model_name,
        'noise_type': noise_type,
        'snr_db': snr_db,
        'metal': metal,
        'seed': seed,
        **metrics,
    }])
    try:
        existing = pd.read_csv(RESULTS_CSV)
        row = pd.concat([existing, row], ignore_index=True)
    except FileNotFoundError:
        pass
    row.to_csv(RESULTS_CSV, index=False)
    print(f"Saved to {RESULTS_CSV}.")


if __name__ == '__main__':
    main()
