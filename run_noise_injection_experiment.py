"""
run_noise_injection_experiment.py
=================================
Complete synthetic noise injection experiment for the CEJA paper:
"Synthetic noise injection asymmetrically affects machine-learning quantification
 of Cd2+ and Pb2+ in multi-scan cyclic voltammetry"

This script reproduces the central empirical contribution of the paper:
    For each (noise_type, snr_db, seed) configuration:
        1. Load 1500 raw voltammograms
        2. Inject synthetic noise into each raw scan (Section 2.6)
        3. Extract 7 peak features per scan (Section 2.5)
        4. Aggregate across 9 non-conditioning scans -> 17-dim feature vector
        5. Train Random Forest with stratified 5-fold CV (Section 2.7)
        6. Compute per-band MAPE (low/mid/high) and R^2

Used to generate Tables 1, Fig 8, Fig 9, Fig 10 of the paper.

Configuration:
    Args (command-line): noise_type snr_db seed
    or edit BATCH_CONFIG at bottom to run full sweep.

Single config example:
    python3 run_noise_injection_experiment.py gaussian 30 42

Output:
    Appends one row per metal (Cd, Pb) to results.csv:
        method, noise_type, snr_db, seed, metal,
        R2, MAPE, MAPE_low, MAPE_mid, MAPE_high

Reference paper sections:
    Section 2.5: Multi-scan feature extraction
    Section 2.6: Synthetic noise injection
    Section 2.7: Random Forest prediction
    Section 2.8: Cross-validation protocol
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '05_utilities'))
import time
import pickle
import argparse
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
from scipy.stats import skew

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
)

from noise_injection import inject_noise, NOISE_TYPES, SNR_LEVELS, SEEDS


# =====================================================================
# Configuration (matches paper Sections 2.5-2.8)
# =====================================================================

DATA_CACHE = "../01_data_preparation/data_cache.pkl"
RESULTS_CSV = "results.csv"

# Concentration bands for stratification + per-band MAPE
BIN_EDGES = [0, 10, 30, 100, 500, 1100]  # ppm

# Random Forest hyperparameters (Section 2.7 of paper)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_leaf': 2,
    'n_jobs': -1,
}

# Savitzky-Golay smoothing parameters
SG_WINDOW = 11
SG_POLY = 3

# Number of CV folds
N_FOLDS = 5

# Number of non-conditioning scans per measurement
N_SCANS_USED = 9


# =====================================================================
# Per-scan feature extraction (Section 2.5 of paper)
# =====================================================================

def split_forward_reverse(E, y):
    """
    Split a CV scan into forward and reverse branches at the turning point.

    The turning point is detected as the first sign change in dE/dx.

    Parameters
    ----------
    E, y : numpy.ndarray
        Potential and current arrays (same length)

    Returns
    -------
    (E_fwd, y_fwd), (E_rev, y_rev) : tuple of tuples
        Forward and reverse branch arrays
    """
    dE = np.diff(E)
    sign_changes = np.where(np.diff(np.sign(dE)) != 0)[0]
    if len(sign_changes) == 0:
        return (E, y), (np.array([]), np.array([]))
    t = sign_changes[0] + 1
    return (E[:t+1], y[:t+1]), (E[t+1:], y[t+1:])


def extract_features_one_scan(E, y, peak_window):
    """
    Extract 7 peak-related features from one CV scan.

    Implements the per-scan feature extractor phi(*) from Section 2.5:
        1. amp  : peak amplitude (baseline-corrected, signed)
        2. pos  : peak position (potential of peak maximum, V)
        3. wid  : FWHM-derived peak width (sigma)
        4. area : peak area (integrated magnitude)
        5. skew : peak skewness
        6. snr  : peak signal-to-noise ratio
        7. bl   : baseline value at peak position

    The branch (forward or reverse) with the larger in-window peak amplitude
    is selected for analysis.

    Parameters
    ----------
    E : numpy.ndarray
        Potential array (V vs Ag/AgCl)
    y : numpy.ndarray
        Current array (uA)
    peak_window : tuple of float
        (E_min, E_max) analytical peak window in V

    Returns
    -------
    features : dict or None
        Dictionary with 7 features, or None if extraction failed.
    """
    # Smooth signal with Savitzky-Golay
    yf = savgol_filter(y, SG_WINDOW, SG_POLY)

    # Quadratic baseline from data OUTSIDE the peak window
    mask_outside = (E < peak_window[0]) | (E > peak_window[1])
    if mask_outside.sum() >= 4:
        baseline = np.polyval(np.polyfit(E[mask_outside], yf[mask_outside], 2), E)
    else:
        baseline = np.full_like(yf, float(np.median(yf)))

    # Split forward and reverse branches
    (E_fwd, y_fwd), (E_rev, y_rev) = split_forward_reverse(E, yf)
    (_, b_fwd), (_, b_rev) = split_forward_reverse(E, baseline)

    def amplitude_in_window(E_arr, y_arr):
        m = (E_arr >= peak_window[0]) & (E_arr <= peak_window[1])
        return float(np.ptp(y_arr[m])) if m.sum() > 0 else 0.0

    # Pick branch with larger peak amplitude in the analytical window
    if amplitude_in_window(E_rev, y_rev) >= amplitude_in_window(E_fwd, y_fwd):
        E_use, y_use, b_use = E_rev, y_rev, b_rev
    else:
        E_use, y_use, b_use = E_fwd, y_fwd, b_fwd

    # Restrict to peak window
    mask = (E_use >= peak_window[0]) & (E_use <= peak_window[1])
    if mask.sum() < 3:
        return None

    E_w = E_use[mask]
    y_w = y_use[mask]
    b_w = b_use[mask]
    y_corr = y_w - b_w  # baseline-corrected signal

    # Peak position and amplitude
    i_peak = int(np.argmax(np.abs(y_corr)))
    amp = float(y_corr[i_peak])

    # FWHM-derived width: find half-max crossings
    half = abs(amp) / 2.0
    above = (y_corr >= half) if amp > 0 else (y_corr <= -half)
    if above.sum() >= 2:
        idxs = np.where(above)[0]
        sigma = float(abs(E_w[idxs[-1]] - E_w[idxs[0]])) / 2.355  # FWHM -> sigma
    else:
        sigma = 0.0

    # Peak area (integrated magnitude)
    area = float(np.trapz(np.abs(y_corr), E_w))

    # Peak skewness
    sk = float(skew(y_corr)) if len(y_corr) > 3 and np.std(y_corr) > 0 else 0.0

    # Peak SNR (peak amplitude / signal std on chosen branch)
    noise_proxy = float(np.std(y_use)) if len(y_use) > 5 else 1.0
    peak_snr = abs(amp) / max(noise_proxy, 1e-3)

    # Baseline value at peak position
    bl = float(b_w[i_peak])

    return {
        'amp':  amp,
        'pos':  float(E_w[i_peak]),
        'wid':  sigma,
        'area': area,
        'skew': sk,
        'snr':  peak_snr,
        'bl':   bl,
    }


# =====================================================================
# Multi-scan aggregation (Section 2.5 of paper)
# =====================================================================

def aggregate_multiscan(scan_features):
    """
    Aggregate per-scan features across the 9 non-conditioning scans.

    Aggregation rules (Section 2.5):
        - All 7 features: mean and std (14 features)
        - amp, pos, area: also coefficient of variation (3 features)
        Total: 17 features per voltammogram

    Parameters
    ----------
    scan_features : list of dict
        Per-scan feature dicts from extract_features_one_scan()

    Returns
    -------
    agg : dict
        17-dimensional aggregated feature vector
    """
    df = pd.DataFrame(scan_features)
    agg = {}

    # Mean and std for all 7 features
    for col in df.columns:
        agg[f'{col}_mean'] = float(np.mean(df[col].values))
        agg[f'{col}_std'] = float(np.std(df[col].values))

    # CV (coefficient of variation) for amp, pos, area only
    for col in ['amp', 'pos', 'area']:
        m_abs = abs(agg[f'{col}_mean'])
        if m_abs > 0:
            agg[f'{col}_cv'] = agg[f'{col}_std'] / (m_abs + 1e-6)
        else:
            agg[f'{col}_cv'] = 0.0

    return agg


# =====================================================================
# Build feature matrix for one experimental configuration
# =====================================================================

def build_feature_matrix(data, noise_type, snr_db, seed):
    """
    Apply noise injection + feature extraction across all measurements.

    For each measurement:
        1. Inject noise into each raw scan (or skip if noise_type='baseline')
        2. Extract 7 features per scan
        3. Aggregate to 17-dim vector

    Parameters
    ----------
    data : list of dict
        List of measurement dicts, each with:
            'metal' : 'Cd' or 'Pb'
            'conc'  : concentration in ppm
            'pw'    : peak window (E_min, E_max) in V
            'scans' : list of (E, y) tuples
    noise_type : str
        'baseline' (no injection) or one of NOISE_TYPES keys
    snr_db : float
        Target SNR if injecting noise (ignored for baseline)
    seed : int
        Random seed (deterministic feature extraction across runs)

    Returns
    -------
    df_features : pandas.DataFrame
        One row per measurement, columns:
            metal, conc, file, + 17 feature columns
    """
    rows = []

    for meas_idx, d in enumerate(data):
        # Per-measurement seed for independent noise across measurements
        # within one configuration, but reproducible across runs
        meas_seed = seed * 1000 + meas_idx

        scan_feats = []
        for scan_idx, (E, y) in enumerate(d['scans']):
            # Inject noise (or pass through if baseline)
            if noise_type == 'baseline':
                y_used = y
            else:
                # Unique seed per scan for independent realisations
                scan_seed = meas_seed * 100 + scan_idx
                y_used = inject_noise(y, noise_type, snr_db, scan_seed)

            # Extract features
            features = extract_features_one_scan(E, y_used, d['pw'])
            if features is not None:
                scan_feats.append(features)

        # Need at least 3 valid scans to aggregate
        if len(scan_feats) < 3:
            continue

        agg = aggregate_multiscan(scan_feats)
        rows.append({
            'metal': d['metal'],
            'conc': float(d['conc']),
            'file': d.get('file', f'meas_{meas_idx}'),
            **agg,
        })

    return pd.DataFrame(rows)


# =====================================================================
# Random Forest evaluation with stratified 5-fold CV
# =====================================================================

def evaluate_rf(df_features, seed):
    """
    Train Random Forest and evaluate via stratified 5-fold CV (per metal).

    Implements Section 2.7-2.8 of paper:
        - Stratified 5-fold over concentration bins
        - Standardize features per fold
        - RF with paper hyperparameters
        - Predictions clipped at 0.1 ppm

    Parameters
    ----------
    df_features : pandas.DataFrame
        Output of build_feature_matrix()
    seed : int
        Random seed for both fold assignment and RF

    Returns
    -------
    results : list of dict
        One dict per metal with keys:
            metal, R2, MAPE, MAPE_low, MAPE_mid, MAPE_high
    """
    feat_cols = [c for c in df_features.columns
                 if c not in ('metal', 'conc', 'file')]

    results = []

    for metal in ['Cd', 'Pb']:
        sub = df_features[df_features['metal'] == metal].copy()
        if len(sub) < 30:
            print(f"  WARN: insufficient data for {metal} ({len(sub)} rows)")
            continue

        X = sub[feat_cols].values
        y = sub['conc'].values

        # Stratification bins
        bins = np.digitize(y, BIN_EDGES) - 1

        # Stratified 5-fold CV
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        y_pred = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in skf.split(X, bins):
            # Standardize features (fit on train, transform both)
            scaler = StandardScaler().fit(X[train_idx])
            X_train = scaler.transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])

            # Train RF
            rf = RandomForestRegressor(random_state=seed, **RF_PARAMS)
            rf.fit(X_train, y[train_idx])

            # Predict out-of-fold
            y_pred[test_idx] = rf.predict(X_test)

        # Clip predictions to physically meaningful range
        y_pred = np.clip(y_pred, 0.1, None)

        # Compute metrics
        r2 = float(r2_score(y, y_pred))
        mape = float(mean_absolute_percentage_error(y, y_pred) * 100)
        rel_err = np.abs(y - y_pred) / y * 100

        # Per-band MAPE
        low_mask = y < 20
        mid_mask = (y >= 20) & (y < 100)
        high_mask = y >= 100

        results.append({
            'metal': metal,
            'R2': r2,
            'MAPE': mape,
            'MAPE_low': float(np.mean(rel_err[low_mask])) if low_mask.sum() > 0 else np.nan,
            'MAPE_mid': float(np.mean(rel_err[mid_mask])) if mid_mask.sum() > 0 else np.nan,
            'MAPE_high': float(np.mean(rel_err[high_mask])) if high_mask.sum() > 0 else np.nan,
        })

    return results


# =====================================================================
# Single configuration run
# =====================================================================

def run_one_config(data, noise_type, snr_db, seed, verbose=True):
    """
    Execute one experimental configuration end-to-end.

    Parameters
    ----------
    data : list of dict
        Pre-loaded voltammogram data
    noise_type : str
        'baseline' or one of NOISE_TYPES keys
    snr_db : float
        Target SNR (ignored for baseline)
    seed : int
        Random seed
    verbose : bool
        Print progress messages

    Returns
    -------
    rows : list of dict
        One dict per metal with full result row
    """
    if verbose:
        print(f"  Building features (noise={noise_type}, snr={snr_db}, seed={seed})...")

    t0 = time.time()
    df_features = build_feature_matrix(data, noise_type, snr_db, seed)
    if verbose:
        print(f"    -> {len(df_features)} feature rows  ({time.time()-t0:.1f}s)")

    if verbose:
        print("  Running stratified 5-fold CV with Random Forest...")
    metrics = evaluate_rf(df_features, seed)

    # Build full result rows
    result_rows = []
    for m in metrics:
        result_rows.append({
            'method': 'multiscan',
            'noise_type': noise_type,
            'snr_db': snr_db,
            'seed': seed,
            **m,
        })

    return result_rows


def append_to_csv(rows, csv_path=RESULTS_CSV):
    """Append result rows to results CSV (creates file if absent)."""
    df_out = pd.DataFrame(rows)
    try:
        existing = pd.read_csv(csv_path)
        df_out = pd.concat([existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass
    df_out.to_csv(csv_path, index=False)


# =====================================================================
# Main entry points
# =====================================================================

def main_single():
    """Command-line single-configuration mode."""
    parser = argparse.ArgumentParser(
        description="Run one synthetic noise injection experiment configuration."
    )
    parser.add_argument('noise_type',
                        choices=['baseline'] + list(NOISE_TYPES.keys()),
                        help='Noise distribution to inject')
    parser.add_argument('snr_db', type=int,
                        help='Target SNR in dB (use 0 for baseline)')
    parser.add_argument('seed', type=int,
                        help='Random seed')
    parser.add_argument('--data', default=DATA_CACHE,
                        help=f'Path to data cache (default: {DATA_CACHE})')
    parser.add_argument('--output', default=RESULTS_CSV,
                        help=f'Path to results CSV (default: {RESULTS_CSV})')
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    with open(args.data, 'rb') as f:
        data = pickle.load(f)
    print(f"  -> {len(data)} measurements loaded")
    print()

    print(f"Configuration: noise={args.noise_type}, snr={args.snr_db}, seed={args.seed}")
    rows = run_one_config(data, args.noise_type, args.snr_db, args.seed)

    print()
    print("Results:")
    for r in rows:
        print(f"  {r['metal']}: R2={r['R2']:.3f}  "
              f"MAPE={r['MAPE']:.1f}%  "
              f"mid={r['MAPE_mid']:.1f}%  high={r['MAPE_high']:.1f}%")

    append_to_csv(rows, args.output)
    print(f"\nAppended to {args.output}.")


def main_full_sweep():
    """
    Run the complete experimental grid used in the paper:
        - 1 baseline x 3 seeds  = 3 runs
        - 4 noise types x 6 SNRs x 3 seeds = 72 runs
        Total: 75 runs (~30-60 min on modern laptop)
    """
    print("Loading data...")
    with open(DATA_CACHE, 'rb') as f:
        data = pickle.load(f)
    print(f"  -> {len(data)} measurements loaded\n")

    all_rows = []

    # Clean baseline (3 seeds)
    print("=== BASELINE ===")
    for seed in SEEDS:
        print(f"[BASELINE seed={seed}]")
        rows = run_one_config(data, 'baseline', 0, seed)
        all_rows.extend(rows)
        for r in rows:
            print(f"  {r['metal']}: R2={r['R2']:.3f}  "
                  f"MAPE_mid={r['MAPE_mid']:.1f}%")

    # Full noise grid: 4 x 6 x 3 = 72
    print()
    print("=== NOISE INJECTION SWEEP ===")
    total = len(NOISE_TYPES) * len(SNR_LEVELS) * len(SEEDS)
    run_idx = 0
    for noise_type in NOISE_TYPES.keys():
        for snr_db in SNR_LEVELS:
            for seed in SEEDS:
                run_idx += 1
                print(f"[{run_idx}/{total}] {noise_type} SNR={snr_db} seed={seed}")
                rows = run_one_config(data, noise_type, snr_db, seed,
                                      verbose=False)
                all_rows.extend(rows)
                for r in rows:
                    print(f"  {r['metal']}: MAPE_mid={r['MAPE_mid']:.1f}%")

    # Save consolidated results
    append_to_csv(all_rows)
    print(f"\nAll {len(all_rows)} result rows saved to {RESULTS_CSV}.")


if __name__ == '__main__':
    # If called with command-line args -> single config mode
    # If called with no args -> run full sweep
    if len(sys.argv) > 1:
        main_single()
    else:
        print("No arguments given: running FULL SWEEP")
        print("(For single config: python3 run_noise_injection_experiment.py "
              "<noise_type> <snr_db> <seed>)\n")
        main_full_sweep()
