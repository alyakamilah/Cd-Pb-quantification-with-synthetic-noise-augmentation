"""
run_experiment.py
=================
Main experimental pipeline (Sections 3.2-3.4 of paper).

Purpose:
    Runs the SNR sensitivity analysis and the noise-augmentation experiment.
    Single configuration per call (designed to be invoked from a wrapper for
    parallel/batch execution).

Pipeline:
    1. Load cached voltammograms
    2. (Optional) Inject synthetic noise into raw signals
    3. Per-scan feature extraction (7 peak-related features)
    4. Multi-scan aggregation (mean, std, CV) -> 17-dim feature vector
    5. Random Forest regression with stratified 5-fold CV
    6. Report MAPE per concentration band + R²

Configuration (via command-line args):
    noise_type: baseline | gaussian | student_t | pink | hetero
    snr_db:     SNR in decibels (irrelevant if noise_type=baseline)
    seed:       random seed (42, 7, 123 used in paper)

Usage:
    # Clean baseline
    python3 run_experiment.py baseline 0 42

    # Gaussian noise at SNR=30 dB, seed=42
    python3 run_experiment.py gaussian 30 42

    # Pink noise at SNR=15 dB, seed=7
    python3 run_experiment.py pink 15 7

Output:
    Appends one row to results.csv with columns:
        method, noise_type, snr_db, seed, metal, R2, MAPE, MAPE_low,
        MAPE_mid, MAPE_high
"""

import sys
import time
import pickle
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew, t as student_t
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_CACHE = "../01_data_preparation/data_cache.pkl"
RESULTS_CSV = "results.csv"

# Concentration bands for stratified CV and per-band MAPE
BIN_EDGES = [0, 10, 30, 100, 500, 1100]

# Random Forest hyperparameters (Section 2.7 of paper)
RF_PARAMS = dict(n_estimators=100, max_depth=10,
                 min_samples_leaf=2, n_jobs=-1)


# ------------------------------------------------------------------
# Noise injection (Section 2.6 of paper)
# ------------------------------------------------------------------
def gaussian_sigma(y, snr_db):
    """Compute noise std for given peak-relative SNR.

    Following IUPAC: SNR_dB = 20 log10(peak_height / sigma_noise).
    Peak height = max baseline-corrected amplitude in central region.
    """
    n = len(y)
    coeffs = np.polyfit(np.arange(n), y, 2)
    detrended = y - np.polyval(coeffs, np.arange(n))
    s, e = 70, min(110, n)
    peak_height = float(np.max(np.abs(detrended[s:e]))) if e > s else 1.0
    if peak_height < 1e-3:
        peak_height = 1.0
    return peak_height / (10 ** (snr_db / 20.0))


def add_gaussian(y, snr_db, rng):
    sigma = gaussian_sigma(y, snr_db)
    return y + rng.normal(0, sigma, size=y.shape)


def add_student_t(y, snr_db, rng, df=1.5):
    sigma = gaussian_sigma(y, snr_db)
    raw = student_t.rvs(df, size=y.shape, random_state=rng)
    raw_std = np.std(raw)
    return y + (sigma / max(raw_std, 1e-3)) * raw


def add_pink(y, snr_db, rng, beta=3.0):
    """Pink (1/f^beta) noise via spectral synthesis."""
    n = len(y)
    sigma = gaussian_sigma(y, snr_db)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid 0 frequency
    spectrum = 1.0 / (freqs ** (beta / 2.0))
    phases = np.exp(2j * np.pi * rng.uniform(size=len(freqs)))
    noise = np.fft.irfft(spectrum * phases, n=n)
    noise = noise / max(np.std(noise), 1e-3) * sigma
    return y + noise


def add_heteroscedastic(y, snr_db, rng):
    """Heteroscedastic Gaussian: variance scales with local signal amplitude."""
    sigma_mean = gaussian_sigma(y, snr_db)
    weight = np.abs(y - np.median(y))
    weight = weight / max(np.mean(weight), 1e-3)
    sigma_local = sigma_mean * (0.5 + 0.5 * weight)
    return y + rng.normal(0, sigma_local, size=y.shape)


NOISE_FNS = {
    'baseline': lambda y, snr, rng: y,  # no injection
    'gaussian': add_gaussian,
    'student_t': add_student_t,
    'pink': add_pink,
    'hetero': add_heteroscedastic,
}


# ------------------------------------------------------------------
# Combined noise distributions (major revision, Reviewer 2 point R)
# ------------------------------------------------------------------
# The originals above are left untouched so previously reported results reproduce
# bit-for-bit. Combined noise sums independent components, each scaled to
# sigma/sqrt(k) so the component variances add to the single-distribution target
# variance -> the COMBINED signal keeps the same peak-relative SNR (IUPAC, Eq. 1).
def _gen_component(y, kind, sigma, rng):
    """Zero-mean noise vector of std ~ sigma for one component distribution."""
    n = len(y)
    if kind == 'gaussian':
        return rng.normal(0, sigma, size=n)
    if kind == 'student_t':
        raw = student_t.rvs(1.5, size=n, random_state=rng)
        return (sigma / max(np.std(raw), 1e-3)) * raw
    if kind == 'pink':
        freqs = np.fft.rfftfreq(n); freqs[0] = freqs[1]
        spectrum = 1.0 / (freqs ** (3.0 / 2.0))
        phases = np.exp(2j * np.pi * rng.uniform(size=len(freqs)))
        noise = np.fft.irfft(spectrum * phases, n=n)
        return noise / max(np.std(noise), 1e-3) * sigma
    if kind == 'hetero':
        weight = np.abs(y - np.median(y)); weight = weight / max(np.mean(weight), 1e-3)
        return rng.normal(0, sigma * (0.5 + 0.5 * weight), size=n)
    raise ValueError(kind)


def make_combined(kinds):
    """Factory: returns a noise function that injects the sum of the given components."""
    def _fn(y, snr_db, rng):
        sigma = gaussian_sigma(y, snr_db) / np.sqrt(len(kinds))
        noise = np.zeros_like(y, dtype=float)
        for k in kinds:
            noise = noise + _gen_component(y, k, sigma, rng)
        return y + noise
    return _fn


# Register the combined distributions reviewers asked about
NOISE_FNS.update({
    'gaussian_pink': make_combined(['gaussian', 'pink']),
    'gaussian_student_t': make_combined(['gaussian', 'student_t']),
    'gaussian_hetero': make_combined(['gaussian', 'hetero']),
    'pink_student_t': make_combined(['pink', 'student_t']),
    'gaussian_pink_student_t': make_combined(['gaussian', 'pink', 'student_t']),
})


# ------------------------------------------------------------------
# Per-scan feature extraction (Section 2.5 of paper)
# ------------------------------------------------------------------
def split_forward_reverse(E, y):
    """Split a CV scan into forward and reverse branches at the turning point."""
    dE = np.diff(E)
    sign_changes = np.where(np.diff(np.sign(dE)) != 0)[0]
    if len(sign_changes) == 0:
        return (E, y), (np.array([]), np.array([]))
    t = sign_changes[0] + 1
    return (E[:t+1], y[:t+1]), (E[t+1:], y[t+1:])


def extract_features_one_scan(E, y, peak_window):
    """Extract 7 peak-related features from one CV scan.

    Returns dict with keys: amp, pos, wid, area, skew, snr, bl
    Returns None if extraction fails.
    """
    yf = savgol_filter(y, 11, 3)

    # Quadratic baseline outside peak window
    mo = (E < peak_window[0]) | (E > peak_window[1])
    if mo.sum() >= 4:
        baseline = np.polyval(np.polyfit(E[mo], yf[mo], 2), E)
    else:
        baseline = np.full_like(yf, np.median(yf))

    # Pick branch with larger peak amplitude
    (Ef, yf_), (Er, yr) = split_forward_reverse(E, yf)
    (_, bf), (_, br) = split_forward_reverse(E, baseline)

    def amp_in_window(E_, y_):
        m = (E_ >= peak_window[0]) & (E_ <= peak_window[1])
        return np.ptp(y_[m]) if m.sum() > 0 else 0.0

    if amp_in_window(Er, yr) >= amp_in_window(Ef, yf_):
        Eu, yu, bu = Er, yr, br
    else:
        Eu, yu, bu = Ef, yf_, bf

    mask = (Eu >= peak_window[0]) & (Eu <= peak_window[1])
    if mask.sum() < 3:
        return None
    Ew, yw, bw = Eu[mask], yu[mask], bu[mask]
    yc = yw - bw  # baseline-corrected

    ip = np.argmax(np.abs(yc))
    A = float(yc[ip])

    # FWHM-derived sigma
    half = abs(A) / 2.0
    above = (yc >= half) if A > 0 else (yc <= -half)
    if above.sum() >= 2:
        idxs = np.where(above)[0]
        sigma = float(abs(Ew[idxs[-1]] - Ew[idxs[0]])) / 2.355
    else:
        sigma = 0.0

    area = float(np.trapz(np.abs(yc), Ew))
    sk = float(skew(yc)) if len(yc) > 3 and np.std(yc) > 0 else 0.0
    noise_proxy = np.std(yu) if len(yu) > 5 else 1.0
    peak_snr = abs(A) / max(float(noise_proxy), 1e-3)
    bl = float(bw[ip])

    return {
        'amp': A, 'pos': float(Ew[ip]), 'wid': sigma,
        'area': area, 'skew': sk, 'snr': peak_snr, 'bl': bl,
    }


# ------------------------------------------------------------------
# Multi-scan aggregation (Section 2.5 of paper)
# ------------------------------------------------------------------
def aggregate_multiscan(scan_features):
    """Aggregate per-scan features across scans.

    Args:
        scan_features: list of feature dicts (one per scan)

    Returns:
        dict with mean and std for all 7 features, plus CV for amp/pos/area.
        Total: 7 + 7 + 3 = 17 features.
    """
    df = pd.DataFrame(scan_features)
    agg = {}
    for col in df.columns:
        agg[f'{col}_mean'] = float(np.mean(df[col].values))
        agg[f'{col}_std'] = float(np.std(df[col].values))
    for col in ['amp', 'pos', 'area']:
        m = abs(agg[f'{col}_mean'])
        agg[f'{col}_cv'] = (agg[f'{col}_std'] / (m + 1e-6)) if m > 0 else 0.0
    return agg


# ------------------------------------------------------------------
# Build feature matrix for one (noise_type, snr, seed) configuration
# ------------------------------------------------------------------
def build_features(data, noise_fn, snr_db, seed):
    """Apply noise injection + feature extraction across all measurements.

    Returns:
        DataFrame with columns: metal, conc, file, + 17 feature columns
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in data:
        scan_feats = []
        for E, y in d['scans']:
            y_n = noise_fn(y, snr_db, rng)
            f = extract_features_one_scan(E, y_n, d['pw'])
            if f is not None:
                scan_feats.append(f)
        if len(scan_feats) < 3:
            continue
        agg = aggregate_multiscan(scan_feats)
        rows.append({'metal': d['metal'], 'conc': d['conc'],
                     'file': d['file'], **agg})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Random Forest evaluation with stratified CV
# ------------------------------------------------------------------
def evaluate_rf(df_features, seed):
    """Train RF with stratified 5-fold CV; return per-band metrics per metal."""
    feat_cols = [c for c in df_features.columns
                 if c not in ('metal', 'conc', 'file')]
    results = []

    for metal in ['Cd', 'Pb']:
        sub = df_features[df_features['metal'] == metal].copy()
        if len(sub) < 30:
            continue
        X = sub[feat_cols].values
        y = sub['conc'].values
        bins = np.digitize(y, BIN_EDGES) - 1

        # Stratified 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        y_pred = np.zeros_like(y, dtype=float)
        for tr, te in skf.split(X, bins):
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr])
            Xte = sc.transform(X[te])
            rf = RandomForestRegressor(random_state=seed, **RF_PARAMS)
            rf.fit(Xtr, y[tr])
            y_pred[te] = rf.predict(Xte)
        y_pred = np.clip(y_pred, 0.1, None)

        # Metrics
        r2 = r2_score(y, y_pred)
        mape = mean_absolute_percentage_error(y, y_pred) * 100
        rel_err = np.abs(y - y_pred) / y * 100
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


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def main():
    if len(sys.argv) != 4:
        print("Usage: python3 run_experiment.py <noise_type> <snr_db> <seed>")
        print("  noise_type: baseline | gaussian | student_t | pink | hetero")
        print("              | gaussian_pink | gaussian_student_t | gaussian_hetero")
        print("              | pink_student_t | gaussian_pink_student_t   (combined)")
        print("  snr_db:     integer (e.g., 5, 10, 15, 20, 25, 30)")
        print("  seed:       integer (e.g., 42, 7, 123)")
        sys.exit(1)

    noise_type = sys.argv[1]
    snr_db = int(sys.argv[2])
    seed = int(sys.argv[3])

    if noise_type not in NOISE_FNS:
        print(f"ERROR: unknown noise_type '{noise_type}'")
        sys.exit(1)

    print(f"Loading data...")
    with open(DATA_CACHE, 'rb') as f:
        data = pickle.load(f)
    print(f"  {len(data)} files loaded.")

    print(f"Building features (noise={noise_type}, snr={snr_db}, seed={seed})...")
    t0 = time.time()
    df = build_features(data, NOISE_FNS[noise_type], snr_db, seed)
    print(f"  {len(df)} feature rows ({time.time()-t0:.1f}s)")

    print("Evaluating RF with stratified 5-fold CV...")
    metrics = evaluate_rf(df, seed)

    # Append to results CSV
    rows = []
    for m in metrics:
        rows.append({
            'method': 'multiscan',
            'noise_type': noise_type,
            'snr_db': snr_db,
            'seed': seed,
            **m,
        })
    df_out = pd.DataFrame(rows)
    try:
        existing = pd.read_csv(RESULTS_CSV)
        df_out = pd.concat([existing, df_out], ignore_index=True)
    except FileNotFoundError:
        pass
    df_out.to_csv(RESULTS_CSV, index=False)

    print("\nResults:")
    for r in metrics:
        print(f"  {r['metal']}: R2={r['R2']:.3f}  MAPE={r['MAPE']:.1f}%  "
              f"mid={r['MAPE_mid']:.1f}%  high={r['MAPE_high']:.1f}%")
    print(f"\nAppended to {RESULTS_CSV}.")


if __name__ == '__main__':
    main()
