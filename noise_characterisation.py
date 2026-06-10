"""
noise_characterisation.py
=========================
Empirical characterisation of CV measurement noise (Section 3.1 of paper).

Purpose:
    Characterises the noise structure of replicate cyclic voltammograms:
    - Distribution shape: excess kurtosis, Gaussian vs Student-t QQ comparison,
      maximum-likelihood Student-t df fit
    - Spectral content: Welch power spectral density of detrended residuals,
      fit to 1/f^beta power-law
    - Heteroscedasticity: linear regression of within-replicate noise std
      against concentration in log-log space

Outputs:
    noise_summary.csv        - per-(metal, ppm) noise statistics
    noise_psd_data.npz       - PSD curves for Figure 5 (PSD)
    noise_residuals.npz      - residual histograms for Figure 7 (distribution)
    noise_heteroscedastic.csv - data for Figure 6 (heteroscedastic scaling)

Usage:
    python3 noise_characterisation.py
"""

import pickle
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, t as student_t


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_CACHE = "../01_data_preparation/data_cache.pkl"
SAMPLE_RATE_HZ = 5.0  # nominal: 0.1 V/s scan rate, 0.02 V step => 5 Hz


def load_data():
    with open(DATA_CACHE, 'rb') as f:
        return pickle.load(f)


def detrend_quadratic(y):
    """Remove quadratic baseline from a 1D signal."""
    n = len(y)
    x = np.arange(n)
    coeffs = np.polyfit(x, y, 2)
    return y - np.polyval(coeffs, x)


def replicate_residuals(samples_for_one_concentration):
    """Compute residual signals from replicate scans of identical (metal, ppm).

    Strategy: for each measurement, treat the across-replicate mean as
    the 'true' signal, and the deviation per-replicate as residual.

    Args:
        samples_for_one_concentration: list of measurement dicts, all with
            same (metal, ppm)

    Returns:
        residuals: 1D numpy array of pooled residual values
    """
    # Stack all scans of all replicates; align to common length
    all_y = []
    for d in samples_for_one_concentration:
        for E, y in d['scans']:
            y_dt = detrend_quadratic(y)
            all_y.append(y_dt)
    if len(all_y) < 3:
        return np.array([])
    Y = np.array(all_y)
    mean_signal = np.mean(Y, axis=0)
    residuals = (Y - mean_signal[None, :]).flatten()
    return residuals


def fit_psd(residuals_2d, fs=SAMPLE_RATE_HZ):
    """Fit 1/f^beta power-law to PSD.

    Args:
        residuals_2d: shape (n_replicates, n_points) per-scan residuals
        fs: sampling frequency in Hz

    Returns:
        freqs, psd, beta_estimate
    """
    # Welch PSD per scan, average across scans
    psds = []
    freqs = None
    for row in residuals_2d:
        f, p = welch(row, fs=fs, nperseg=min(64, len(row)))
        if freqs is None:
            freqs = f
        psds.append(p)
    psd = np.mean(psds, axis=0)

    # Fit 1/f^beta in log-log, exclude DC
    mask = (freqs > 0) & (psd > 0)
    if mask.sum() < 5:
        return freqs, psd, np.nan
    log_f = np.log(freqs[mask])
    log_p = np.log(psd[mask])
    slope, _ = np.polyfit(log_f, log_p, 1)
    beta = -slope
    return freqs, psd, beta


def fit_student_t(residuals):
    """ML fit Student's-t to residuals; return df."""
    if len(residuals) < 100:
        return np.nan
    try:
        df, loc, scale = student_t.fit(residuals)
        return df
    except Exception:
        return np.nan


def characterise(data):
    """Run full noise characterisation. Returns summary DataFrame."""
    rows = []
    psd_data = {}
    residual_data = {}
    hetero_data = []

    # Group by (metal, conc)
    by_group = {}
    for d in data:
        key = (d['metal'], d['conc'])
        by_group.setdefault(key, []).append(d)

    for (metal, conc), group in sorted(by_group.items()):
        # Build pooled residuals
        all_per_scan = []
        for d in group:
            for E, y in d['scans']:
                all_per_scan.append(detrend_quadratic(y))
        if len(all_per_scan) < 6:
            continue
        Y = np.array(all_per_scan)
        mean_sig = np.mean(Y, axis=0)
        residuals_2d = Y - mean_sig[None, :]
        flat_res = residuals_2d.flatten()

        # Distribution stats
        rms = float(np.std(flat_res))
        kurt = float(kurtosis(flat_res, fisher=True))
        df_t = fit_student_t(flat_res[::5])  # subsample for speed

        # PSD
        freqs, psd, beta = fit_psd(residuals_2d)

        rows.append({
            'metal': metal,
            'ppm': conc,
            'n_scans': Y.shape[0],
            'noise_rms': rms,
            'excess_kurtosis': kurt,
            'student_t_df': df_t,
            'psd_beta': beta,
        })

        hetero_data.append({'metal': metal, 'ppm': conc, 'noise_rms': rms})

        # Cache for figures
        if conc in (10, 100):
            psd_data[f'{metal}_{int(conc)}ppm'] = {
                'freqs': freqs, 'psd': psd, 'beta': beta,
            }
            residual_data[f'{metal}_{int(conc)}ppm'] = flat_res[:5000]

    df = pd.DataFrame(rows)

    return df, psd_data, residual_data, pd.DataFrame(hetero_data)


def main():
    print("Loading dataset...")
    data = load_data()
    print(f"  {len(data)} files loaded.\n")

    print("Computing noise statistics...")
    summary, psd_data, residual_data, hetero_df = characterise(data)

    summary.to_csv("noise_summary.csv", index=False)
    print(f"\nSaved noise_summary.csv ({len(summary)} groups)")
    print(summary.to_string(index=False))

    np.savez("noise_psd_data.npz", **{
        k: np.array([v['freqs'], v['psd']])
        for k, v in psd_data.items()
    })
    np.savez("noise_residuals.npz", **residual_data)
    hetero_df.to_csv("noise_heteroscedastic.csv", index=False)
    print("\nSaved PSD/residual/heteroscedastic data.")

    # Heteroscedastic fit log-log
    print("\nHeteroscedastic scaling exponent (noise_rms ~ C^alpha):")
    for metal in ['Cd', 'Pb']:
        sub = hetero_df[hetero_df['metal'] == metal].copy()
        sub = sub[sub['ppm'] > 0]
        log_c = np.log10(sub['ppm'].values)
        log_n = np.log10(sub['noise_rms'].values)
        if len(sub) >= 3:
            slope, intercept = np.polyfit(log_c, log_n, 1)
            print(f"  {metal}: alpha = {slope:.2f}")


if __name__ == '__main__':
    main()
