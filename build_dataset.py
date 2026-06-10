"""
build_dataset.py
================
Prepares the voltammogram dataset for downstream experiments.

Purpose:
    Reads raw cyclic voltammetry .xlsx files from the experimental campaign,
    extracts the 9 non-conditioning scans per measurement, and serializes
    them into a single pickle file for fast loading by experiment scripts.

Inputs (expected directory structure):
    DATA_ROOT/
        CADMIUM/
            2 ppm/
                cad_2ppm_01.xlsx
                cad_2ppm_02.xlsx
                ...
            4 ppm/
                ...
            ...
            1000 ppm/
        TIMBAL/
            2PPM/
                pb_2ppm_01.xlsx
                ...
            ...

Each .xlsx file has 10 consecutive CV scans, with E (potential, V) and
i (current, μA) columns alternating. Scan-1 is excluded (conditioning).

Output:
    data_cache.pkl  --  list of dicts:
        {
            'metal': 'Cd' | 'Pb',
            'conc': float,           # concentration in ppm
            'pw':   (E_min, E_max),  # analytical peak window in V
            'file': 'cad_2ppm_01.xlsx',
            'scans': [(E_arr, y_arr), ...]  # 9 (E, current) tuples
        }

Usage:
    Edit DATA_ROOT below to point to your raw data directory, then:
        python3 build_dataset.py

    The output `data_cache.pkl` is saved next to this script.

Notes:
    - Peak windows are: Cd at -0.95 to -0.75 V, Pb at -0.50 to -0.30 V
      (vs Ag/AgCl reference electrode in saturated KCl).
    - Each scan is truncated/padded to exactly 180 data points.
    - Files with fewer than 9 valid scans are skipped.
"""

import os
import glob
import time
import pickle
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Configuration -- EDIT THIS PATH to point to your raw data
# ------------------------------------------------------------------
DATA_ROOT = r"C:\Users\alyak\Documents\S2\S2 (SEMESTER 2)\BISMILLAH THESIS\DATA 2 KELAS"  # contains CADMIUM/ and TIMBAL/ subdirs
OUTPUT_PATH = "data_cache.pkl"

# Concentration directory names per metal (case-sensitive on Linux)
CD_CONCS = ['2 ppm', '4 ppm', '6 ppm', '8 ppm', '10 ppm', '20 ppm', '40 ppm',
            '60 ppm', '80 ppm', '100 ppm', '200 ppm', '400 ppm', '600 ppm',
            '800 ppm', '1000 ppm']
PB_CONCS = ['2PPM', '4PPM', '6PPM', '8PPM', '10PPM', '20PPM', '40PPM', '60PPM',
            '80PPM', '100PPM', '200PPM', '400PPM', '600PPM', '800PPM', '1000PPM']
CONC_VALUES = [2, 4, 6, 8, 10, 20, 40, 60, 80, 100, 200, 400, 600, 800, 1000]

# Analytical peak windows (V vs Ag/AgCl)
PEAK_WINDOW = {
    'Cd': (-0.95, -0.75),
    'Pb': (-0.50, -0.30),
}

# Acquisition constants
N_POINTS_PER_SCAN = 180
N_SCANS_PER_FILE = 10
N_USED_SCANS = 9  # scan-1 excluded as conditioning


def load_one_file(filepath, n_used_scans=N_USED_SCANS,
                  n_points=N_POINTS_PER_SCAN):
    """Read one .xlsx file and return list of (E, y) tuples for scans 2..10.

    Each .xlsx has columns alternating: [E_scan1, i_scan1, E_scan2, i_scan2, ...]
    with header on row 1.

    Returns:
        list of (E_array, y_array), each length=n_points; or [] if invalid.
    """
    try:
        df = pd.read_excel(filepath, header=1)
    except Exception:
        return []

    scans = []
    # Skip scan-1 (idx 0); use scans 2..10 (idx 1..9)
    for s_idx in range(1, n_used_scans + 1):
        col_E = 2 * s_idx
        col_i = 2 * s_idx + 1
        if col_i >= df.shape[1]:
            break
        E = pd.to_numeric(df.iloc[:, col_E], errors='coerce').values
        y = pd.to_numeric(df.iloc[:, col_i], errors='coerce').values
        valid = ~(np.isnan(E) | np.isnan(y))
        if valid.sum() < n_points:
            continue
        E = E[valid][:n_points]
        y = y[valid][:n_points]
        if len(E) == n_points:
            scans.append((E.astype(np.float32), y.astype(np.float32)))
    return scans


def build_dataset():
    """Walk dataset directories and return list of measurement dicts."""
    rows = []
    t0 = time.time()

    for metal, conc_dirs, subdir in [
        ('Cd', CD_CONCS, 'CADMIUM'),
        ('Pb', PB_CONCS, 'TIMBAL'),
    ]:
        root = os.path.join(DATA_ROOT, subdir)
        pw = PEAK_WINDOW[metal]

        for conc_dir, conc_val in zip(conc_dirs, CONC_VALUES):
            conc_path = os.path.join(root, conc_dir)
            if not os.path.isdir(conc_path):
                print(f"  WARN: missing dir {conc_path}")
                continue

            files = sorted(glob.glob(os.path.join(conc_path, '*.xlsx')))
            for fp in files:
                scans = load_one_file(fp)
                if len(scans) < 3:  # need at least 3 valid scans
                    continue
                rows.append({
                    'metal': metal,
                    'conc': float(conc_val),
                    'pw': pw,
                    'file': os.path.basename(fp),
                    'scans': scans,
                })
            print(f"  {metal} {conc_dir}: "
                  f"{len([r for r in rows if r['metal']==metal and r['conc']==conc_val])} files "
                  f"({time.time()-t0:.0f}s)")

    return rows


def main():
    print(f"Building dataset from {DATA_ROOT}...")
    rows = build_dataset()
    print(f"\nTotal: {len(rows)} measurement files loaded.")
    if rows:
        from collections import Counter
        c = Counter((r['metal'], int(r['conc'])) for r in rows)
        print("\nFiles per (metal, ppm):")
        for (m, ppm), n in sorted(c.items()):
            print(f"  {m} {ppm:>5} ppm: {n} files")

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(rows, f)
    print(f"\nSaved {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")


if __name__ == '__main__':
    main()
