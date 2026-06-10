# RUNBOOK REVISI — CEJA Major Revision

Paket kode lengkap untuk menjawab seluruh poin reviewer, **opsi kuat dipilih** untuk
setiap poin. Semua skrip baru memakai-ulang pipeline asli (`run_experiment.py`)
lewat `import`, jadi tidak ada reimplementasi dan hasil lama tetap reproducible.

> Bahasa runbook ini: Indonesia. Komentar di dalam kode: Inggris (mengikuti paket asli).

---

## 0. SETUP (sekali saja)

```bash
cd CEJA_complete_code
pip install -r requirements.txt          # numpy, pandas, scipy, scikit-learn, xgboost, matplotlib, shap
```

`shap` opsional: kalau gagal di-install, `feature_importance.py` tetap jalan dan
hanya melewati kolom SHAP (permutation + RF built-in tetap dihitung).

### Build dataset → `data_cache.pkl`

Edit **satu baris** di `01_data_preparation/build_dataset.py`:

```python
DATA_ROOT = "/path/ke/DATA 2 KELAS"      # folder yang berisi CADMIUM/ dan TIMBAL/
```

Lalu:

```bash
cd 01_data_preparation
python3 build_dataset.py                 # ~2 menit, menghasilkan data_cache.pkl (~50 MB)
cd ..
```

Struktur data sudah cocok dengan punyamu: `CADMIUM/10 ppm/Cad_10ppm_01.xlsx`,
`TIMBAL/10PPM/Pb_...xlsx`, header baris-1, kolom E/i berselang-seling, 10 scan
(scan-1 dibuang → 9 dipakai), 180 titik/scan, peak window Cd `(-0.95,-0.75)` V dan
Pb `(-0.50,-0.30)` V.

---

## 1. EKSPERIMEN BERBASIS KODE (jalankan di folder masing-masing)

Urutan yang disarankan: **A → E/F/R → B → C → D**. Total ± 1.5–3 jam di laptop 8-core.

### (E + F + R) Sweep 20-seed + kombinasi noise + statistik eksak — INTI

> Reviewer 1 #5 (lebih banyak seed), Reviewer 2 (p-value & CI lengkap), Reviewer 2 (kombinasi noise)

```bash
cd 02_experiments
chmod +x run_revision_sweep.sh
./run_revision_sweep.sh
```

Apa yang dijalankan:
- baseline + 4 noise tunggal (gaussian, student_t, pink, hetero) **@ SNR=30, 20 seed**
- 5 **kombinasi noise** (gaussian+pink, gaussian+student_t, gaussian+hetero,
  pink+student_t, gaussian+pink+student_t) **@ SNR=30, 20 seed**
- lalu otomatis memanggil `statistical_analysis.py`

Output:
- `02_experiments/results.csv` — semua hasil per (noise, snr, seed, metal)
- `02_experiments/augmentation_stats.csv` — **per perbandingan**: mean baseline & noise
  + 95% CI, **delta_pct + 95% CI**, **p-value Wilcoxon eksak**, p-value paired-t,
  effect size (Cohen dz, rank-biserial)

→ **Manuskrip**: ganti Tabel 1 dengan kolom `delta_pct [CI]`, `p_wilcoxon_exact`,
`cohen_dz`. Baris kombinasi noise → subbagian/Tabel baru di Hasil (bukan lagi sekadar
limitation). CI 20-seed → kalimat metode "20 random seeds; mean ± 95% CI".

Catatan desain: SNR *line plot* (Fig 8/9) boleh tetap 3 seed; CI ketat hanya perlu di
SNR=30. Kalau mau 20 seed untuk SEMUA SNR juga, tambah loop SNR di skrip (jauh lebih lama).

### (A) Studi simulasi — voltammogram sintetik

> Reviewer 1 #1 (controlled simulation study)

Folder terpisah `sim/` (lihat README di dalamnya). Sudah dikonfirmasi mereproduksi:
asimetri Cd vs Pb + disosiasi R²/MAPE hanya dengan memvariasikan bentuk peak. Jalankan:

```bash
cd sim
python3 compute.py anchors Cd_like 0 8     # 2 anchor × 8 seed (sudah ada di results/)
python3 compute.py anchors Pb_like 0 8
python3 compute.py sweeps width            # sweep lebar peak
python3 compute.py sweeps bumpiness        # sweep degenerasi apex
python3 compute.py sweeps overlap          # sweep overlap antar-peak
```

→ **Manuskrip**: subbagian baru "Mechanistic validation on synthetic voltammograms"
+ figur anchors (bar) & sweeps (garis ΔMAPE vs faktor bentuk).

### (B) Feature importance: sebelum vs sesudah augmentation

> Reviewer 1 #10 (buktikan SD & CV jadi lebih informatif)

```bash
cd 03_analysis
python3 feature_importance.py 20 gaussian 30
```

Menghitung **tiga** ukuran (Permutation, RF built-in, SHAP) pada split train/test
(scaler fit di train saja), lintas 20 seed, lalu mengelompokkan importance ke
`mean / std / cv` dan melaporkan **share** tiap grup + 95% CI.

Output:
- `feature_importance.csv` — per (metal, kondisi, seed, fitur) untuk ketiga ukuran
- `feature_importance_grouped.csv` — share per grup (mean/std/cv) + 95% CI

→ **Manuskrip**: tabel/figur "importance share mean/std/cv, clean vs augmented".
Klaim yang didukung: **share std+cv NAIK** setelah augmentation (di subset uji pun sudah
terlihat: Cd 0.06→0.20, Pb 0.03→0.20; jalankan penuh untuk angka final).

### (C) Hyperparameter tuning (GridSearchCV, nested-CV)

> Reviewer 1 #4 (kenapa tidak ada tuning?) — OPSI KUAT

```bash
cd 02_experiments
python3 hyperparameter_tuning.py 10
# Mesin lambat? kecilkan grid dulu untuk cek: HP_FAST=1 python3 hyperparameter_tuning.py 2
# Default n_jobs=-1; kalau spawning lambat: HP_NJOBS=1 python3 hyperparameter_tuning.py 10
```

GridSearch (`n_estimators × max_depth × min_samples_split × min_samples_leaf`) di dalam
**nested-CV** (inner memilih param di train; outer mengukur skor tuned di test) → angka
tuned tak bias **dan** memperkuat argumen anti-bocor.

Output:
- `hyperparameter_tuning.csv` — default vs tuned MAPE_mid per (metal, kondisi, seed)
- `hyperparameter_sensitivity.csv` — sensitivity satu-per-satu (n_estimators / max_depth
  / min_samples_leaf) lewat `evaluate_rf` paper

→ **Manuskrip**: kalimat "tuning hanya mengubah MAPE_mid ±X% → setelan paper sudah robust",
+ pernyataan "efek asimetris tetap muncul di bawah model ter-tuning", + (opsional) tabel
sensitivity di Supplementary.

### (D) Bukti tidak ada information leakage

> Reviewer 1 #6

```bash
cd 03_analysis
python3 leakage_safe_augmentation.py 20
```

Membandingkan 3 protokol pada CV yang ketat-anti-bocor (split level-file, scaler &
RF fit di train fold saja):
- **A** clean→clean (baseline)
- **B** noisy→noisy (protokol paper)
- **C** noisy(train)→clean(test) (augmentasi train-only ketat)

Output: `leakage_safe_augmentation.csv`, `leakage_safe_stats.csv` (Wilcoxon eksak B-vs-A, C-vs-A).

→ **Manuskrip** (poin penting & jujur): urutannya **split → (noise per-file) → scaler fit
train → transform test → RF fit train**. Tidak ada duplikasi augmentasi (1 file = 1 vektor
fitur, jadi salinan sampel train tak mungkin bocor ke test). Hasil protokol C biasanya
**tidak** memberi benefit karena test bersih (std/CV≈0) → mismatch distribusi; ini justru
menunjukkan benefit paper berasal dari **akuisisi multi-scan ber-noise** (protokol B),
bukan augmentasi train-only klasik. Selaras dengan reframing Reviewer N (poin di bawah).

---

## 2. POIN TULISAN (tanpa kode) — ringkasan arahan

### (M) Literatur baru di Introduction — WAJIB
Masukkan & bahas gap dari 3 DOI yang diminta reviewer:
- `10.1016/j.hazadv.2024.100532`
- `10.1016/j.psep.2025.107671`
- `10.1016/j.jenvman.2023.119968`

Framing gap: studi lain fokus ke model ML / ekstraksi fitur / voltammetri, **belum ada**
controlled noise augmentation + efek asimetris Cd vs Pb → ini novelty utama.

### (N) Reframe judul/abstrak/intro
Dari "multi-scan aggregation" → **"controlled synthetic noise augmentation in multi-scan
cyclic voltammetry"**. Hasil protokol D-C di atas mendukung: efek = properti akuisisi
ber-noise, bukan sekadar agregasi.

### (H) Mengapa Cd ≠ Pb (mekanistik)
Hubungkan dengan morfologi sinyal: peak Cd lebih lebar/asimetris/berpotensi overlap →
peak-finding tak stabil → std/CV antar-scan jadi informatif setelah noise; peak Pb sempit &
bersih → tidak ada ambiguitas untuk diregularisasi. **Dukung dengan hasil simulasi (A) dan
feature importance (B).**

### (I) Kenapa MAPE > R²
R² bisa tinggi sementara error relatif konsentrasi tetap besar; untuk sensor logam berat
yang penting error relatif aktual → MAPE. Contoh: R² ~sama, MAPE 20%→10% lebih berguna
secara praktis. (Disosiasi ini persis yang ditunjukkan simulasi A.)

### (J) Batas deteksi 15–20 ppm → limitation
Sistem optimal di **mid-range**; belum untuk trace. Untuk trace perlu DPV/SWV atau
preconcentration. Tulis sebagai limitation, bukan klaim berlebih.

### (K) SNR threshold → implikasi praktis
Di bawah SNR tertentu model mulai gagal → bisa dipakai sebagai **kriteria quality-control**
akuisisi sensor. Ambil angka ambang dari `results.csv` (Fig 8/9).

### (L) Implikasi desain sensor
Noise moderat tidak selalu buruk → preprocessing tak perlu terlalu agresif; protokol
akuisisi boleh mempertahankan variabilitas alami; augmentation berguna saat dataset kecil.

### (G) Diskusi keterbatasan noise sintetik
Gaussian belum merepresentasikan drift instrumental/temporal correlation; pink noise
sebagian; future work: noise instrumen riil. (Sekarang kombinasi noise sudah diuji, jadi
itu pindah dari limitation ke hasil — lihat R.)

### (O) Tabel alasan 17 fitur
Tabel: fitur → makna elektrokimia (amp=magnitudo, wid=FWHM, area=muatan, skew=asimetri,
snr, bl=baseline; lalu mean/std/cv = tendensi/dispersi/dispersi relatif).

### (P) Detail formula noise (Metode)
Tulis formula + parameter (sudah ada di kode):
- Gaussian `N(0,σ²)`; Student-t `t(ν=1.5)`; Pink `1/f^β, β=3`; Heteroscedastic
  `σ(x)=σ̄·(0.5+0.5·|y−median|/mean|·|)`. SNR (Eq.1) `SNR_dB=20·log10(peak_height/σ)`.
- Kombinasi: jumlah komponen independen, tiap komponen diskalakan `σ/√k` agar SNR target
  terjaga (lihat `make_combined` di `run_experiment.py`).

### (Q) Definisi mid-range
Nyatakan eksplisit: **mid-range = 20–100 ppm** (sesuai band MAPE_mid & bin CV di kode:
`BIN_EDGES=[0,10,30,100,500,1100]`, MAPE_mid pada `20≤c<100`).

### (S) Repository + Zenodo — REPRODUCIBILITY
1. Push folder ini ke GitHub (kode + RUNBOOK + README).
2. Hubungkan repo ke **Zenodo** → buat rilis → dapat **DOI**.
3. Sertakan dataset (atau tautan "available on request" sesuai kebijakan).
4. Update *Data/Code Availability statement* di manuskrip dengan DOI Zenodo + URL GitHub.

### (Conclusion)
Tulis ulang (bukan ringkasan hasil): temuan utama (asimetri Cd/Pb) → mekanisme dugaan
(divalidasi simulasi) → implikasi praktis (akuisisi & QC) → keterbatasan (trace, noise
sintetik) → future work (noise instrumen riil, DPV/SWV).

---

## 3. FILE YANG BERUBAH / BARU

```
01_data_preparation/build_dataset.py          (tak berubah; edit DATA_ROOT saja)
02_experiments/run_experiment.py              MODIF in-place: + 5 kombinasi noise (R)
02_experiments/hyperparameter_tuning.py       BARU  (C)
02_experiments/run_revision_sweep.sh          BARU  (E,F,R)
03_analysis/statistical_analysis.py           UPGRADE in-place: paired Wilcoxon+CI+effect (F)
03_analysis/statistical_analysis_ORIGINAL.py.bak   cadangan versi lama (boleh dihapus)
03_analysis/feature_importance.py             BARU  (B)
03_analysis/leakage_safe_augmentation.py      BARU  (D)
requirements.txt                              + shap (opsional)
```

Skrip lama (`run_full_sweep.sh`, `make_figures.py`, dll.) tetap utuh & berfungsi.

---

## 4. CATATAN PENTING

- **Jangan mengarang angka.** Laporkan hanya angka dari run penuh di mesinmu.
- Smoke-test di subset kecil sudah memastikan semua skrip **jalan tanpa error**; angka
  final harus dari `data_cache.pkl` lengkap (1500+ file).
- Reproducibility: seed eksplisit (numpy `default_rng`, sklearn `random_state`). Seed +
  versi library sama → hasil identik.
- Kalau `shap` bikin masalah versi (mis. numpy terlalu baru): biarkan saja, skrip otomatis
  skip SHAP; permutation + RF built-in sudah cukup untuk klaim utama.
