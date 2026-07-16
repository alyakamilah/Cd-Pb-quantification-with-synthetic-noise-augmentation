# Synthetic Noise Injection Asymmetrically Affects Machine-Learning Quantification of Cd²⁺ and Pb²⁺ in Multi-Scan Cyclic Voltammetry

This repository contains the official source code implementation for the paper:  
**"Synthetic noise injection asymmetrically affects machine-learning quantification of Cd²⁺ and Pb²⁺ in multi-scan cyclic voltammetry"**

### Authors
Alya Kamilah, Riyanarto Sarno, Kelly Rossa Sungkono, Rizqy Ahsana Putri, Taufiq Choirul Amri, Fadlilatul Taufany, Arif Abdullah Sagran, and Sang-Seok Lee

---

## 📌 Overview

This repository is dedicated to providing the source code for reproducibility and transparency of the proposed **synthetic noise injection** data-augmentation methodology for deep chemometric analysis. The pipeline couples a multi-scan feature extraction stage with a Random Forest regressor and systematically evaluates four synthetic noise distributions (Gaussian, Student-t, pink, and heteroscedastic) across a range of signal-to-noise ratios, characterizing an asymmetric augmentation effect on the quantification of heavy metals ($\text{Cd}^{2+}$ and $\text{Pb}^{2+}$) using multi-scan cyclic voltammetry data.

For convenience and keeping the workspace tidy during the review/publication process, the entire codebase has been packaged into a single archive file.

---

## 📂 Repository Contents

*   **`CEJA_complete_code.rar`**: The core package containing:
    *   Data preparation and multi-scan feature extraction scripts (raw `.xlsx` to cached dataset).
    *   Synthetic noise injection utilities and the full experimental sweep (Random Forest, XGBoost, SVR, and MLP baselines).
    *   Statistical analysis, results aggregation, and the complete figure-generation pipeline that reproduces every figure in the paper.

---

## ✉️ Contact & Data Requests

The datasets used in this study and any supplementary material are available upon reasonable request. If you have any questions regarding the code implementation, hyperparameters, or require access to the data for replication, please feel free to contact:

*   **Email:** [alyakamilah36@gmail.com](mailto:alyakamilah36@gmail.com)

---

## 📝 Citation

If you find this code useful for your research, please cite our journal https://doi.org/10.1016/j.ceja.2026.101356
