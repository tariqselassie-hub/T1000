# Kaggle Titanic — Machine Learning from Disaster
## Sovereign Contest Architecture, Classical Baselines & Variational Quantum Circuit Exploration

SPDX-License-Identifier: MIT OR Apache-2.0  
Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

---

### Executive Overview

This directory contains the contest data, exploratory data analysis, graph walks, classical ensemble models, and Variational Quantum Circuit (VQC) pipelines for the **Kaggle Titanic: Machine Learning from Disaster** competition.

All raw data, processed folds, predictions, and submission files are isolated in the dedicated [`data/`](data/) directory to preserve clean separation from the root `T1000` simulation engine.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              TITANIC CONTEST TAXONOMY                                  │
├─────────────────┬─────────────────────────────┬──────────────────────────┬─────────────┤
│ Model Family    │ Target Architecture         │ Features / Inputs        │ CV Accuracy │
├─────────────────┼─────────────────────────────┼──────────────────────────┼─────────────┤
│ Baseline        │ Gender Model (Heuristic)    │ Sex only                 │ 0.787       │
│ Classical       │ HistGradientBoosting + Reg  │ Passenger columns        │ 0.828       │
│ Group Augmented │ Gradient Boosting + Group   │ Ticket / Surname leak-free│ 0.840       │
│ Ensemble        │ LogReg + RF + HGB (Blended) │ Group + Role-Split Group │ 0.847       │
│ Quantum VQC     │ 8-Qubit Ry/Rz Ring Ansatz   │ 8 Scaled Angular Feats   │ 0.820       │
└─────────────────┴─────────────────────────────┴──────────────────────────┴─────────────┘
```

---

### Directory Architecture

```
titanic-contest/
├── README.md               # Master specification & tournament overview
├── REPORT.md               # Empirical findings, ablation analysis & residual search
├── .gitignore              # Ignores large CSVs, fold exports, and logs
├── data/                   # Dedicated contest dataset directory
│   ├── train.csv           # 891 labeled training passengers
│   ├── test.csv            # 418 unlabeled evaluation passengers
│   ├── gender_submission.csv # Baseline benchmark submission
│   ├── kernel_input.csv    # Label-free features for quantum kernel benchmarks
│   ├── submission.csv      # Generated competition submission (0.847 CV ensemble)
│   ├── vqc_folds/          # Exported cross-validation fold CSVs for quantum training
│   └── vqc_final/          # Final 25-fold evaluation splits for ensemble blending
├── analysis.py             # Repeated stratified cross-validation of classical baselines
├── angles.py               # Feature angle mapping and spherical coordinate projections
├── fares.py                # BIC ticket price clustering and fare-tier analysis
├── features.py             # Feature engineering and dataset ingestion
├── graph_walk.py           # Passenger relationship graph construction and diffusion
├── graph_walk_alone.py     # Graph-only baseline prediction
├── improve.py              # Candidate feature screening and residual correlation search
├── kernel_input.py         # Generation of clean feature matrices for quantum kernels
├── quantum_member.py       # Blending VQC predictions with classical ensemble
├── relations.py            # Family and ticket role structures (women/children vs men)
├── spooky.py               # Search for non-local correlations across cabin/ticket digits
├── submit.py               # Generates final submission.csv with role-split group features
├── vqc_compare.py          # Direct comparison of VQC vs classical models on identical folds
└── vqc_export.py           # Exports fold CSVs scaled to [0, pi] for quantum angle encoding
```

---

### Reproducing Results

1. **Feature Extraction & Classical Baselines:**
   ```bash
   python titanic-contest/analysis.py
   ```

2. **Generate Competition Submission:**
   ```bash
   python titanic-contest/submit.py
   ```

3. **Export Quantum Cross-Validation Folds:**
   ```bash
   python titanic-contest/vqc_export.py data/vqc_folds 2026 10
   ```

4. **Train Variational Quantum Circuit:**
   ```bash
   cargo run --release --bin titanic_vqc -- --dir titanic-contest/data/vqc_folds --qubits 8 --depth 3 --epochs 30 --lr 0.05
   ```

5. **Benchmark VQC vs Classical Models:**
   ```bash
   python titanic-contest/vqc_compare.py titanic-contest/data/vqc_folds
   ```

