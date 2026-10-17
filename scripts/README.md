# T-1000 Operational Scripts Directory

This directory contains autonomous tooling, empirical benchmarking drivers, data ingestion pipelines, hardware co-processor bridges, and sovereign visualization suites.

---

## Directory Organization

```
scripts/
├── charting/       # 16 Sovereign research charts & master suite (chart_all.py)
├── data/           # High-throughput dataset feature extractors (MobileNetV3 PCA)
├── fpga/           # Artix-7 XC7A35T hardware UART drivers & chaos verification
├── qec/            # Heterogeneous CPU-GPU cut-bridge pipelines (OpenCL + AVX2)
└── screening/      # Chemical latent space screening & molecule search drivers
```

---

## Subsystem Details

### 1. `scripts/charting/`
Visual telemetry suite adhering to the Deen OS Sovereign Visual Specification (Obsidian `#05080d`, Quantum Green `#00e676`, Radiant Gold `#ffb300`, Alert Red `#ef4444`, Polar Slate `#f8fafc`).
* **`chart_all.py`**: Master orchestrator executing all 16 scientific visualization modules.
* **`chart_ultrametric_tree.py`**: Visual telemetry for the 3-register Ultrametric Tree Sorter (`results/charts/ultrametric_tree_sorter_telemetry.png`).
* **`chart_birds_hybrid.py`**: Waterbirds subpopulation robustness and accuracy trajectories.
* **`chart_exascale_showdown.py`**: Desktop CPU/FPGA vs supercomputing infrastructure comparison.
* **`chart_nighthawk_layout.py`**: 120-qubit IBM Nighthawk R2 and Kagome strip layout rendering.
* **`deen_theme.py`**: Shared typography, color tokens, and card styling utilities.

### 2. `scripts/data/`
* **`extract_birds_features.py`**: Deep visual transfer feature extraction from full-resolution Waterbirds images using PyTorch MobileNetV3-Small. Performs CPU-accelerated inference (~245 img/s), extracts 576-D embeddings, computes PCA projection down to 10 principal components ($37.48\%$ visual variance), and writes standardized $[-\pi, \pi]^{10}$ binary tensors to `data/birds/`.

### 3. `scripts/fpga/`
* **`test_chaos_streaming.py`**: Connects to the physical Artix-7 FPGA over 3.0 MBaud UART (`COM6`) to verify cycle-accurate Porter-Thomas chaos streaming and hardware attestation hashes.
* **`fpga_terminal.py`**: Low-level interactive register inspection and telemetry readout.

### 4. `scripts/qec/`
* **`heterogeneous_bridge_pipeline.py`**: Orchestrates concurrent dual-register evaluation across CPU L3 cache (AVX2+FMA) and Intel Arc Xe iGPU (OpenCL), evaluating non-local Tsirelson CHSH Bell inequality witnesses ($S = 2.82842712$).

### 5. `scripts/screening/`
* Automated chemical property screening and latent space exploration drivers for ZINC 250k and MIMAC datasets.

---

## Usage Examples

```bash
# Render all 16 scientific charts to results/charts/
python scripts/charting/chart_all.py

# Extract 10-D PCA features from raw Waterbirds images
python scripts/data/extract_birds_features.py --dataset-zip birds_dataset.zip

# Test FPGA hardware streaming over UART
python scripts/fpga/test_chaos_streaming.py --port COM6 --baud 3000000

# Run heterogeneous CPU+iGPU cut-bridge pipeline
python scripts/qec/heterogeneous_bridge_pipeline.py
```
