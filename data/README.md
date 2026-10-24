# T-1000 Empirical Data Repositories

Datasets are **not** distributed with this repository. Fetch your own copies into the directories
below; everything under `data/` except the readmes and `mnist_demo/` is git-ignored.

| Directory | What | Where to get it |
| :--- | :--- | :--- |
| `data/mnist_demo/` | 1,000 training and 200 test digits (100 and 20 per class), IDX format, tracked in git | included; enough to run `mnist_qnn_train` and `mnist_ultrametric_sorter` as a demo |
| `data/mnist/` | Full MNIST (`train-images.idx3-ubyte`, `train-labels.idx1-ubyte`, `t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`) | https://yann.lecun.com/exdb/mnist/ ; if present it is used instead of the demo set |
| `data/qm9/` | QM9 properties (`qm9.csv`) | see [`qm9/README.md`](qm9/README.md) |
| `data/birds/`, `data/mimac/` | Waterbirds features and the MIMAC chemistry tables described below | built by the extraction and screening scripts in `scripts/`; the Waterbirds images come from the public Waterbirds benchmark |

Results measured on the demo subset are not comparable with results on full MNIST.

---

This directory houses scientific datasets, binary feature maps, chemical descriptors, and benchmark manifests utilized across quantum neural network training and molecular screening.

---

## Directory Organization

```
data/
├── birds/      # Waterbirds 11k dataset (raw pixels, binary labels, 10-D PCA embeddings)
└── mimac/      # Molecular Interaction Map & Activity Cliffs (MIMAC) chemical datasets
```

---

## Subsystem Manifests

### 1. `data/birds/` (Waterbirds 11k Avian Confounder Benchmark)
Used to evaluate quantum robustness, active measurement orientation sorting, and the Ultrametric Tree Sorter.

* **Binary Feature Tensors (`*_features_10d.bin`):**
  * Extracted via PyTorch MobileNetV3-Small from full-resolution $224 \times 224$ images.
  * Dimension: 10 principal components ($37.48\%$ visual variance explained).
  * Format: Raw IEEE 754 32-bit single-precision floats (`f32`), normalized to $[-\pi, \pi]^{10}$.
  * Split Counts: Train (4,795 samples, 191.8 KB), Val (1,199 samples, 48.0 KB), Test (5,794 samples, 231.8 KB).
  * Manifest: [`features_manifest.json`](features_manifest.json).
* **Labels & Confounders (`*_labels.bin`, `*_places.bin`):**
  * `labels.bin`: Core class $y \in \{0, 1\}$ (0 = Landbird, 1 = Waterbird).
  * `places.bin`: Background domain $a \in \{0, 1\}$ (0 = Land, 1 = Water).
  * Confounder Groups: $G_0$ (Land/Land), $G_1$ (Land/Water), $G_2$ (Water/Land), $G_3$ (Water/Water).
* **Raw Amplitudes (`*_images.bin`):**
  * $32 \times 32$ grayscale photographic pixels mapped into $1,024$-amplitude normalized statevectors ($|\psi\rangle = \sum_i x_i |i\rangle$).

### 2. `data/mimac/` (Molecular Activity Cliffs & Interaction Manifolds)
Contains chemical structure datasets and activity cliff matrices for ultra-high-throughput latent screening:
* `activity_cliffs.csv`: Pairwise activity cliffs and topological molecular distances.
* `fingerprints_multiview.npy`: Multi-view chemical fingerprints for latent VQC injection.
* `molecular_interaction_manifest.csv`: Binding affinity metadata and zwitterion profiles.

---

## Security & Storage Notice

All binary feature files are stored in native little-endian formats for direct zero-copy `mmap` or `slice::from_raw_parts` parsing without serialization runtime penalties.
