# T-1000 Empirical Data Repositories

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
