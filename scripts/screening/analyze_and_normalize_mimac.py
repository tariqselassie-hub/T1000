# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
MIMAC (Molecular Interaction Manifold Dataset) Comprehensive Analysis & Normalization Pipeline.
Processes manifest, 1024-bit multiview fingerprints, activity cliffs, and KNN graph topology.
Prepares datasets for classical and quantum Hilbert-space state injection.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

def main() -> None:
    data_dir = Path("data/mimac")
    out_dir = data_dir / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=========================================================================")
    print("      MIMAC DATASET ANALYSIS & QUANTUM NORMALIZATION PIPELINE           ")
    print("=========================================================================")

    # 1. LOAD DATASETS
    manifest_path = data_dir / "molecular_interaction_manifest.csv"
    fp_path = data_dir / "fingerprints_multiview.npy"
    cliffs_path = data_dir / "activity_cliffs.csv"
    knn_path = data_dir / "knn_graph_edges.csv"

    manifest = pd.read_csv(manifest_path)
    fps = np.load(fp_path)
    cliffs = pd.read_csv(cliffs_path)
    knn = pd.read_csv(knn_path)

    n_mols = len(manifest)
    print(f" [+] Loaded Manifest       : {n_mols} molecules, {manifest.shape[1]} columns")
    print(f" [+] Loaded Fingerprints   : shape {fps.shape}, dtype {fps.dtype}")
    print(f" [+] Loaded Activity Cliffs: {len(cliffs)} cliff pairs")
    print(f" [+] Loaded KNN Edges      : {len(knn)} directed interaction edges\n")

    # 2. INTEGRITY CHECKS
    print("[1/4] Performing Rigorous Data Integrity Verification...")
    assert len(fps) == n_mols, f"Fingerprints count ({len(fps)}) mismatch with manifest ({n_mols})"
    null_counts = manifest.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"  -> Total Missing / Null Values: {total_nulls}")
    assert total_nulls == 0, f"Found unexpected null values in manifest: {null_counts[null_counts > 0]}"

    # Validate cliff indices
    max_idx_cliff = max(cliffs["Mol_A"].max(), cliffs["Mol_B"].max())
    assert max_idx_cliff < n_mols, f"Cliff molecule index {max_idx_cliff} exceeds dataset size {n_mols}"

    # Validate KNN indices
    knn_src_col = knn.columns[0]
    knn_dst_col = knn.columns[1]
    max_idx_knn = max(knn[knn_src_col].max(), knn[knn_dst_col].max())
    assert max_idx_knn < n_mols, f"KNN molecule index {max_idx_knn} exceeds dataset size {n_mols}"
    print("  -> Alignment check PASSED: All indices, dimensions, and keys are 100% consistent.\n")

    # 3. STATISTICAL DISTRIBUTIONS & FEATURE ANALYSIS
    print("[2/4] Analyzing Physicochemical & Manifold Distributions...")
    feature_cols = [
        "Manifold_X", "Manifold_Y", "Local_Density", "LogP", "TPSA",
        "MolWt", "Dipole_Proxy", "Max_Partial_Charge", "Balaban_J", "Bertz_CT"
    ]
    target_col = "Bioactivity_Score"

    stats_summary = {}
    for col in feature_cols + [target_col]:
        series = manifest[col]
        skew = float(stats.skew(series))
        kurt = float(stats.kurtosis(series))
        stats_summary[col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "25%": float(series.quantile(0.25)),
            "50%": float(series.median()),
            "75%": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skewness": skew,
            "kurtosis": kurt,
        }

    print(f"{'Feature':<20} | {'Mean':<9} | {'Std':<9} | {'Min':<9} | {'Max':<9} | {'Skew':<7}")
    print("-" * 75)
    for col, s in stats_summary.items():
        print(f"{col:<20} | {s['mean']:<9.3f} | {s['std']:<9.3f} | {s['min']:<9.3f} | {s['max']:<9.3f} | {s['skewness']:<7.2f}")

    # Feature correlation with Bioactivity_Score
    correlations = {}
    for col in feature_cols:
        r_pearson, p_val = stats.pearsonr(manifest[col], manifest[target_col])
        r_spearman, _ = stats.spearmanr(manifest[col], manifest[target_col])
        correlations[col] = {
            "pearson_r": float(r_pearson),
            "p_value": float(p_val),
            "spearman_r": float(r_spearman),
        }

    print("\n  -> Top Feature Correlations with Bioactivity Score:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]["pearson_r"]), reverse=True)
    for col, c in sorted_corr[:5]:
        print(f"     * {col:<18}: Pearson r = {c['pearson_r']:+.4f} (Spearman = {c['spearman_r']:+.4f})")

    # Fingerprint Sparsity Analysis
    active_bits_per_mol = np.count_nonzero(fps, axis=1)
    bit_frequencies = np.count_nonzero(fps, axis=0) / n_mols
    non_zero_bits = np.count_nonzero(bit_frequencies > 0)
    print(f"\n  -> 1024-bit Multiview Fingerprint Sparsity:")
    print(f"     * Active Bits per Molecule: Mean = {active_bits_per_mol.mean():.1f} ± {active_bits_per_mol.std():.1f} (Min: {active_bits_per_mol.min()}, Max: {active_bits_per_mol.max()})")
    print(f"     * Informative Bit Positions: {non_zero_bits} / 1024 bits active in library ({non_zero_bits / 1024 * 100:.1f}%)")

    # 4. NORMALIZATION PIPELINE
    print("\n[3/4] Generating Multi-Scale Normalization Formats...")
    norm_df = manifest.copy()

    # A. Min-Max Normalization [0, 1]
    min_max_dict = {}
    for col in feature_cols + [target_col]:
        mn = float(manifest[col].min())
        mx = float(manifest[col].max())
        rng = mx - mn if mx > mn else 1.0
        norm_df[f"{col}_minmax"] = (manifest[col] - mn) / rng
        min_max_dict[col] = {"min": mn, "max": mx, "range": rng}

    # B. Standard Z-Score Normalization (mean=0, std=1)
    zscore_matrix = np.zeros((n_mols, len(feature_cols)), dtype=np.float64)
    zscore_dict = {}
    for i, col in enumerate(feature_cols):
        mu = float(manifest[col].mean())
        sigma = float(manifest[col].std()) if float(manifest[col].std()) > 1e-12 else 1.0
        z_col = (manifest[col] - mu) / sigma
        norm_df[f"{col}_zscore"] = z_col
        zscore_matrix[:, i] = z_col.values
        zscore_dict[col] = {"mean": mu, "std": sigma}

    # C. Quantum Parameterized Rotation Angle Mapping: [-pi/2, +pi/2]
    # Injects features directly into parameterized Ry(theta) rotation gates
    quantum_angles_matrix = np.zeros((n_mols, len(feature_cols)), dtype=np.float64)
    for i, col in enumerate(feature_cols):
        u = norm_df[f"{col}_minmax"].values
        # map u in [0, 1] to theta in [-pi/2, +pi/2]
        theta = np.pi * u - (np.pi / 2.0)
        norm_df[f"{col}_theta"] = theta
        quantum_angles_matrix[:, i] = theta

    # 5. ACTIVITY CLIFF ANALYSIS
    print("\n[4/4] Profiling Activity Cliff Topography...")
    top_cliffs = cliffs.sort_values(by="Activity_Delta", ascending=False).head(10)
    print("  -> Top 5 Steepest Activity Cliffs (High Similarity, Enormous Activity Jumps):")
    for _, row in top_cliffs.head(5).iterrows():
        idx_a, idx_b = int(row["Mol_A"]), int(row["Mol_B"])
        smiles_a = manifest.iloc[idx_a]["SMILES"]
        smiles_b = manifest.iloc[idx_b]["SMILES"]
        act_a = manifest.iloc[idx_a]["Bioactivity_Score"]
        act_b = manifest.iloc[idx_b]["Bioactivity_Score"]
        print(f"     * Delta = {row['Activity_Delta']:.3f} | Sim = {row['Similarity']:.3f}")
        print(f"       Mol A (#{idx_a}, Act: {act_a:.2f}): {smiles_a}")
        print(f"       Mol B (#{idx_b}, Act: {act_b:.2f}): {smiles_b}")

    # 6. EXPORT NORMALIZED ARTIFACTS
    norm_csv_path = out_dir / "mimac_normalized_manifest.csv"
    norm_df.to_csv(norm_csv_path, index=False)
    print(f"\n [+] Exported Normalized Manifest: {norm_csv_path}")

    zscore_path = out_dir / "mimac_zscore_features.npy"
    np.save(zscore_path, zscore_matrix)
    print(f" [+] Exported Z-Score Matrix   : {zscore_path} (shape: {zscore_matrix.shape})")

    angles_path = out_dir / "mimac_quantum_angles.npy"
    np.save(angles_path, quantum_angles_matrix)
    print(f" [+] Exported Quantum Angles   : {angles_path} (shape: {quantum_angles_matrix.shape})")

    report = {
        "dataset_name": "MIMAC (Molecular Interaction Manifold Dataset)",
        "total_molecules": n_mols,
        "features_count": len(feature_cols),
        "feature_names": feature_cols,
        "target_name": target_col,
        "cliffs_count": len(cliffs),
        "knn_edges_count": len(knn),
        "multiview_fingerprint_bits": fps.shape[1],
        "active_fingerprint_bits": int(non_zero_bits),
        "statistical_distributions": stats_summary,
        "bioactivity_correlations": correlations,
        "scaling_parameters": {
            "min_max": min_max_dict,
            "z_score": zscore_dict,
        },
    }

    report_path = out_dir / "mimac_analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f" [+] Exported Analysis Report  : {report_path}")

    print("\n=========================================================================")
    print("      DATASET IS PROPERLY VALIDATED, NORMALIZED, AND READY FOR SEARCH    ")
    print("=========================================================================\n")

if __name__ == "__main__":
    main()
