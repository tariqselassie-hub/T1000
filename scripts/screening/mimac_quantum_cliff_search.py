# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
MIMAC Search A: Quantum Activity Cliff & SAR Discontinuity Hunter.
Computes quantum state fidelity F(A, B) across 10-qubit Hilbert space and
combines with 1024-bit multiview fingerprint Tanimoto similarity to identify
structural magic switches unlocking maximal potency leaps with minimal molecular perturbation.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

def compute_quantum_state_amplitudes(theta_vec: np.ndarray) -> np.ndarray:
    """Prepares a separable 10-qubit product state |psi> from normalized rotation angles.
    theta in [-pi/2, +pi/2] -> angle in [0, pi]
    |q_i> = cos(phi/2) |0> + sin(phi/2) |1>
    Returns 1024-dimensional real statevector.
    """
    phi = theta_vec + (np.pi / 2.0) # [0, pi]
    c = np.cos(phi / 2.0)
    s = np.sin(phi / 2.0)

    # Tensor product across 10 qubits
    state = np.array([1.0], dtype=np.float64)
    for q in range(len(theta_vec)):
        qubit_state = np.array([c[q], s[q]], dtype=np.float64)
        state = np.kron(state, qubit_state)
    return state

def compute_tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    dot = np.dot(fp_a, fp_b)
    denom = np.sum(fp_a) + np.sum(fp_b) - dot
    return float(dot / denom) if denom > 0 else 0.0

def main() -> None:
    print("=========================================================================")
    print("      MIMAC SEARCH A: QUANTUM ACTIVITY CLIFF & SAR DISCONTINUITY HUNTER   ")
    print("=========================================================================")

    data_dir = Path("data/mimac")
    norm_manifest_path = data_dir / "normalized/mimac_normalized_manifest.csv"
    angles_path = data_dir / "normalized/mimac_quantum_angles.npy"
    fp_path = data_dir / "fingerprints_multiview.npy"
    cliffs_path = data_dir / "activity_cliffs.csv"

    manifest = pd.read_csv(norm_manifest_path)
    quantum_angles = np.load(angles_path)
    fps = np.load(fp_path)
    raw_cliffs = pd.read_csv(cliffs_path)

    n_mols = len(manifest)
    print(f" [+] Ingested {n_mols} molecules, {len(raw_cliffs)} cliff candidate pairs.")

    # 1. PRECOMPUTE QUANTUM STATEVECTORS
    print(" [+] Preparing 10-qubit Hilbert space statevectors (2^10 = 1024 dims)...")
    quantum_states = np.zeros((n_mols, 1024), dtype=np.float64)
    for i in range(n_mols):
        quantum_states[i] = compute_quantum_state_amplitudes(quantum_angles[i])

    # 2. EVALUATE ALL CANDIDATE PAIRS & SEARCH FOR CLIFFS
    print(" [+] Evaluating Quantum-Classical SAR Discontinuity Metric...")
    cliff_records = []

    for _, row in raw_cliffs.iterrows():
        idx_a = int(row["Mol_A"])
        idx_b = int(row["Mol_B"])

        mol_a = manifest.iloc[idx_a]
        mol_b = manifest.iloc[idx_b]

        act_a = float(mol_a["Bioactivity_Score"])
        act_b = float(mol_b["Bioactivity_Score"])
        act_delta = abs(act_b - act_a)

        # Classical Tanimoto on 1024-bit Multiview Fingerprints
        tanimoto = compute_tanimoto(fps[idx_a], fps[idx_b])

        # Quantum Fidelity in 10-Qubit Hilbert Space: F = |<psi_A|psi_B>|^2
        inner_prod = np.dot(quantum_states[idx_a], quantum_states[idx_b])
        quantum_fidelity = float(inner_prod ** 2)

        # Combined Quantum-Classical Structural Overlap
        hybrid_similarity = float(np.sqrt(tanimoto * quantum_fidelity))

        # Cliff Salience / Potency Multiplier:
        # High structural similarity with enormous activity leap
        cliff_salience = act_delta / (1.0 - hybrid_similarity + 1e-4)
        potency_ratio = max(act_a, act_b) / max(min(act_a, act_b), 0.1)

        # Directional: from low activity (inactive/parent) to high activity (potent lead)
        if act_a <= act_b:
            idx_parent, idx_lead = idx_a, idx_b
            act_parent, act_lead = act_a, act_b
        else:
            idx_parent, idx_lead = idx_b, idx_a
            act_parent, act_lead = act_b, act_a

        smiles_parent = manifest.iloc[idx_parent]["SMILES"]
        smiles_lead = manifest.iloc[idx_lead]["SMILES"]

        # Physicochemical shifts
        delta_logp = float(manifest.iloc[idx_lead]["LogP"] - manifest.iloc[idx_parent]["LogP"])
        delta_tpsa = float(manifest.iloc[idx_lead]["TPSA"] - manifest.iloc[idx_parent]["TPSA"])
        delta_mw = float(manifest.iloc[idx_lead]["MolWt"] - manifest.iloc[idx_parent]["MolWt"])

        cliff_records.append({
            "idx_parent": idx_parent,
            "idx_lead": idx_lead,
            "smiles_parent": smiles_parent,
            "smiles_lead": smiles_lead,
            "act_parent": act_parent,
            "act_lead": act_lead,
            "act_delta": act_delta,
            "potency_gain_fold": potency_ratio,
            "tanimoto_similarity": tanimoto,
            "quantum_fidelity": quantum_fidelity,
            "hybrid_similarity": hybrid_similarity,
            "cliff_salience": cliff_salience,
            "delta_logp": delta_logp,
            "delta_tpsa": delta_tpsa,
            "delta_mw": delta_mw,
        })

    # Deduplicate symmetric pairs & sort by Cliff Salience
    # Create unique pair key
    unique_cliffs = {}
    for r in cliff_records:
        pair_key = (min(r["idx_parent"], r["idx_lead"]), max(r["idx_parent"], r["idx_lead"]))
        if pair_key not in unique_cliffs or r["cliff_salience"] > unique_cliffs[pair_key]["cliff_salience"]:
            unique_cliffs[pair_key] = r

    sorted_cliffs = sorted(unique_cliffs.values(), key=lambda x: x["cliff_salience"], reverse=True)
    for rank, item in enumerate(sorted_cliffs):
        item["rank"] = rank + 1

    top_20_cliffs = sorted_cliffs[:20]

    # 3. SAVE RESULTS
    out_dir = Path("results/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "mimac_quantum_activity_cliffs.json"
    csv_path = out_dir / "mimac_quantum_activity_cliffs.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(top_20_cliffs, f, indent=2)

    df_out = pd.DataFrame(top_20_cliffs)
    df_out.to_csv(csv_path, index=False)

    print(f" [+] Exported Top 20 Quantum Activity Cliffs: {json_path}")
    print(f" [+] Exported CSV Summary                  : {csv_path}\n")

    # 4. LEADERBOARD DISPLAY
    print("=========================================================================")
    print("         TOP 5 QUANTUM ACTIVITY CLIFFS & BIOACTIVITY MAGIC SWITCHES      ")
    print("=========================================================================")
    for c in top_20_cliffs[:5]:
        print(f" #{c['rank']} | Cliff Salience: {c['cliff_salience']:.2f} | Activity Jump: {c['act_parent']:.2f} -> {c['act_lead']:.2f} (+{c['act_delta']:.2f}, {c['potency_gain_fold']:.1f}x)")
        print(f"     Hybrid Sim: {c['hybrid_similarity']:.3f} (Quantum Fid: {c['quantum_fidelity']:.3f}, Tanimoto: {c['tanimoto_similarity']:.3f})")
        print(f"     Parent Inactive [#{c['idx_parent']}]: {c['smiles_parent']}")
        print(f"     Active Lead     [#{c['idx_lead']}]: {c['smiles_lead']}")
        print(f"     Property Delta  : dLogP = {c['delta_logp']:+.2f} | dTPSA = {c['delta_tpsa']:+.1f} | dMW = {c['delta_mw']:+.1f}\n")
    print("=========================================================================")

if __name__ == "__main__":
    main()
