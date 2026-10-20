# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Extracts a curated, high-diversity benchmark set of QM9 quantum molecules
with true B3LYP DFT 3D coordinates, Mulliken partial charges, and Coulomb eigenspectra
for direct ingestion into the T-1000 3D Web Dashboard.
"""

import zipfile
import json
from pathlib import Path
import numpy as np

COV_RADII = {
    'H': 0.31,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57
}

Z_MAP = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9
}

HARTREE_TO_EV = 27.211386245988

def infer_bonds(atoms):
    bonds = []
    n = len(atoms)
    for i in range(n):
        r1 = COV_RADII.get(atoms[i]['element'], 0.75)
        p1 = np.array(atoms[i]['coords'])
        for j in range(i + 1, n):
            r2 = COV_RADII.get(atoms[j]['element'], 0.75)
            p2 = np.array(atoms[j]['coords'])
            dist = float(np.linalg.norm(p1 - p2))
            if 0.4 <= dist <= 1.25 * (r1 + r2):
                bonds.append({'a1': i, 'a2': j, 'dist': round(dist, 3)})
    return bonds

def compute_coulomb_eigenspectrum(atoms, max_dim=50):
    n = len(atoms)
    C = np.zeros((max_dim, max_dim), dtype=np.float64)
    for i in range(n):
        z_i = Z_MAP.get(atoms[i]['element'], 6)
        C[i, i] = 0.5 * (z_i ** 2.4)
        p_i = np.array(atoms[i]['coords'])
        for j in range(i + 1, n):
            z_j = Z_MAP.get(atoms[j]['element'], 6)
            p_j = np.array(atoms[j]['coords'])
            dist = np.linalg.norm(p_i - p_j)
            if dist > 1e-4:
                val = (z_i * z_j) / dist
                C[i, j] = val
                C[j, i] = val
    
    eigs = np.linalg.eigvalsh(C)
    eigs_sorted = np.sort(eigs)[::-1]
    return [round(float(x), 4) for x in eigs_sorted], round(float(np.trace(C)), 4)

def parse_qm9_xyz(content):
    lines = content.strip().splitlines()
    if len(lines) < 5:
        return None
    n_atoms = int(lines[0].strip())
    props = lines[1].split()
    if len(props) < 17:
        return None
    
    index = int(props[1])
    mu = float(props[5])
    alpha = float(props[6])
    homo = float(props[7])
    lumo = float(props[8])
    gap = float(props[9])
    zpve = float(props[11])
    u0 = float(props[12])
    cv = float(props[16])

    atoms = []
    elements_count = {}
    for i in range(n_atoms):
        row = lines[2 + i].split()
        sym = row[0]
        x, y, z = float(row[1]), float(row[2]), float(row[3])
        q = float(row[4])
        atoms.append({
            'element': sym,
            'coords': [round(x, 4), round(y, 4), round(z, 4)],
            'mulliken_charge': round(q, 4),
            'x': round(x, 4),
            'y': round(y, 4),
            'z': round(z, 4)
        })
        elements_count[sym] = elements_count.get(sym, 0) + 1

    formula = "".join(f"{k}{elements_count[k] if elements_count[k] > 1 else ''}" for k in sorted(elements_count.keys()))

    smiles_line = lines[3 + n_atoms].split()
    smiles = smiles_line[1] if len(smiles_line) > 1 else smiles_line[0]

    bonds = infer_bonds(atoms)
    eigs, tr = compute_coulomb_eigenspectrum(atoms, 50)

    return {
        'id': index,
        'index': index,
        'is_qm9_molecule': True,
        'formula': formula,
        'smiles': smiles,
        'u0_hartree': round(u0, 6),
        'u0_ev': round(u0 * HARTREE_TO_EV, 2),
        'gap_hartree': round(gap, 6),
        'gap_ev': round(gap * HARTREE_TO_EV, 3),
        'homo_hartree': round(homo, 6),
        'lumo_hartree': round(lumo, 6),
        'dipole_debye': round(mu, 3),
        'polarizability_bohr3': round(alpha, 3),
        'zpve_hartree': round(zpve, 6),
        'heat_capacity_cv': round(cv, 3),
        'n_atoms': n_atoms,
        'atoms': atoms,
        'bonds': bonds,
        'coulomb_eigenspectrum': eigs,
        'coulomb_trace': tr,
        'display_metric': f"U₀ {round(u0, 1)} Ha | Gap {round(gap * HARTREE_TO_EV, 2)} eV"
    }

def main():
    qm9_zip = Path(r"C:\Users\zoddj\Downloads\QM9.zip")
    if not qm9_zip.exists():
        print(f"Error: {qm9_zip} not found.")
        return

    print("Opening QM9.zip...")
    selected_records = []
    
    with zipfile.ZipFile(qm9_zip, 'r') as z:
        # 1. Grab first 25 canonical molecules
        print("Selecting canonical fundamental structures (1..25)...")
        for i in range(1, 26):
            fname = f"dsgdb9nsd.xyz/dsgdb9nsd_{i:06d}.xyz"
            try:
                content = z.read(fname).decode('utf-8', errors='ignore')
                rec = parse_qm9_xyz(content)
                if rec:
                    rec['tag'] = 'FUNDAMENTAL'
                    selected_records.append(rec)
            except KeyError:
                pass

        # 2. Sample 35 additional diverse molecules across gap and dipole extremes
        print("Selecting diverse structural & electronic extremes...")
        sample_indices = [
            100, 250, 500, 1000, 2500, 5000, 7500, 10000, 15000, 20000,
            25000, 30000, 35000, 40000, 45000, 50000, 60000, 70000, 80000, 90000,
            100000, 110000, 120000, 130000, 133000
        ]
        for idx in sample_indices:
            fname = f"dsgdb9nsd.xyz/dsgdb9nsd_{idx:06d}.xyz"
            try:
                content = z.read(fname).decode('utf-8', errors='ignore')
                rec = parse_qm9_xyz(content)
                if rec:
                    if rec['gap_ev'] > 8.0:
                        rec['tag'] = 'WIDE_GAP'
                    elif rec['gap_ev'] < 5.0:
                        rec['tag'] = 'NARROW_GAP'
                    elif rec['dipole_debye'] > 4.5:
                        rec['tag'] = 'HIGH_DIPOLE'
                    else:
                        rec['tag'] = 'QM9_DFT'
                    selected_records.append(rec)
            except KeyError:
                pass

    print(f"Extracted {len(selected_records)} curated QM9 molecules.")
    out_dir = Path("web/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "qm9_quantum_molecules.json"

    dataset_payload = {
        "dataset_name": "QM9 B3LYP/6-31G(2df,p) Quantum Ground-State Structures",
        "description": "True DFT Cartesian coordinates, Mulliken partial charges, and 50x50 Coulomb eigenspectra",
        "level_of_theory": "B3LYP/6-31G(2df,p)",
        "is_qm9_dataset": True,
        "total_molecules": len(selected_records),
        "molecules": selected_records
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dataset_payload, f, indent=2)

    print(f"[OK] Saved {len(selected_records)} molecules to {out_json} ({out_json.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
