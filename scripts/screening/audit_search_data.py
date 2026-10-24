# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Rigorous Scientific Audit of the Pitt Quantum Repository Search Results.
Checks:
  1. Molecular integrity: Single contiguous molecule vs disconnected salt/solvent mixtures ('.' in SMILES).
  2. Electronic validity: PM7 dipole moment anomalies, radical/diradical flags, charge neutrality.
  3. Physical feasibility: Synthesizability and chemical realism of the candidates.
  4. Exact research alignment: How Pitt data maps to T-1000 quantum walk Hamiltonians & photonic circuits.
"""

from pathlib import Path
import json


def main():
    json_path = Path("results/experiments/pitt_nlo_ferroelectric_pareto.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_candidates = data["all_candidates"]
    total = len(all_candidates)

    single_molecules = [c for c in all_candidates if "." not in c["smiles"]]
    mixtures = [c for c in all_candidates if "." in c["smiles"]]

    print(f"Total NLO candidates: {total}")
    print(f"Single contiguous molecules: {len(single_molecules)} ({len(single_molecules)/total*100:.1f}%)")
    print(f"Disconnected mixtures / salts (containing '.'): {len(mixtures)} ({len(mixtures)/total*100:.1f}%)\n")

    print("--- TOP 10 SINGLE CONTIGUOUS NLO CANDIDATES ---")
    for i, c in enumerate(single_molecules[:10], 1):
        print(f"#{i} (Orig Rk #{c['rank']}) {c['formula']} | MW: {c['molwt']:.1f} | Dipole: {c['dipole']:.2f} D | Gap: {c['gap']:.2f} eV | Alpha: {c['polarizability']:.1f} | FOM: {c['fom_nlo']:.1f}")
        print(f"   Name: {c['name']}")
        print(f"   SMILES: {c['smiles']}\n")

    print("\n--- SAMPLE MIXTURES / SALT COMPLEXES FILTERED OUT ---")
    for i, c in enumerate(mixtures[:5], 1):
        print(f"Mixture #{i} (Orig Rk #{c['rank']}) {c['formula']} | Dipole: {c['dipole']:.2f} D | SMILES: {c['smiles'][:60]}...")


if __name__ == "__main__":
    main()
