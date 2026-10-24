# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Screening: University of Pittsburgh Quantum Repository (106k molecules).
Executes 5 targeted quantum searches across PM7 electronic and topological observables:
  Search A: Near-Zero Gap Organic Conductors & Diradical Candidates (HOMO-LUMO Gap < 2.0 eV)
  Search B: Giant Dipole Ferroelectric & Non-Linear Optical Candidates (Max Dipole Moment)
  Search C: Maximum Specific Polarizability (Electron Cloud Responsiveness per Dalton)
  Search D: High-Symmetry Point Groups for Equivariant QNN Benchmarks (D3h, D2h, C3v)
  Search E: Drug-Like Sulfur & Phosphorus Heterocyclic Scaffolds (Lipinski Ro5 Filtered)
"""

from pathlib import Path
import argparse
import csv
import json
import sys
import time

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def parse_float(val):
    if not val:
        return None
    try:
        f = float(val)
        return f if f == f else None
    except (ValueError, TypeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Pitt Quantum Repository Screening Suite")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/pitt/pitt_quantum.csv",
        help="Path to normalized pitt_quantum.csv",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default="results/experiments/pitt_quantum_searches.json",
        help="Path to output JSON experiment report",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV at {csv_path}. Run extract_pitt_quantum.py first.")

    print("=========================================================================")
    print("      T-1000 SOVEREIGN QUANTUM SCREENING: PITT QUANTUM REPOSITORY        ")
    print("=========================================================================")
    print(f"Loading {csv_path}...")
    t0 = time.perf_counter()

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            molwt = parse_float(row["molwt"])
            homo = parse_float(row["homo"])
            lumo = parse_float(row["lumo"])
            gap = parse_float(row["gap"])
            dipole = parse_float(row["dipole"])
            alpha = parse_float(row["polarizability"])
            hf = parse_float(row["heat_of_formation"])
            n_orbs = int(row["n_orbitals"]) if row["n_orbitals"] else 0

            records.append({
                "inchikey": row["inchikey"],
                "smiles": row["smiles"],
                "formula": row["formula"],
                "name": row["name"] or "unnamed",
                "molwt": molwt,
                "point_group": row["point_group"],
                "homo": homo,
                "lumo": lumo,
                "gap": gap,
                "dipole": dipole,
                "polarizability": alpha,
                "heat_of_formation": hf,
                "n_orbitals": n_orbs,
                "tags": row["tags"].split(";") if row["tags"] else [],
            })

    load_time = time.perf_counter() - t0
    print(f"Loaded {len(records)} molecules in {load_time:.2f} s\n")

    # -------------------------------------------------------------------------
    # SEARCH A: Near-Zero Gap Organic Conductors & Diradical Candidates
    # -------------------------------------------------------------------------
    print("Executing Search A: Near-Zero Gap Conductors (HOMO-LUMO Gap < 2.0 eV)...")
    valid_gap = [
        r for r in records
        if r["gap"] is not None and r["gap"] > 0.05 and r["molwt"] is not None and r["molwt"] >= 60.0
    ]
    valid_gap.sort(key=lambda x: x["gap"])
    top_search_a = valid_gap[:10]

    # -------------------------------------------------------------------------
    # SEARCH B: Giant Dipole Ferroelectric & Non-Linear Optical Candidates
    # -------------------------------------------------------------------------
    print("Executing Search B: Giant Permanent Dipoles (|D| > 10 Debye)...")
    # Exclude non-physical calculation artifacts (> 50 Debye)
    valid_dipole = [
        r for r in records
        if r["dipole"] is not None and 10.0 <= r["dipole"] <= 50.0 and r["molwt"] is not None and r["molwt"] >= 80.0
    ]
    valid_dipole.sort(key=lambda x: x["dipole"], reverse=True)
    top_search_b = valid_dipole[:10]

    # -------------------------------------------------------------------------
    # SEARCH C: Maximum Specific Polarizability (alpha / MW)
    # -------------------------------------------------------------------------
    print("Executing Search C: Maximum Specific Polarizability (alpha / MW)...")
    valid_alpha = [
        r for r in records
        if r["polarizability"] is not None and r["polarizability"] > 0 and r["molwt"] is not None and r["molwt"] >= 50.0
    ]
    for r in valid_alpha:
        r["specific_alpha"] = r["polarizability"] / r["molwt"]
    valid_alpha.sort(key=lambda x: x["specific_alpha"], reverse=True)
    top_search_c = valid_alpha[:10]

    # -------------------------------------------------------------------------
    # SEARCH D: High-Symmetry Molecular Crystals & Point Groups
    # -------------------------------------------------------------------------
    print("Executing Search D: High-Symmetry Point Groups (D3h, D2h, C3v)...")
    target_groups = {"D3h", "D2h", "C3v", "D*h"}
    sym_mols = [
        r for r in records
        if r["point_group"] in target_groups and r["molwt"] is not None and r["molwt"] >= 50.0
    ]
    sym_mols.sort(key=lambda x: (x["point_group"], -(x["molwt"] or 0)))
    top_search_d = sym_mols[:10]

    # -------------------------------------------------------------------------
    # SEARCH E: Drug-Like Sulfur & Phosphorus Heterocycles (Lipinski Filtered)
    # -------------------------------------------------------------------------
    print("Executing Search E: Bioactive Sulfur & Phosphorus Heterocycles (Lipinski Filtered)...")
    bio_mols = []
    for r in records:
        mw = r["molwt"]
        gap = r["gap"]
        f = r["formula"]
        if mw is None or gap is None:
            continue
        # Lipinski MW filter: 150 <= MW <= 500
        if 150.0 <= mw <= 500.0 and 3.0 <= gap <= 7.0:
            if ("S" in f or "P" in f) and ("ring" in r["tags"] or "aromatic" in r["tags"] or "amide" in r["tags"]):
                bio_mols.append(r)

    # Sort by moderate dipole and high polarizability
    bio_mols.sort(key=lambda x: -(x["polarizability"] or 0))
    top_search_e = bio_mols[:10]

    # -------------------------------------------------------------------------
    # DISPLAY TABULAR RESULTS
    # -------------------------------------------------------------------------
    def print_table(title, items, cols):
        print(f"\n┌{'─'*85}┐")
        print(f"│ {title.ljust(83)} │")
        print(f"├{'─'*85}┤")
        header = "│ " + " │ ".join(c[0].ljust(c[1]) for c in cols) + " │"
        print(header)
        print(f"├{'─'*85}┤")
        for item in items:
            row_vals = []
            for c in cols:
                key = c[2]
                val = item.get(key)
                fmt = c[3] if len(c) > 3 else None
                if fmt and val is not None:
                    txt = fmt(val)
                else:
                    txt = str(val or "")
                row_vals.append(txt[:c[1]].ljust(c[1]))
            print("│ " + " │ ".join(row_vals) + " │")
        print(f"└{'─'*85}┘")

    print_table(
        "SEARCH A: TOP 10 NEAR-ZERO GAP ORGANIC CONDUCTORS (HOMO-LUMO GAP < 2.0 eV)",
        top_search_a,
        [
            ("SMILES", 30, "smiles"),
            ("Formula", 12, "formula"),
            ("MW", 8, "molwt", lambda v: f"{v:.1f}"),
            ("HOMO", 8, "homo", lambda v: f"{v:.2f}"),
            ("LUMO", 8, "lumo", lambda v: f"{v:.2f}"),
            ("Gap(eV)", 9, "gap", lambda v: f"{v:.3f}"),
        ],
    )

    print_table(
        "SEARCH B: TOP 10 GIANT DIPOLE NON-LINEAR OPTICAL MOLECULES",
        top_search_b,
        [
            ("SMILES", 30, "smiles"),
            ("Formula", 12, "formula"),
            ("MW", 8, "molwt", lambda v: f"{v:.1f}"),
            ("Dipole(D)", 10, "dipole", lambda v: f"{v:.2f}"),
            ("Gap(eV)", 9, "gap", lambda v: f"{v:.2f}"),
            ("Name", 10, "name"),
        ],
    )

    print_table(
        "SEARCH C: TOP 10 MAXIMUM SPECIFIC POLARIZABILITY (ALPHA / MW)",
        top_search_c,
        [
            ("SMILES", 30, "smiles"),
            ("Formula", 12, "formula"),
            ("MW", 8, "molwt", lambda v: f"{v:.1f}"),
            ("Alpha", 8, "polarizability", lambda v: f"{v:.1f}"),
            ("Alpha/MW", 10, "specific_alpha", lambda v: f"{v:.4f}"),
            ("Gap(eV)", 8, "gap", lambda v: f"{v:.2f}"),
        ],
    )

    print_table(
        "SEARCH D: HIGH-SYMMETRY MOLECULES (D3h, D2h, C3v)",
        top_search_d,
        [
            ("SMILES", 28, "smiles"),
            ("Formula", 12, "formula"),
            ("PointGrp", 9, "point_group"),
            ("MW", 8, "molwt", lambda v: f"{v:.1f}"),
            ("Gap(eV)", 9, "gap", lambda v: f"{v:.2f}"),
            ("Name", 11, "name"),
        ],
    )

    print_table(
        "SEARCH E: TOP 10 BIOACTIVE SULFUR & PHOSPHORUS HETEROCYCLES",
        top_search_e,
        [
            ("SMILES", 28, "smiles"),
            ("Formula", 12, "formula"),
            ("MW", 8, "molwt", lambda v: f"{v:.1f}"),
            ("Alpha", 8, "polarizability", lambda v: f"{v:.1f}"),
            ("Dipole(D)", 9, "dipole", lambda v: f"{v:.2f}"),
            ("Gap(eV)", 9, "gap", lambda v: f"{v:.2f}"),
        ],
    )

    # Save output to JSON
    experiment_results = {
        "dataset": "University of Pittsburgh Quantum Repository (106k molecules)",
        "source_csv": str(csv_path),
        "total_molecules_screened": len(records),
        "search_a_near_zero_gap_conductors": top_search_a,
        "search_b_giant_dipoles": top_search_b,
        "search_c_max_specific_polarizability": top_search_c,
        "search_d_high_symmetry_point_groups": top_search_d,
        "search_e_sulfur_phosphorus_heterocycles": top_search_e,
    }

    with open(out_json, "w", encoding="utf-8") as f_json:
        json.dump(experiment_results, f_json, indent=2)

    print(f"\n[+] Screening Results Successfully Exported to: {out_json}")
    print(f"[+] Total execution time: {time.perf_counter() - t0:.2f} s\n")


if __name__ == "__main__":
    main()
