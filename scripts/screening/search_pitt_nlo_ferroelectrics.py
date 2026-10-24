# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Search Option C: Targeted Pareto Search for Non-Linear Optical (NLO)
and High-Spontaneous-Polarization Ferroelectric Materials across Pitt Quantum Repository.
Conditions:
  - Dipole Moment |D| >= 20.0 Debye (Giant permanent polarization)
  - Optical Transparency Gap >= 3.0 eV (UV/visible laser transmission, no photo-bleaching)
  - Molecular Weight MW <= 500 Da (Processable, crystallizable, volatile for thin-film CVD)
  - Sorted by NLO Figure of Merit: FOM = Dipole * Polarizability
"""

from pathlib import Path
import csv
import json
import sys
import time

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    csv_path = Path("data/pitt/pitt_quantum.csv")
    out_dir = Path("results/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    web_data_dir = Path("web/data")
    web_data_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    candidates = []
    all_screened = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_screened += 1
            try:
                mw = float(row["molwt"])
                gap = float(row["gap"])
                dip = float(row["dipole"])
                alpha = float(row["polarizability"]) if row["polarizability"] else 0.0
                dx = float(row["dipole_x"]) if row["dipole_x"] else 0.0
                dy = float(row["dipole_y"]) if row["dipole_y"] else 0.0
                dz = float(row["dipole_z"]) if row["dipole_z"] else 0.0
                homo = float(row["homo"])
                lumo = float(row["lumo"])
                hf = float(row["heat_of_formation"]) if row["heat_of_formation"] else 0.0
            except (ValueError, TypeError):
                continue

            # Filtering thresholds:
            # High dipole (|D| >= 20 Debye) but excluding calculation artifacts (> 60 Debye)
            # Optical transparency (gap >= 3.0 eV)
            # Processable mass (mw <= 500 Da)
            if 20.0 <= dip <= 60.0 and gap >= 3.0 and mw <= 500.0:
                fom = dip * alpha
                candidates.append({
                    "inchikey": row["inchikey"],
                    "smiles": row["smiles"],
                    "formula": row["formula"],
                    "name": row["name"] or "unnamed",
                    "molwt": mw,
                    "point_group": row["point_group"],
                    "gap": gap,
                    "dipole": dip,
                    "dipole_x": dx,
                    "dipole_y": dy,
                    "dipole_z": dz,
                    "polarizability": alpha,
                    "homo": homo,
                    "lumo": lumo,
                    "heat_of_formation": hf,
                    "fom_nlo": fom,
                    "tags": row["tags"].split(";") if row["tags"] else [],
                })

    candidates.sort(key=lambda x: x["fom_nlo"], reverse=True)

    # Assign rank
    for idx, c in enumerate(candidates, 1):
        c["rank"] = idx

    json_path = out_dir / "pitt_nlo_ferroelectric_pareto.json"
    web_json_path = web_data_dir / "pitt_nlo_ferroelectric_pareto.json"

    report = {
        "search_name": "Option C: Non-Linear Optical (NLO) & Ferroelectric Frontier Candidates",
        "timestamp_unix": time.time(),
        "total_screened": all_screened,
        "total_passing": len(candidates),
        "filter_bounds": {
            "dipole_min_debye": 20.0,
            "dipole_max_debye": 60.0,
            "gap_min_ev": 3.0,
            "mw_max_da": 500.0,
        },
        "top_leads": candidates[:50],
        "all_candidates": candidates,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(web_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    elapsed = time.perf_counter() - t0
    print("=========================================================================")
    print("      NLO FERROELECTRIC PARETO SCREEN COMPLETE                           ")
    print("=========================================================================")
    print(f"Total Screened: {all_screened} molecules")
    print(f"Qualified NLO Leads: {len(candidates)} candidates ({len(candidates)/all_screened*100:.3f}%)")
    print(f"Execution Time: {elapsed:.2f} s")
    print(f"Saved: {json_path}")
    print(f"Web Data: {web_json_path}\n")

    print(f"┌{'─'*88}┐")
    print(f"│ {'TOP 10 NLO & FERROELECTRIC PARETO LEADS (ORDERED BY FOM = DIPOLE * ALPHA)'.ljust(86)} │")
    print(f"├{'─'*88}┤")
    print(f"│ {'Rk'.ljust(3)} │ {'Formula'.ljust(12)} │ {'MW(Da)'.ljust(7)} │ {'Dipole(D)'.ljust(9)} │ {'Gap(eV)'.ljust(7)} │ {'Alpha'.ljust(7)} │ {'FOM(NLO)'.ljust(9)} │ {'Name/Class'.ljust(15)} │")
    print(f"├{'─'*88}┤")
    for c in candidates[:10]:
        print(f"│ {str(c['rank']).ljust(3)} │ {c['formula'][:12].ljust(12)} │ {f'{c['molwt']:.1f}'.ljust(7)} │ {f'{c['dipole']:.2f}'.ljust(9)} │ {f'{c['gap']:.2f}'.ljust(7)} │ {f'{c['polarizability']:.1f}'.ljust(7)} │ {f'{c['fom_nlo']:.1f}'.ljust(9)} │ {c['name'][:15].ljust(15)} │")
    print(f"└{'─'*88}┘\n")


if __name__ == "__main__":
    main()
