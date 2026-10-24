# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Extract and normalize the University of Pittsburgh Quantum Repository (106k molecules).
Parses Pitt_Quantum_Repository_Data.json from zip archive into data/pitt/pitt_quantum.csv
and outputs dataset summary statistics to data/pitt/pitt_quantum_summary.json.
"""

from pathlib import Path
import argparse
import csv
import json
import time
import zipfile
from collections import Counter


def to_float(val):
    if val is None:
        return None
    try:
        f = float(val)
        if f != f:  # NaN
            return None
        return f
    except (ValueError, TypeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Pitt Quantum Repository")
    parser.add_argument(
        "--zip-path",
        type=str,
        default=r"C:\Users\zoddj\Models\pitt-quantum-repository.zip",
        help="Path to pitt-quantum-repository.zip",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/pitt",
        help="Output directory for extracted CSV and summary",
    )
    args = parser.parse_args()

    zip_file = Path(args.zip_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not zip_file.exists():
        raise FileNotFoundError(f"Archive not found at {zip_file}")

    print(f"Opening archive: {zip_file}")
    t0 = time.perf_counter()

    with zipfile.ZipFile(zip_file, "r") as z:
        print("Reading Pitt_Quantum_Repository_Data.json...")
        with z.open("Pitt_Quantum_Repository_Data.json") as f:
            raw_data = json.load(f)

    load_time = time.perf_counter() - t0
    total_raw = len(raw_data)
    print(f"Loaded {total_raw} raw records in {load_time:.2f} s")

    csv_path = out_dir / "pitt_quantum.csv"
    summary_path = out_dir / "pitt_quantum_summary.json"

    fieldnames = [
        "inchikey",
        "smiles",
        "formula",
        "molwt",
        "point_group",
        "homo",
        "lumo",
        "gap",
        "dipole",
        "dipole_x",
        "dipole_y",
        "dipole_z",
        "polarizability",
        "heat_of_formation",
        "surface_area",
        "volume",
        "rot_a",
        "rot_b",
        "rot_c",
        "n_orbitals",
        "tags",
        "name",
    ]

    valid_count = 0
    skipped_smiles = 0
    element_counter = Counter()
    point_group_counter = Counter()

    homo_vals = []
    lumo_vals = []
    gap_vals = []
    dipole_vals = []
    alpha_vals = []
    hf_vals = []
    mw_vals = []

    print(f"Normalizing and writing to {csv_path}...")
    t_write_0 = time.perf_counter()

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
        writer.writeheader()

        for item in raw_data:
            smi = (item.get("smiles") or "").strip()
            if not smi:
                skipped_smiles += 1
                continue

            pm7 = item.get("pm7") or {}
            homo = to_float(pm7.get("homo"))
            lumo = to_float(pm7.get("lumo"))
            gap = None
            if homo is not None and lumo is not None:
                gap = round(lumo - homo, 6)

            dipole = to_float(pm7.get("dipoleMoment"))
            dip_vec = pm7.get("dipole") or []
            dx = to_float(dip_vec[0]) if len(dip_vec) > 0 else None
            dy = to_float(dip_vec[1]) if len(dip_vec) > 1 else None
            dz = to_float(dip_vec[2]) if len(dip_vec) > 2 else None

            alpha = to_float(pm7.get("polarizability"))
            hf = to_float(pm7.get("heatOfFormation"))
            sa = to_float(pm7.get("surfaceArea"))
            vol = to_float(pm7.get("volume"))

            rot_vec = pm7.get("rotational") or []
            ra = to_float(rot_vec[0]) if len(rot_vec) > 0 else None
            rb = to_float(rot_vec[1]) if len(rot_vec) > 1 else None
            rc = to_float(rot_vec[2]) if len(rot_vec) > 2 else None

            orbs = pm7.get("orbitals") or []
            n_orbs = len(orbs) if isinstance(orbs, list) else 0

            mw = to_float(item.get("molecular mass"))
            pg = item.get("pointGroup") or "C1"
            tags = ";".join(item.get("tags") or [])
            name = (item.get("name") or "").strip()
            inchikey = (item.get("inchikey") or "").strip()
            formula = (item.get("formula") or "").strip()

            row = {
                "inchikey": inchikey,
                "smiles": smi,
                "formula": formula,
                "molwt": f"{mw:.4f}" if mw is not None else "",
                "point_group": pg,
                "homo": f"{homo:.4f}" if homo is not None else "",
                "lumo": f"{lumo:.4f}" if lumo is not None else "",
                "gap": f"{gap:.4f}" if gap is not None else "",
                "dipole": f"{dipole:.4f}" if dipole is not None else "",
                "dipole_x": f"{dx:.4f}" if dx is not None else "",
                "dipole_y": f"{dy:.4f}" if dy is not None else "",
                "dipole_z": f"{dz:.4f}" if dz is not None else "",
                "polarizability": f"{alpha:.4f}" if alpha is not None else "",
                "heat_of_formation": f"{hf:.4f}" if hf is not None else "",
                "surface_area": f"{sa:.4f}" if sa is not None else "",
                "volume": f"{vol:.4f}" if vol is not None else "",
                "rot_a": f"{ra:.6f}" if ra is not None else "",
                "rot_b": f"{rb:.6f}" if rb is not None else "",
                "rot_c": f"{rc:.6f}" if rc is not None else "",
                "n_orbitals": n_orbs,
                "tags": tags,
                "name": name,
            }
            writer.writerow(row)
            valid_count += 1

            for el in item.get("elements") or []:
                element_counter[el] += 1
            point_group_counter[pg] += 1

            if homo is not None:
                homo_vals.append(homo)
            if lumo is not None:
                lumo_vals.append(lumo)
            if gap is not None:
                gap_vals.append(gap)
            if dipole is not None:
                dipole_vals.append(dipole)
            if alpha is not None:
                alpha_vals.append(alpha)
            if hf is not None:
                hf_vals.append(hf)
            if mw is not None:
                mw_vals.append(mw)

    write_time = time.perf_counter() - t_write_0
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)

    def stats(arr):
        if not arr:
            return {}
        s = sorted(arr)
        return {
            "count": len(arr),
            "min": round(float(min(arr)), 4),
            "max": round(float(max(arr)), 4),
            "median": round(float(s[len(s) // 2]), 4),
            "mean": round(float(sum(arr) / len(arr)), 4),
        }

    summary_data = {
        "dataset_name": "University of Pittsburgh Quantum Repository (PM7)",
        "source_archive": str(zip_file),
        "total_records_ingested": total_raw,
        "valid_molecules_written": valid_count,
        "skipped_missing_smiles": skipped_smiles,
        "csv_output_file": str(csv_path),
        "csv_file_size_mb": round(file_size_mb, 2),
        "top_elements": dict(element_counter.most_common(20)),
        "top_point_groups": dict(point_group_counter.most_common(12)),
        "properties_statistics": {
            "molecular_weight": stats(mw_vals),
            "homo_ev": stats(homo_vals),
            "lumo_ev": stats(lumo_vals),
            "gap_ev": stats(gap_vals),
            "dipole_debye": stats(dipole_vals),
            "polarizability": stats(alpha_vals),
            "heat_of_formation_kj_mol": stats(hf_vals),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f_sum:
        json.dump(summary_data, f_sum, indent=2)

    print("\nExtraction & Normalization Complete:")
    print(f"  Valid records written: {valid_count}")
    print(f"  Skipped (no SMILES):  {skipped_smiles}")
    print(f"  CSV file size:        {file_size_mb:.2f} MB ({csv_path})")
    print(f"  Summary JSON:         {summary_path}")
    print(f"  Total processing time: {time.perf_counter() - t0:.2f} s")


if __name__ == "__main__":
    main()
