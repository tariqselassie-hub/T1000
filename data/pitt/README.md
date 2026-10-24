# University of Pittsburgh Quantum Repository (PM7)

`pitt_quantum.csv` is extracted from the Hutchison Lab Pitt Quantum Repository (`pitt-quantum-repository.zip`, 106,066 molecules).

## Provenance & Overview
- **Source:** University of Pittsburgh Quantum Repository (Geoffrey Hutchison Lab).
- **Calculated Quantum Method:** PM7 Semi-Empirical Quantum Mechanics.
- **Archive:** `C:\Users\zoddj\Models\pitt-quantum-repository.zip` (`Pitt_Quantum_Repository_Data.json`).
- **Normalized Table:** `data/pitt/pitt_quantum.csv` (105,926 valid SMILES records, 35.2 MB).
- **Summary Metrics:** `data/pitt/pitt_quantum_summary.json`.

## Physical & Topological Coverage
Unlike QM9 (which is restricted to <= 9 heavy atoms and 5 elements), the Pitt Quantum Repository represents real-world medicinal chemistry:
- **Molecular Weight:** 2.0 to 2,857.1 Da (Median: 281.3 Da, Mean: 304.2 Da).
- **Heteroatom Diversity:**
  - Sulfur: 17,908 molecules
  - Chlorine: 8,446 molecules
  - Phosphorus: 7,993 molecules
  - Fluorine: 6,965 molecules
  - Bromine: 2,533 molecules
  - Iodine: 1,268 molecules
- **Point Groups:** C1 (95,250), Cs (7,203), C2 (1,295), C2v (872), C2h (286), D2h (128).

## Extracted Columns
1. `inchikey`: Unique InChI key identifier.
2. `smiles`: Canonical OpenSMILES representation.
3. `formula`: Molecular formula.
4. `molwt`: Molecular weight (g/mol).
5. `point_group`: Molecular symmetry point group.
6. `homo`: Highest Occupied Molecular Orbital energy (eV).
7. `lumo`: Lowest Unoccupied Molecular Orbital energy (eV).
8. `gap`: Frontier orbital gap (eV) [LUMO - HOMO].
9. `dipole`: Scalar dipole moment (Debye).
10. `dipole_x`, `dipole_y`, `dipole_z`: 3D Cartesian dipole vector components (Debye).
11. `polarizability`: Mean scalar polarizability alpha (Bohr^3 / A^3).
12. `heat_of_formation`: Heat of formation (kJ/mol).
13. `surface_area`: Solvent-accessible surface area (A^2).
14. `volume`: Molecular van der Waals volume (A^3).
15. `rot_a`, `rot_b`, `rot_c`: Rotational constants.
16. `n_orbitals`: Count of calculated molecular orbital eigenvalues.
17. `tags`: Structural classification tags (alkene, amide, ring, etc.).
18. `name`: Common or IUPAC chemical name.

## Extraction Script
To re-extract or update the table:
```bash
python scripts/data/extract_pitt_quantum.py --zip-path "C:\Users\zoddj\Models\pitt-quantum-repository.zip" --out-dir data/pitt
```
