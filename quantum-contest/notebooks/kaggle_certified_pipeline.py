# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
"""
Kaggle Notebook Execution Pipeline for Google Quantum AI Challenge.
Self-contained script to synthesize certified circuits and package submission.zip.
"""

import sys, os, shutil, zipfile, importlib.util
from pathlib import Path
import numpy as np

# 1. Clean working directory
for p in Path("/kaggle/working").glob("*"):
    if p.is_file():
        p.unlink()
    elif p.is_dir() and p.name != ".virtual_documents":
        shutil.rmtree(p)
print("[+] Cleaned working directory.")

# 2. Locate competition files and link kickmix binary
possible_roots = [
    Path("/kaggle/input/competitions/google-verifiable-quantum-circuit-compilation"),
    Path("/kaggle/input/google-verifiable-quantum-circuit-compilation"),
]
comp_dir = next(r for r in possible_roots if (r / "kickmix" / "kickmix.so").exists())

so_path = comp_dir / "kickmix" / "kickmix.so"
spec = importlib.util.spec_from_file_location("kickmix", str(so_path))
km = importlib.util.module_from_spec(spec)
sys.modules["kickmix"] = km
spec.loader.exec_module(km)

if str(comp_dir / "verifiers") not in sys.path:
    sys.path.insert(0, str(comp_dir / "verifiers"))

from invsqrt import nr_reference_circuit, verify_submission_invsqrt
nr_reference_circuit.km = km
verify_submission_invsqrt.km = km

from invsqrt.nr_reference_circuit import Precision, build_nr_circuit
from invsqrt.verify_submission_invsqrt import (
    verify as verify_invsqrt,
    ProblemSpecInvSqrt,
    reference_values,
    validate_registers,
    half_turns_to_units,
    to_bits,
)

# 3. Synthesize certified circuit (Cand5: 255 qubits, 7470.4 Toffolis)
print("[+] Synthesizing certified Cand5 invsqrt circuit...")
cand5_prec = Precision(
    delta_bits=12,
    coef_bits=17,
    square_bits=27,
    product_bits=27,
    result_bits=28,
)
circuit = build_nr_circuit(10, prec=cand5_prec)
opt_invsqrt_path = Path("/kaggle/working/invsqrt.kmb")
circuit.to_file(str(opt_invsqrt_path), format="kmb")
print(f"[+] Wrote certified circuit to {opt_invsqrt_path}")

# 4. Specifically test the grader boundary point: x = 2684415
spec_prob = ProblemSpecInvSqrt.for_n_p(10)
target, tol = reference_values(spec_prob, np.array([2684415], dtype=np.uint64))
x_reg, scratch = validate_registers(circuit, spec_prob)
sim = km.Simulator(batch_size=1, counting_operations=False, seed=42)
sim.write_across_shots(x_reg, to_bits(np.array([2684415], dtype=np.uint64), spec_prob.width))
sim.do(circuit)
theta = half_turns_to_units(sim.read_shot_phase(0))

# Safe modular phase distance mod 2^64 (zero numpy scalar underflow warnings)
raw_diff = int(theta) - int(target[0])
mod_diff = ((raw_diff + (1 << 63)) % (1 << 64)) - (1 << 63)
err = abs(mod_diff)

passed_edge = err <= tol[0]
print(f"[+] Edge Check (x=2684415): Error={err/2**64:.3e} | Allowed={tol[0]/2**64:.3e} | PASS={passed_edge}")
assert passed_edge, "Edge case x=2684415 failed"

# 5. Fast batch verification (256 shots)
print("[+] Running batch verification test...")
res = verify_invsqrt(
    circuit,
    ProblemSpecInvSqrt(width=22, max_x=256, beta=1, precision_bits=24, shots_per_x=1, system_qubits=6540),
    max_workers=4,
)
print(f"[+] Batch Result -> Status: {res['status']} | Qubits: {res['num_qubits']} | Toffolis: {res['toffoli_equivalent']:.1f} | Score: {res['score']:.2f}")

# 6. Package final submission (Dual-Format Output)
sample_dir = comp_dir / "sample_submission"

# Copy unzipped files directly to /kaggle/working/
shutil.copy(sample_dir / "dqi.kmb", "/kaggle/working/dqi.kmb")
shutil.copytree(sample_dir / "dfthcsos", "/kaggle/working/dfthcsos", dirs_exist_ok=True)

# Build submission.zip alongside
sub_zip_path = Path("/kaggle/working/submission.zip")
print(f"[+] Packaging final submission to {sub_zip_path}...")
with zipfile.ZipFile(sub_zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
    zf.write(opt_invsqrt_path, arcname="invsqrt.kmb")
    zf.write(sample_dir / "dqi.kmb", arcname="dqi.kmb")
    dfthc_dir = sample_dir / "dfthcsos"
    for f in dfthc_dir.glob("angle_k_*.kmb"):
        zf.write(f, arcname=f"dfthcsos/{f.name}")
    zf.write(dfthc_dir / "submission.npz", arcname="dfthcsos/submission.npz")

print("\n=== WORKING DIRECTORY AUDIT ===")
for root, dirs, files in os.walk("/kaggle/working"):
    rel = os.path.relpath(root, "/kaggle/working")
    prefix = "" if rel == "." else rel + "/"
    for f in files:
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {prefix}{f} ({sz:.2f} MB)")
print("================================")
print("[+] All files exposed and ready for submission!")
