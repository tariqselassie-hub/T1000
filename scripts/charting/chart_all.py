# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Master Scientific Charting Suite Orchestrator for T-1000.
Executes all specialized charting modules and compiles visual research artifacts.
Strictly adheres to Deen OS Sovereign Visual Specification.
"""

from pathlib import Path
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure local modules are importable
current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from chart_mnist_qnn import chart_mnist_qnn
from chart_qnn_scaling import chart_qnn_scaling
from chart_lattice_phase import chart_lattice_phase
from chart_lattice_quench import chart_lattice_quench
from chart_frontier_rcs import chart_frontier_rcs
from chart_pc_vs_fpga import chart_pc_vs_fpga

def run_all_charts() -> None:
    ws_root = current_dir.parents[1]
    exp_dir = ws_root / "results" / "experiments"
    chart_dir = ws_root / "results" / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    print("╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║            T-1000 SCIENTIFIC TELEMETRY & SOVEREIGN CHARTING SUITE             ║")
    print("╚═══════════════════════════════════════════════════════════════════════════════╝")
    print(f" [DATA REPOSITORY] : {exp_dir}")
    print(f" [OUTPUT GALLERY]  : {chart_dir}\n")

    generated = []

    # 1. MNIST QNN Learning Dynamics
    mnist_csv = exp_dir / "mnist_qnn_training.csv"
    if mnist_csv.exists():
        png, _ = chart_mnist_qnn(mnist_csv, chart_dir)
        size_kb = png.stat().st_size / 1024.0
        generated.append(("MNIST QNN Optimization & Coherence", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")
    else:
        print(f" [!] Skipped MNIST QNN: {mnist_csv.name} not found.")

    # 2. 10-to-26 Qubit Statevector Scaling
    scaling_csv = exp_dir / "qnn_scaling_26_qubits.csv"
    if scaling_csv.exists():
        png = chart_qnn_scaling(scaling_csv, chart_dir)
        size_kb = png.stat().st_size / 1024.0
        generated.append(("10-to-26 Qubit SIMD Scaling", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")
    else:
        print(f" [!] Skipped Scaling: {scaling_csv.name} not found.")

    # 3. 2D Quantum Spin Lattice Phase Transition
    phase_json = exp_dir / "lattice_phase_transition.json"
    if phase_json.exists():
        png = chart_lattice_phase(phase_json, chart_dir)
        size_kb = png.stat().st_size / 1024.0
        generated.append(("2D Lattice Phase Transition", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")
    else:
        print(f" [!] Skipped Phase Transition: {phase_json.name} not found.")

    # 4. 2D Magnon Quench & Spin-Wave Dispersion
    quench_json = exp_dir / "lattice_quench_dynamics.json"
    if quench_json.exists():
        png = chart_lattice_quench(quench_json, chart_dir)
        size_kb = png.stat().st_size / 1024.0
        generated.append(("2D Magnon Quench Dynamics", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")
    else:
        print(f" [!] Skipped Quench Dynamics: {quench_json.name} not found.")

    # 5. 14-Qubit 36-Depth Frontier RCS Chaos & Telemetry
    rcs_json = exp_dir / "frontier_rcs_results.json"
    if rcs_json.exists():
        png = chart_frontier_rcs(rcs_json, chart_dir)
        size_kb = png.stat().st_size / 1024.0
        generated.append(("14-Qubit 36-Depth Frontier RCS", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")
    else:
        print(f" [!] Skipped Frontier RCS: {rcs_json.name} not found.")

    # 6. PC vs FPGA 26-Qubit Scaling
    pc_fpga_csv = exp_dir / "pc_vs_fpga_26q_scaling.csv"
    if pc_fpga_csv.exists():
        png = chart_pc_vs_fpga(pc_fpga_csv, chart_dir)
        size_kb = png.stat().st_size / 1024.0
        generated.append(("PC vs FPGA 26-Qubit Scaling", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")
    else:
        print(f" [!] Skipped PC vs FPGA Scaling: {pc_fpga_csv.name} not found.")

    print("\n╔═══════════════════════════════════════════════════════════════════════════════╗")
    print("║                        RENDERED ARTIFACT MANIFEST                             ║")
    print("╠═════════════════════════════════════╦══════════════════════════════════╦══════════╣")
    print("║ Research Experiment                 ║ Output File                      ║ Size     ║")
    print("╠═════════════════════════════════════╬══════════════════════════════════╬══════════╣")
    for exp_name, fname, fsize in generated:
        print(f"║ {exp_name:<35} ║ {fname:<32} ║ {fsize:>8} ║")
    print("╚═════════════════════════════════════╩══════════════════════════════════╩══════════╝\n")

if __name__ == "__main__":
    run_all_charts()
