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
from chart_exascale_showdown import chart_exascale_showdown

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

    # 9. Search 3: ZINC Quantum Chemical Anomalies
    anom_json = exp_dir / "zinc_quantum_anomalies.json"
    if anom_json.exists():
        import chart_zinc_anomalies
        chart_zinc_anomalies.main()
        png = chart_dir / "zinc_quantum_anomalies.png"
        size_kb = png.stat().st_size / 1024.0
        generated.append(("Search 3: Quantum Anomalies", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 10. Search 1: ZINC Frontier Pareto Drug Leads
    pareto_json = exp_dir / "zinc_pareto_leads.json"
    if pareto_json.exists():
        import chart_zinc_pareto
        chart_zinc_pareto.main()
        png = chart_dir / "zinc_pareto_leads.png"
        size_kb = png.stat().st_size / 1024.0
        generated.append(("Search 1: Frontier Pareto Leads", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 11. MIMAC Search A: Quantum Activity Cliffs & Magic Switches
    mimac_json = exp_dir / "mimac_quantum_activity_cliffs.json"
    if mimac_json.exists():
        import chart_mimac_activity_cliffs
        chart_mimac_activity_cliffs.main()
        png = chart_dir / "mimac_quantum_activity_cliffs.png"
        size_kb = png.stat().st_size / 1024.0
        generated.append(("MIMAC: Quantum Activity Cliffs", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 12. Lead #109 Attractor Analysis (Convergence of 3 Magic Switches)
    lead_json = exp_dir / "lead_109_convergence_analysis.json"
    if lead_json.exists():
        import chart_lead_109_attractor
        chart_lead_109_attractor.main()
        png = chart_dir / "lead_109_attractor_analysis.png"
        size_kb = png.stat().st_size / 1024.0
        generated.append(("Lead #109 Attractor Convergence", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 13. T-1000 Desktop vs Exascale Supercomputer Showdown
    png = chart_exascale_showdown(chart_dir)
    size_kb = png.stat().st_size / 1024.0
    generated.append(("Exascale Desktop Showdown", png.name, f"{size_kb:.1f} KB"))
    print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 14. IBM Quantum Nighthawk R2 & Kagome Embedding
    import chart_nighthawk_layout
    chart_nighthawk_layout.create_nighthawk_visualization()
    png = chart_dir / "nighthawk_r2_kagome_layout.png"
    size_kb = png.stat().st_size / 1024.0
    generated.append(("Nighthawk R2 & Kagome Layout", png.name, f"{size_kb:.1f} KB"))
    print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 15. Waterbirds 11k Robustness & Quantum Sorter Telemetry
    hybrid_report = ws_root / "data" / "birds" / "birds_hybrid_qnn_report.json"
    if hybrid_report.exists():
        import chart_birds_hybrid
        chart_birds_hybrid.generate_telemetry_chart()
        png = chart_dir / "waterbirds_hybrid_qnn_telemetry.png"
        size_kb = png.stat().st_size / 1024.0
        generated.append(("Waterbirds Quantum Sorter Telemetry", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 16. Ultrametric Tree Sorter (QMUX) Architecture & Telemetry
    tree_report = exp_dir / "ultrametric_tree_sorter_report.json"
    if tree_report.exists():
        import chart_ultrametric_tree
        chart_ultrametric_tree.generate_telemetry_chart()
        png = chart_dir / "ultrametric_tree_sorter_telemetry.png"
        size_kb = png.stat().st_size / 1024.0
        generated.append(("Ultrametric Tree QMUX Telemetry", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 17. Childs et al. Glued Trees Exponential Speedup Telemetry
    glued_report = exp_dir / "glued_trees_report.json"
    if glued_report.exists():
        import chart_glued_trees
        chart_glued_trees.generate_telemetry_chart()
        png = chart_dir / "glued_trees_speedup_telemetry.png"
        size_kb = png.stat().st_size / 1024.0
        generated.append(("Glued Trees Exponential Speedup", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 18. Multi-Exit Topological Glued-Tree Valve & Quantum Router Telemetry
    valve_report = exp_dir / "glued_tree_valve_report.json"
    if valve_report.exists():
        import chart_glued_tree_valve
        chart_glued_tree_valve.generate_telemetry_chart()
        png = chart_dir / "glued_tree_valve_telemetry.png"
        size_kb = png.stat().st_size / 1024.0
        generated.append(("Glued-Tree Valve & Router", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 19. Cascaded Multi-Tier Topological Glued-Tree Valve Telemetry
    cascaded_report = exp_dir / "cascaded_tree_valve_report.json"
    if cascaded_report.exists():
        import chart_cascaded_tree_valve
        chart_cascaded_tree_valve.generate_telemetry_chart()
        png = chart_dir / "cascaded_tree_valve_telemetry.png"
        size_kb = png.stat().st_size / 1024.0
        generated.append(("Cascaded Multi-Tier Valve", png.name, f"{size_kb:.1f} KB"))
        print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 20. RCS Physical Noise Chaos & Coherence Transition Telemetry
    rcs_noise_json = ws_root / "results" / "rcs_physical_noise_chaos.json"
    if rcs_noise_json.exists():
        import chart_rcs_coherence_chaos
        chart_rcs_coherence_chaos.main()
        png = ws_root / "assets" / "rcs_physical_noise_chaos.png"
        if png.exists():
            size_kb = png.stat().st_size / 1024.0
            generated.append(("RCS Physical Noise Chaos", png.name, f"{size_kb:.1f} KB"))
            print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 21. Continuous Topological Walk Sorter Showdown Telemetry
    sorter_json = ws_root / "results" / "topological_walk_sorter_telemetry.json"
    if sorter_json.exists():
        import chart_topological_walk_sorter
        chart_topological_walk_sorter.main()
        png = ws_root / "assets" / "topological_walk_sorter_showdown.png"
        if png.exists():
            size_kb = png.stat().st_size / 1024.0
            generated.append(("Topological Walk Sorter Showdown", png.name, f"{size_kb:.1f} KB"))
            print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 22. ZINC Topological Walk Molecular Screening Telemetry
    zinc_screen_json = ws_root / "results" / "zinc_topological_walk_telemetry.json"
    if zinc_screen_json.exists():
        import chart_zinc_topological_screen
        chart_zinc_topological_screen.main()
        png = ws_root / "assets" / "zinc_topological_walk_screen.png"
        if png.exists():
            size_kb = png.stat().st_size / 1024.0
            generated.append(("ZINC Topological Walk Screening", png.name, f"{size_kb:.1f} KB"))
            print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 23. Orthogonal K-Patch Proxy RCS Showdown Telemetry
    showdown_json = ws_root / "results" / "orthogonal_proxy_showdown_telemetry.json"
    if showdown_json.exists():
        import chart_orthogonal_proxy_showdown
        chart_orthogonal_proxy_showdown.main()
        png = ws_root / "assets" / "orthogonal_proxy_showdown.png"
        if png.exists():
            size_kb = png.stat().st_size / 1024.0
            generated.append(("Orthogonal Proxy RCS Showdown", png.name, f"{size_kb:.1f} KB"))
            print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

    # 24. Lie-Algebraic Geometric QGNN Telemetry
    qgnn_json = ws_root / "results" / "data" / "qgnn_molecular_telemetry.json"
    if qgnn_json.exists():
        import chart_qgnn_showdown
        chart_qgnn_showdown.main()
        png = ws_root / "results" / "charts" / "qgnn_geometric_showdown.png"
        if png.exists():
            size_kb = png.stat().st_size / 1024.0
            generated.append(("Geometric QGNN Showdown", png.name, f"{size_kb:.1f} KB"))
            print(f" [+] Rendered: {png.name} ({size_kb:.1f} KB)")

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
