# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
36-Qubit Virtual Cut-Bridge & Heterogeneous Co-Processor Dynamics Visualization (Monograph M155)
Renders a 4-panel publication visual strictly adhering to the Deen OS Sovereign Visual Specification:
- Root Canvas: Deep Obsidian (#05080d)
- Card Panels: Void Charcoal (#0f172a)
- Quantum Telemetry: Laser Emerald Green (#00e676)
- Hardware & Clocks: Radiant Gold (#ffb300)
- Bounds & Thresholds: Crimson Red (#ef4444)
- Text & Readouts: Polar Slate (#f8fafc)
Zero cyan, zero purple, zero emojis.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_path = os.path.join("results", "data", "virtual_36q_cut_bridge_report.json")
    if not os.path.exists(json_path):
        print(f"Error: Telemetry file {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data["benchmarks"]
    layers = [b["circuit_layers"] for b in benchmarks]
    snap_ms = [b["checkpoint_prep_time_ms"] for b in benchmarks]
    branch_ms = [b["branch_evaluation_time_ms"] for b in benchmarks]
    total_ms = [b["total_execution_time_ms"] for b in benchmarks]
    speedups = [b["checkpoint_speedup_ratio"] for b in benchmarks]
    zz_corr = [b["cross_bridge_zz_correlator"] for b in benchmarks]
    xx_corr = [b["cross_bridge_xx_correlator"] for b in benchmarks]
    bell_s = [b["bell_chsh_parameter"] for b in benchmarks]
    entropy = [b["entanglement_entropy_nats"] for b in benchmarks]

    dense_ram_gb = data["dense_ram_bytes"] / (1024**3)
    active_ram_mb = data["active_ram_bytes"] / (1024**2)
    reduction_factor = data["ram_reduction_factor"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#05080d")

    # Panel 1: Memory Footprint Compression (1.00 TB vs 8.00 MB)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#0f172a")
    categories = ["Dense 36Q Statevector\n(Theoretical)", "Virtual Cut Dual-18Q\n(100% L3 Resident)"]
    mem_vals_mb = [dense_ram_gb * 1024.0, active_ram_mb]
    colors1 = ["#ef4444", "#00e676"]
    bars1 = ax1.bar(categories, mem_vals_mb, color=colors1, width=0.45, edgecolor="#1e293b", linewidth=1.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Memory Footprint (MB, Log Scale)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_title(f"36-Qubit Memory Collapse: {reduction_factor:,.0f}x L3 Compression", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax1.tick_params(colors="#e2e8f0", labelsize=10)

    for bar, val in zip(bars1, mem_vals_mb):
        if val > 100000:
            txt = f"{val/1024:,.0f} GB (1.00 TB)"
        else:
            txt = f"{val:.2f} MB"
        ax1.annotate(txt,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", color="#f8fafc", fontweight="bold", fontsize=10)

    # Panel 2: Latency Scaling & Checkpointing Speedup
    ax2 = axes[0, 1]
    ax2.set_facecolor("#0f172a")
    width = 0.28
    x_pos = np.arange(len(layers))
    b_snap = ax2.bar(x_pos - width/2, snap_ms, width=width, label="Snapshot Prep Time", color="#ffb300", edgecolor="#1e293b")
    b_branch = ax2.bar(x_pos + width/2, branch_ms, width=width, label="16-Branch Contraction", color="#00e676", edgecolor="#1e293b")
    ax2.plot(x_pos, total_ms, color="#f8fafc", marker="o", linewidth=2.0, label="Total Latency (ms)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Depth {l}" for l in layers], color="#e2e8f0", fontsize=10)
    ax2.set_ylabel("Execution Latency (ms)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_title("Mid-Circuit State Checkpointing Latency Scaling", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(loc="upper left", facecolor="#111827", edgecolor="#334155", fontsize=9, labelcolor="#f8fafc")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax2.tick_params(colors="#e2e8f0")

    # Panel 3: Non-Local Cross-Boundary Correlators (<Z17 Z0> and <X17 X0>)
    ax3 = axes[1, 0]
    ax3.set_facecolor("#0f172a")
    ax3.plot(layers, zz_corr, color="#00e676", marker="s", linewidth=2.2, label=r"$\langle Z_{17} \otimes Z_0 \rangle$ (Cross-Bridge)")
    ax3.plot(layers, xx_corr, color="#ffb300", marker="^", linewidth=2.2, label=r"$\langle X_{17} \otimes X_0 \rangle$ (Transverse)")
    ax3.plot(layers, entropy, color="#94a3b8", marker="d", linestyle="--", linewidth=1.8, label=r"Entanglement Entropy $S_{\mathrm{vN}}$")
    ax3.axhline(0.0, color="#64748b", linestyle=":", alpha=0.7)
    ax3.set_xlabel("Ansatz Variational Depth (Layers)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Correlator / Entropy Readout", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_title("Cross-Boundary Non-Local Quantum Correlators", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax3.legend(loc="upper right", facecolor="#111827", edgecolor="#334155", fontsize=9, labelcolor="#f8fafc")
    ax3.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax3.tick_params(colors="#e2e8f0")

    # Panel 4: Bell-CHSH Parameter S & Physical Bounds
    ax4 = axes[1, 1]
    ax4.set_facecolor("#0f172a")
    ax4.plot(layers, bell_s, color="#00e676", marker="o", linewidth=2.5, label="Measured Bell-CHSH Parameter S")
    ax4.axhline(2.0, color="#ef4444", linestyle="--", linewidth=1.8, label=r"Classical Local Realism Ceiling ($S \leq 2.0$)")
    ax4.axhline(2.0 * np.sqrt(2), color="#ffb300", linestyle="-.", linewidth=1.8, label=r"Tsirelson Bound ($2\sqrt{2} \approx 2.8284$)")
    ax4.set_xlabel("Ansatz Variational Depth (Layers)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Bell Parameter S", color="#f8fafc", fontsize=11, fontweight="bold")
    ax4.set_title("Bell-CHSH Witness Violation & Entanglement Bounds", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax4.set_ylim(-0.1, 3.2)
    ax4.legend(loc="upper right", facecolor="#111827", edgecolor="#334155", fontsize=9, labelcolor="#f8fafc")
    ax4.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax4.tick_params(colors="#e2e8f0")

    plt.suptitle("T-1000: 36-Qubit Virtual Cut-Bridge & Co-Processor Dynamics (Monograph M155)\nBipartite Quasiprobability Decomposition & 131,072x L3 Cache Compression",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_plot_dir = os.path.join("results", "plots")
    os.makedirs(out_plot_dir, exist_ok=True)
    out_plot_path = os.path.join(out_plot_dir, "virtual_36q_cut_bridge_dynamics.png")
    plt.savefig(out_plot_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Publication chart generated: {out_plot_path}")

if __name__ == "__main__":
    main()
