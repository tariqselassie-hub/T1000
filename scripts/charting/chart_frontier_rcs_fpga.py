# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes 14-Qubit 36-Depth Frontier RCS: Host CPU (AVX2) vs FPGA Artix-7 Silicon.
Adheres strictly to Deen OS Sovereign Visual Specification.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_MUTED, GRID_SLATE
)

def chart_frontier_rcs_fpga(json_path: Path, output_dir: Path) -> Path:
    apply_deen_theme()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 1000_QPU // 14-QUBIT 36-DEPTH FRONTIER RCS: HOST CPU VS ARTIX-7 SILICON",
        color=TEXT_POLAR, fontsize=12, fontweight="bold", y=0.98
    )

    # -------------------------------------------------------------------------
    # Panel 1: Execution Latency (CPU vs FPGA Wall vs FPGA Pure Silicon)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    bars_x = ["Host CPU (AVX2)\nCore Ultra 7", "FPGA Total Wall\n(COM6 @ 3MBaud)", "FPGA Pure Silicon\n(100 MHz, 32 DSP)"]
    times_ms = [data["pc_cpu_36d_ms"], data["fpga_wall_mean_ms"], data["fpga_silicon_36d_ms"]]
    bar_colors = [RED_ALERT, GOLD_BRIGHT, GREEN_QUANTUM]

    bars1 = ax1.bar(bars_x, times_ms, color=bar_colors, width=0.55, edgecolor=GRID_SLATE, alpha=0.9)
    ax1.set_yscale("log")
    ax1.set_ylabel("Execution Time (ms, log scale)", color=TEXT_MUTED)

    ax1.annotate(f"{times_ms[0]:.2f} ms\n(Baseline)", xy=(0, times_ms[0]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=RED_ALERT, fontsize=8, fontweight="bold")
    ax1.annotate(f"{times_ms[1]:.3f} ms\n(59.4x Faster)", xy=(1, times_ms[1]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GOLD_BRIGHT, fontsize=8, fontweight="bold")
    ax1.annotate(f"{times_ms[2]:.4f} ms\n(169.1x Faster)", xy=(2, times_ms[2]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GREEN_EMERALD, fontsize=8, fontweight="bold")

    style_card(ax1, "PANEL 1: 36-DEPTH CIRCUIT LATENCY", "14 Qubits | 16,384 Amplitudes | 1,008 Gates")

    # -------------------------------------------------------------------------
    # Panel 2: Compute Acceleration & Gate Throughput
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    metrics = ["Throughput\n(Gates / Sec)", "Raw Silicon\nSpeedup", "End-to-End\nSpeedup"]
    vals = [data["fpga_gate_throughput"] / 1000.0, data["raw_silicon_speedup"], data["wall_speedup"]]
    colors2 = [GOLD_HARDWARE, GREEN_EMERALD, GREEN_QUANTUM]

    bars2 = ax2.bar(metrics, vals, color=colors2, width=0.55, edgecolor=GRID_SLATE, alpha=0.9)
    ax2.set_ylabel("Metric Value", color=TEXT_MUTED)

    ax2.annotate(f"{vals[0]:,.0f} kGates/s\n(59x Throughput)", xy=(0, vals[0]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GOLD_BRIGHT, fontsize=8, fontweight="bold")
    ax2.annotate(f"{vals[1]:.1f}x\n(Compute Core)", xy=(1, vals[1]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GREEN_EMERALD, fontsize=8, fontweight="bold")
    ax2.annotate(f"{vals[2]:.1f}x\n(Total System)", xy=(2, vals[2]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GREEN_QUANTUM, fontsize=8, fontweight="bold")

    style_card(ax2, "PANEL 2: ACCELERATION & THROUGHPUT", "Hardware Gate Execution Rate & Speedup Multiplier")

    # -------------------------------------------------------------------------
    # Panel 3: Power Consumption & Energy Footprint
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    power_labels = ["Host CPU Active\n(Core Ultra 7)", "FPGA Dynamic\n(Artix-7 1000)", "Energy/Circuit\n(CPU: 6.2 J)", "Energy/Circuit\n(FPGA: 0.09 mJ)"]
    power_vals = [100.0, data["power_watts"], 6.23, 0.0000936 * 1000.0]  # mJ for comparison
    colors3 = [RED_ALERT, GREEN_QUANTUM, RED_ALERT, GREEN_EMERALD]

    bars3 = ax3.bar(power_labels, power_vals, color=colors3, width=0.55, edgecolor=GRID_SLATE, alpha=0.9)
    ax3.set_yscale("log")
    ax3.set_ylabel("Power (W) / Energy (mJ) [Log Scale]", color=TEXT_MUTED)

    ax3.annotate("100.0 W", xy=(0, power_vals[0]), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", color=RED_ALERT, fontsize=8, fontweight="bold")
    ax3.annotate("0.254 W", xy=(1, power_vals[1]), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", color=GREEN_QUANTUM, fontsize=8, fontweight="bold")
    ax3.annotate("6,230 mJ", xy=(2, power_vals[2]), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", color=RED_ALERT, fontsize=8, fontweight="bold")
    ax3.annotate("0.0936 mJ\n(66,550x Less)", xy=(3, power_vals[3]), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", color=GREEN_EMERALD, fontsize=8, fontweight="bold")

    style_card(ax3, "PANEL 3: SUSTAINABILITY & POWER DENSITY", "66,550x Energy Reduction per 36-Depth Quantum Circuit")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

    png_path = output_dir / "frontier_rcs_fpga_comparison.png"
    save_deen_figure(fig, png_path)
    return png_path

if __name__ == "__main__":
    ws_root = Path(__file__).resolve().parents[2]
    json_file = ws_root / "results" / "experiments" / "frontier_rcs_fpga_results.json"
    out_dir = ws_root / "results" / "charts"
    if json_file.exists():
        png = chart_frontier_rcs_fpga(json_file, out_dir)
        print(f" [+] Generated Frontier RCS FPGA Chart: {png}")
    else:
        print(f" [-] Missing data file: {json_file}")
