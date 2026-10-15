# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes T-1000 26-Qubit FPGA Silicon Acceleration & Telemetry.
Target Silicon: AMD/Xilinx Artix-7 XC7A35T-1CPG236C (Digilent Basys 3).
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

def chart_fpga_26q(json_path: Path, output_dir: Path) -> Path:
    apply_deen_theme()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 1000_QPU // ARTIX-7 XC7A35T 26-QUBIT SILICON ACCELERATOR TELEMETRY",
        color=TEXT_POLAR, fontsize=12, fontweight="bold", y=0.98
    )

    # -------------------------------------------------------------------------
    # Panel 1: Multi-Scale Execution Latency Hierarchy (Log Scale)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    stages = [
        "14Q Chunk\n(128 KB BRAM)",
        "26Q Layer\n(4,096 Chunks)",
        "12-Layer Run\n(Total Silicon)",
        "Host UART Link\n(COM6 Ping)"
    ]
    latencies_ms = [
        data["silicon_per_layer_ms"] / data["num_chunks_128kb"],  # 0.01024 ms (10.24 us)
        data["silicon_per_layer_ms"],                             # 41.943 ms
        data["silicon_total_ms"],                                 # 503.316 ms
        0.964                                                     # Host ping (0.964 ms)
    ]
    colors = [GREEN_QUANTUM, GOLD_HARDWARE, GOLD_BRIGHT, TEXT_MUTED]

    bars1 = ax1.bar(stages, latencies_ms, color=colors, width=0.55, edgecolor=GRID_SLATE, alpha=0.9)
    ax1.set_yscale("log")
    ax1.set_ylabel("Execution Time (ms, log scale)", color=TEXT_MUTED)

    # Annotations
    ax1.annotate(f"{latencies_ms[0]*1000:.2f} us\n(1,024 cycles)", xy=(0, latencies_ms[0]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GREEN_EMERALD, fontsize=8, fontweight="bold")
    ax1.annotate(f"{latencies_ms[1]:.2f} ms\n(67M states)", xy=(1, latencies_ms[1]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GOLD_BRIGHT, fontsize=8, fontweight="bold")
    ax1.annotate(f"{latencies_ms[2]:.1f} ms\n(0.503 s)", xy=(2, latencies_ms[2]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GOLD_HARDWARE, fontsize=8, fontweight="bold")
    ax1.annotate(f"{latencies_ms[3]:.3f} ms\n(JTAG/UART)", xy=(3, latencies_ms[3]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=TEXT_POLAR, fontsize=8)

    style_card(ax1, "PANEL 1: MULTI-SCALE LATENCY HIERARCHY", "128 KB Chunk -> 26Q Full Layer -> 12-Layer Circuit")

    # -------------------------------------------------------------------------
    # Panel 2: Compute Density & Energy Consumption (Hardware Telemetry)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    # Dual-metric normalized radar or bar comparison
    metrics = ["Throughput\n(M ops/s)", "Total Ops\n(x10^7)", "Power\n(Watts)", "Energy/Op\n(nJ)"]
    vals = [
        data["butterfly_throughput_ops_sec"] / 1e6,       # 800 M ops/sec
        data["total_butterfly_operations"] / 1e7,          # 40.26 x10^7 ops
        data["power_watts"],                               # 0.254 W
        data["energy_nanojoules_per_op"]                   # 0.317 nJ
    ]
    colors2 = [GOLD_HARDWARE, GREEN_QUANTUM, RED_ALERT, GREEN_EMERALD]

    bars2 = ax2.bar(metrics, vals, color=colors2, width=0.55, edgecolor=GRID_SLATE, alpha=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("Metric Value (Log Scale)", color=TEXT_MUTED)

    ax2.annotate(f"800.0 M/s\n(8 Lanes)", xy=(0, vals[0]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GOLD_BRIGHT, fontsize=8, fontweight="bold")
    ax2.annotate(f"40.26 x10^7\n(402.6M)", xy=(1, vals[1]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GREEN_QUANTUM, fontsize=8, fontweight="bold")
    ax2.annotate(f"0.254 W\n(Artix-7)", xy=(2, vals[2]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=RED_ALERT, fontsize=8, fontweight="bold")
    ax2.annotate(f"0.317 nJ\n(per op)", xy=(3, vals[3]), xytext=(0, 6),
                 textcoords="offset points", ha="center", va="bottom", color=GREEN_EMERALD, fontsize=8, fontweight="bold")

    style_card(ax2, "PANEL 2: SILICON COMPUTE DENSITY", "800 Mops/s Throughput @ 0.254 W Dynamic Power")

    # -------------------------------------------------------------------------
    # Panel 3: 26-Qubit Porter-Thomas Chaos Convergence
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    chaos_names = ["Linear XEB\n(F_XEB)", "Haar Fidelity\n(Scrambling)", "Peak Mult.\n(Ratio / 20)", "Entropy\n(nats / 20)"]
    chaos_vals = [
        data["f_xeb"],
        data["haar_fidelity"],
        data["peak_multiplier"] / 20.0,
        data["shannon_entropy_nats"] / 20.0
    ]
    raw_labels = [
        f"{data['f_xeb']:.4f}\n(PT limit)",
        f"{data['haar_fidelity']:.4f}\n(Haar)",
        f"{data['peak_multiplier']:.2f}x\n(Peaking)",
        f"{data['shannon_entropy_nats']:.2f} nats\n(ln(2^26)-γ)"
    ]

    bars3 = ax3.bar(chaos_names, chaos_vals, color=[GREEN_EMERALD, GREEN_QUANTUM, GOLD_HARDWARE, GOLD_BRIGHT], width=0.55, edgecolor=GRID_SLATE, alpha=0.9)
    ax3.axhline(1.0, color=TEXT_POLAR, ls="--", lw=1.2, alpha=0.6, label="Ideal Haar-Random Reference (1.00)")
    ax3.set_ylim([0.0, 1.25])
    ax3.set_ylabel("Normalized Chaos Score", color=TEXT_MUTED)

    for i, (b, label) in enumerate(zip(bars3, raw_labels)):
        ax3.annotate(label, xy=(b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 5),
                     textcoords="offset points", ha="center", va="bottom", color=TEXT_POLAR, fontsize=8, fontweight="bold")

    style_card(ax3, "PANEL 3: 26-QUBIT CHAOS CONVERGENCE", "Linear Cross-Entropy & Shannon Entropy (PT Limit)")
    ax3.legend(loc="lower right")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

    png_path = output_dir / "fpga_26q_telemetry.png"
    save_deen_figure(fig, png_path)
    return png_path

if __name__ == "__main__":
    ws_root = Path(__file__).resolve().parents[2]
    json_file = ws_root / "results" / "experiments" / "fpga_26q_scaling_telemetry.json"
    out_dir = ws_root / "results" / "charts"
    if json_file.exists():
        png = chart_fpga_26q(json_file, out_dir)
        print(f" [+] Generated 26Q FPGA Telemetry Chart: {png}")
    else:
        print(f" [-] Missing data file: {json_file}")
