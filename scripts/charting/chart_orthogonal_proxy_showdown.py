# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizer for Orthogonal Proxy RCS Showdown: PyQrack vs T-1000.
Strictly conforms to Deen OS Sovereign Visual Specification.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from deen_theme import (
    apply_deen_theme, save_deen_figure,
    BG_OBSIDIAN, BG_DARK, CARD_CHARCOAL, GRID_SLATE,
    GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_SLATE, TEXT_MUTED
)

def main():
    apply_deen_theme()

    telemetry_path = Path("results/orthogonal_proxy_showdown_telemetry.json")
    if not telemetry_path.exists():
        print(f"Error: {telemetry_path} not found.")
        return

    with open(telemetry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qrack = data["qrack"]
    t1000 = data["t1000"]
    comp = data["comparative"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor(BG_OBSIDIAN)

    # -------------------------------------------------------------------------
    # Panel 1: Execution Latency (Log Scale)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor(CARD_CHARCOAL)

    categories = ["Proxy Sim & Sampling", "Band Ideal Probability"]
    qrack_times = [qrack.get("ace_proxy_seconds", 9.35), qrack.get("ideal_patches_seconds", 0.0028)]
    t1000_times = [t1000.get("sampling_latency_secs", 0.000795), t1000.get("ideal_patches_latency_secs", 0.000026)]

    x = np.arange(len(categories))
    width = 0.35

    b1 = ax1.bar(x - width/2, qrack_times, width, label="PyQrack (ACE/OpenCL)", color=RED_ALERT, alpha=0.9)
    b2 = ax1.bar(x + width/2, t1000_times, width, label="T-1000 (AVX2 L1/L2)", color=GREEN_QUANTUM, alpha=0.9)

    ax1.set_yscale("log")
    ax1.set_ylabel("Execution Time (Seconds, Log Scale)", color=TEXT_POLAR, fontsize=10)
    ax1.set_title("Runtime Latency Comparison", color=TEXT_POLAR, fontsize=11, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, color=TEXT_POLAR, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3, color=GRID_SLATE)
    ax1.legend(loc="upper right", framealpha=0.4, facecolor=BG_DARK, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8)

    # Annotate speedup ratios
    speedups = [comp["proxy_speedup_x"], comp["ideal_speedup_x"]]
    for i, s in enumerate(speedups):
        ax1.text(x[i] + width/2, t1000_times[i] * 2.5, f"{s:.0f}x\nFaster",
                 ha="center", va="bottom", color=GREEN_EMERALD, fontsize=8, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 2: Cross-Entropy Benchmarking (XEB) & HOG
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor(CARD_CHARCOAL)

    metrics_names = ["Factorized XEB", "HOG Probability"]
    qrack_scores = [qrack.get("xeb_proxy", -0.076), qrack.get("hog_proxy", 0.417)]
    t1000_scores = [t1000.get("xeb_proxy", 0.999), t1000.get("hog_proxy", 0.975)]

    x2 = np.arange(len(metrics_names))
    b3 = ax2.bar(x2 - width/2, qrack_scores, width, label="PyQrack ACE (Approximation)", color=GOLD_HARDWARE, alpha=0.85)
    b4 = ax2.bar(x2 + width/2, t1000_scores, width, label="T-1000 Engine (Exact)", color=GREEN_QUANTUM, alpha=0.9)

    ax2.axhline(0.0, color=TEXT_MUTED, linestyle="-", linewidth=0.8)
    ax2.set_ylabel("Score / Fidelity", color=TEXT_POLAR, fontsize=10)
    ax2.set_title("Fidelity Across Orthogonal Seams", color=TEXT_POLAR, fontsize=11, fontweight="bold", pad=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_names, color=TEXT_POLAR, fontsize=9)
    ax2.set_ylim(-0.25, 1.15)
    ax2.grid(True, linestyle="--", alpha=0.3, color=GRID_SLATE)
    ax2.legend(loc="upper left", framealpha=0.4, facecolor=BG_DARK, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8)

    for bar in b3:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (0.04 if yval >= 0 else -0.07), f"{yval:.3f}",
                 ha="center", va="bottom", color=GOLD_HARDWARE, fontsize=8, fontweight="bold")

    for bar in b4:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.04, f"{yval:.3f}",
                 ha="center", va="bottom", color=GREEN_EMERALD, fontsize=8, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: Orthogonal Grid Seam Topology
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor(CARD_CHARCOAL)

    # 3x4 grid representation
    grid = np.zeros((3, 4))
    grid[0, :] = 1.0  # Band 0 (4 qubits)
    grid[1:, :] = 2.0 # Band 1 (8 qubits)

    im = ax3.imshow(grid, cmap="viridis", aspect="auto", alpha=0.6)
    
    # Draw ACE vertical cuts (columns)
    ax3.axvline(0.5, color=RED_ALERT, linestyle="--", linewidth=2.5, label="ACE Seam 1 (Col 1)")
    ax3.axvline(2.5, color=RED_ALERT, linestyle="--", linewidth=2.5, label="ACE Seam 2 (Col 3)")
    
    # Draw Orthogonal horizontal proxy band cuts (rows)
    ax3.axhline(0.5, color=GOLD_BRIGHT, linestyle="-", linewidth=3.0, label="Proxy Cut (Orthogonal Band)")

    for r in range(3):
        for c in range(4):
            q_id = r * 4 + c
            ax3.text(c, r, f"Q{q_id}", ha="center", va="center", color=TEXT_POLAR, fontsize=10, fontweight="bold")

    ax3.set_title("Orthogonal Seam Topology (3 x 4 Grid)", color=TEXT_POLAR, fontsize=11, fontweight="bold", pad=12)
    ax3.set_xlabel("Lattice Columns (Cut by ACE)", color=TEXT_POLAR, fontsize=10)
    ax3.set_ylabel("Lattice Rows (Cut by Proxy)", color=TEXT_POLAR, fontsize=10)
    ax3.legend(loc="lower right", framealpha=0.5, facecolor=BG_DARK, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=7)

    plt.tight_layout()
    out_asset = Path("assets/orthogonal_proxy_showdown.png")
    save_deen_figure(fig, out_asset)
    print(f"Figure saved to {out_asset}")

if __name__ == "__main__":
    main()
