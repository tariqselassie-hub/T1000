#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Holographic Bulk Quantum Error Correction & Size-Winding Optimization Visualizer.
Plots 4 panels strictly adhering to the Deen OS Sovereign Visual Specification:
- Root Canvas: Deep Obsidian (#05080d)
- Card Panels: Void Charcoal (#0f172a)
- Quantum Signals: Laser Emerald (#00e676)
- Hardware / Gold: Radiant Amber (#ffb300)
- Alert / Threshold: Crimson Red (#ef4444)
- Gridlines: Muted Slate Grid (#1e293b)
- Typography: Polar Slate (#f8fafc)
Zero emojis. Zero prohibited colors (no cyan, purple, magenta).
"""

import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_BG_ROOT = "#05080d"
COLOR_BG_CARD = "#0f172a"
COLOR_EMERALD = "#00e676"
COLOR_AMBER = "#ffb300"
COLOR_CRIMSON = "#ef4444"
COLOR_GRID = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"

def load_telemetry(json_path: Path) -> dict:
    if not json_path.exists():
        # Fallback to scratch path if needed
        scratch_path = Path("target/bench-scratch/unknown-cpu") / json_path
        if scratch_path.exists():
            json_path = scratch_path
        else:
            raise FileNotFoundError(f"Telemetry file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def render_qec_charts(data: dict, output_png: Path):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=COLOR_BG_ROOT)
    fig.subplots_adjust(hspace=0.28, wspace=0.22)

    # -------------------------------------------------------------
    # Panel 1: SPSA Size-Winding Hamiltonian Learning Trajectory
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor(COLOR_BG_CARD)
    trajectory = data.get("trajectory", [])
    iters = [pt["iteration"] for pt in trajectory]
    fidelities = [pt["fidelity"] for pt in trajectory]
    objectives = [pt["objective"] for pt in trajectory]

    ax1.plot(iters, fidelities, color=COLOR_EMERALD, linewidth=2.4, label="Bell Process Fidelity F(g)")
    ax1.plot(iters, objectives, color=COLOR_AMBER, linewidth=1.8, linestyle="--", label="SPSA Objective (F + 1.5*Asym)")

    ax1.axhline(0.25, color=COLOR_CRIMSON, linestyle=":", linewidth=1.5, label="Uncoupled Classical Floor (0.25)")
    base_fid = data.get("baseline_fidelity", 0.25)
    opt_fid = data.get("optimized_fidelity", 0.5)
    ax1.scatter([0], [base_fid], color=COLOR_AMBER, s=80, zorder=5, label=f"Baseline: {base_fid:.4f}")
    ax1.scatter([iters[-1]], [opt_fid], color=COLOR_EMERALD, s=100, zorder=5, label=f"Optimized: {opt_fid:.4f}")

    ax1.set_title("Size-Winding Hamiltonian Learning (SPSA)", fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)
    ax1.set_xlabel("Optimization Iteration", fontsize=10, color=COLOR_TEXT)
    ax1.set_ylabel("Process Fidelity / Objective", fontsize=10, color=COLOR_TEXT)
    ax1.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax1.tick_params(colors=COLOR_MUTED)
    ax1.legend(loc="lower right", facecolor=COLOR_BG_ROOT, edgecolor=COLOR_GRID, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Holographic Bulk QEC - Boundary Erasure Curves
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor(COLOR_BG_CARD)
    left_pts = data["left_erasure"]["points"]
    right_pts = data["right_erasure"]["points"]

    left_fracs = [p["erasure_fraction"] * 100.0 for p in left_pts]
    left_fids = [p["recovered_fidelity"] for p in left_pts]
    right_fracs = [p["erasure_fraction"] * 100.0 for p in right_pts]
    right_fids = [p["recovered_fidelity"] for p in right_pts]

    ax2.plot(left_fracs, left_fids, marker="s", markersize=7, color=COLOR_EMERALD, linewidth=2.2, label="Left Horizon Erasure")
    ax2.plot(right_fracs, right_fids, marker="o", markersize=7, color=COLOR_AMBER, linewidth=2.2, linestyle="-.", label="Right Horizon Erasure")

    ax2.axhline(0.25, color=COLOR_CRIMSON, linestyle=":", linewidth=1.8, label="Entanglement Throat Pinch-Off (0.25)")
    ax2.fill_between(left_fracs, 0.25, left_fids, color=COLOR_EMERALD, alpha=0.15, label="Bulk Throat Open Zone")

    ax2.set_title("Holographic Bulk Recovery Under Boundary Erasures", fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)
    ax2.set_xlabel("Boundary Erasure Fraction (%)", fontsize=10, color=COLOR_TEXT)
    ax2.set_ylabel("Reconstructed Probe Fidelity", fontsize=10, color=COLOR_TEXT)
    ax2.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax2.tick_params(colors=COLOR_MUTED)
    ax2.legend(loc="upper right", facecolor=COLOR_BG_ROOT, edgecolor=COLOR_GRID, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Entanglement Wedge Reconstruction Phase Diagram
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor(COLOR_BG_CARD)
    erased_counts = [p["erased_count"] for p in left_pts]
    margins = [max(0.0, p["recovered_fidelity"] - 0.25) for p in left_pts]

    bars = ax3.bar(
        [f"{c} Erased\n({c/left_pts[-1]['erased_count']*100:.0f}%)" if left_pts[-1]['erased_count'] > 0 else f"{c}" for c in erased_counts],
        margins,
        color=[COLOR_EMERALD if m > 0.001 else COLOR_CRIMSON for m in margins],
        edgecolor=COLOR_GRID,
        width=0.45,
    )
    for bar, val in zip(bars, margins):
        if val > 0.001:
            ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.008, f"+{val:.4f}", ha="center", va="bottom", fontsize=9, color=COLOR_EMERALD, fontweight="bold")
        else:
            ax3.text(bar.get_x() + bar.get_width() / 2, 0.008, "COLLAPSED", ha="center", va="bottom", fontsize=8, color=COLOR_CRIMSON, fontweight="bold")

    ax3.set_title("Bulk Information Margin Above Classical Horizon", fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)
    ax3.set_xlabel("Boundary Loss Condition", fontsize=10, color=COLOR_TEXT)
    ax3.set_ylabel("Fidelity Excess: F(k) - 0.2500", fontsize=10, color=COLOR_TEXT)
    ax3.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID, axis="y")
    ax3.tick_params(colors=COLOR_MUTED)

    # -------------------------------------------------------------
    # Panel 4: Code Distance & Horizon Threshold Telemetry
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor(COLOR_BG_CARD)
    ax4.axis("off")

    num_modes = data["left_erasure"]["num_modes"]
    total_q = data["left_erasure"]["total_qubits"]
    d_holo = data["left_erasure"]["holographic_code_distance"]
    thresh = data["left_erasure"]["entanglement_wedge_threshold"] * 100.0
    gain_pct = (opt_fid - base_fid) / base_fid * 100.0

    telemetry_text = (
        "┌────────────────────────────────────────────────────────┐\n"
        "│ HOLOGRAPHIC QUANTUM ERROR CORRECTION METRICS           │\n"
        "├────────────────────────────────────────────────────────┤\n"
        f"│ Boundary Horizon Size:         {num_modes:2} modes / {total_q} qubits total\n"
        f"│ Unoptimized Raw Fidelity:      {base_fid:.4f} (Generic Chaos)\n"
        f"│ Size-Wound Bulk Fidelity:      {opt_fid:.4f} (Phase Aligned)\n"
        f"│ Optimization Fidelity Boost:   +{gain_pct:.2f}%\n"
        "├────────────────────────────────────────────────────────┤\n"
        "│ ALMHEIRI-DONG-HARLOW BULK RECONSTRUCTION (ADH):       │\n"
        f"│ Holographic Code Distance:     d_holo = {d_holo} modes\n"
        f"│ Entanglement Wedge Threshold:  {thresh:.1f}% boundary loss\n"
        "│ Sub-threshold Transmission:    Traversable (F > 0.2500)\n"
        "│ Super-threshold Horizon:       Pinched / Causally Disconnected\n"
        "└────────────────────────────────────────────────────────┘"
    )

    ax4.text(
        0.05, 0.5,
        telemetry_text,
        fontfamily="monospace",
        fontsize=10.5,
        color=COLOR_TEXT,
        va="center",
        bbox=dict(boxstyle="square,pad=0.8", facecolor=COLOR_BG_ROOT, edgecolor=COLOR_GRID)
    )

    fig.suptitle("T-1000 Sovereign Telemetry: Holographic Bulk QEC & Size-Winding Hamiltonian Learning",
                 fontsize=15, fontweight="bold", color=COLOR_TEXT, y=0.98)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=COLOR_BG_ROOT)
    plt.close()
    print(f"Chart successfully saved to: {output_png}")

if __name__ == "__main__":
    json_path = Path("results/experiments/holographic_qec.json")
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    out_png = Path("results/charts/holographic_qec_telemetry.png")
    if len(sys.argv) > 2:
        out_png = Path(sys.argv[2])

    data = load_telemetry(json_path)
    render_qec_charts(data, out_png)

