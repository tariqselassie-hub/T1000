# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visual Telemetry: Childs et al. Glued Trees Exponential Quantum Speedup.
Illustrates continuous-time quantum walk (CTQW) ballistic wavepacket traversal
versus classical random walk exponential trapping across welded binary tree graphs.
Strictly adheres to Deen OS Sovereign Visual Specification.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from deen_theme import (
    apply_deen_theme,
    style_card,
    save_deen_figure,
    BG_OBSIDIAN,
    CARD_CHARCOAL,
    GREEN_QUANTUM,
    GREEN_EMERALD,
    GREEN_FLUX,
    GOLD_HARDWARE,
    GOLD_BRIGHT,
    RED_ALERT,
    TEXT_POLAR,
    TEXT_MUTED,
)

def generate_telemetry_chart():
    apply_deen_theme()

    report_path = Path("results/experiments/glued_trees_report.json")
    assert report_path.exists(), f"Missing report: {report_path}"

    with open(report_path, "r") as f:
        data = json.load(f)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG_OBSIDIAN)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Classical vs Quantum Hitting Time / Steps Scaling (Exponential Gap)
    style_card(
        ax1,
        title="EXPONENTIAL SPEEDUP: HITTING TIME DIVERGENCE",
        subtitle="Classical Random Walk Omega(2^(n/2)) vs Quantum CTQW O(n)",
    )
    depths = [d["depth"] for d in data["depths"]]
    c_steps = [d["classical_mean_steps"] for d in data["depths"]]
    q_time = [d["quantum_time_to_peak"] * 20.0 for d in data["depths"]] # Normalized equivalent scale
    vertices = [d["total_vertices"] for d in data["depths"]]

    ax1.plot(depths, c_steps, color=RED_ALERT, marker="o", linewidth=2.2, label="Classical Walk Steps (Trapped in Weld)")
    ax1.plot(depths, q_time, color=GREEN_QUANTUM, marker="s", linewidth=2.2, label="Quantum Ballistic Time (Normalized)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Tree Depth (n)", color=TEXT_POLAR, fontsize=10)
    ax1.set_ylabel("Traversal Cost (Log Scale)", color=TEXT_POLAR, fontsize=10)
    ax1.legend(loc="upper left", facecolor=CARD_CHARCOAL, edgecolor=TEXT_MUTED, labelcolor=TEXT_POLAR)

    # Secondary axis for total vertices
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks(depths[::2])
    ax1_top.set_xticklabels([f"V={v}" for v in vertices[::2]], color=GOLD_BRIGHT, fontsize=9)

    # 2. Spatiotemporal Wavepacket Heatmap across Columns (Depth n=6, 14 Columns)
    style_card(
        ax2,
        title="SPATIOTEMPORAL BALISTIC WAVEPACKET HEATMAP",
        subtitle="Probability Density P(col j, t) Crossing Weld at Columns 6-7",
    )
    traj = data["wavepacket_spatiotemporal_trajectory"]
    times = [s["time"] for s in traj]
    probs_matrix = np.array([s["col_probabilities"] for s in traj]) # Shape: [time_steps, n_cols]

    im = ax2.imshow(
        probs_matrix.T,
        aspect="auto",
        origin="lower",
        extent=[times[0], times[-1], 0, probs_matrix.shape[1] - 1],
        cmap="Greens",
    )
    cbar = fig.colorbar(im, ax=ax2, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_POLAR)
    cbar.set_label("Probability Density", color=TEXT_POLAR, fontsize=9)

    ax2.axhline(6.5, color=GOLD_BRIGHT, linestyle="--", linewidth=1.2, label="Weld Interface (Col 6-7)")
    ax2.set_xlabel("Evolution Time (t)", color=TEXT_POLAR, fontsize=10)
    ax2.set_ylabel("Column Index (0 = ROOT_A, 13 = ROOT_B)", color=TEXT_POLAR, fontsize=10)
    ax2.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=TEXT_MUTED, labelcolor=TEXT_POLAR)

    # 3. Exit Root Probability Trajectory P_exit(t)
    style_card(
        ax3,
        title="EXIT ROOT HIT PROBABILITY TRAJECTORY",
        subtitle="Coherent Wavepacket Arrival at ROOT_B vs Classical Trap",
    )
    p_exit = [s["p_exit"] * 100.0 for s in traj]
    p_entrance = [s["p_entrance"] * 100.0 for s in traj]
    p_weld = [s["p_weld"] * 100.0 for s in traj]

    ax3.plot(times, p_entrance, color=GOLD_BRIGHT, linestyle=":", linewidth=1.5, label="ROOT_A (Entrance)")
    ax3.plot(times, p_weld, color=RED_ALERT, linestyle="--", linewidth=1.5, label="Weld Layer (Cols 6-7)")
    ax3.plot(times, p_exit, color=GREEN_EMERALD, linewidth=2.4, label="ROOT_B (Exit Hit)")

    ax3.set_xlabel("Evolution Time (t)", color=TEXT_POLAR, fontsize=10)
    ax3.set_ylabel("State Probability (%)", color=TEXT_POLAR, fontsize=10)
    ax3.set_ylim(-2, 105)
    ax3.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=TEXT_MUTED, labelcolor=TEXT_POLAR)

    # 4. Defect Injection & Quantum Steering Recovery Telemetry
    style_card(
        ax4,
        title="WELD DEFECT & QUANTUM STEERING TELEMETRY",
        subtitle="Unperturbed vs Perturbed vs Sorter Steered Recovery",
    )
    d_rep = data["defect_recovery_analysis"]
    states = ["Unperturbed", "Weld Phase Defect", "Steered Recovery"]
    probs = [d_rep["unperturbed_peak_exit_prob"] * 100.0, d_rep["perturbed_exit_prob"] * 100.0, d_rep["steered_exit_prob"] * 100.0]
    bar_colors = [GREEN_QUANTUM, RED_ALERT, GOLD_BRIGHT]

    bars = ax4.bar(states, probs, color=bar_colors, width=0.5, edgecolor=TEXT_POLAR, linewidth=0.5)
    for bar, val in zip(bars, probs):
        y_pos = bar.get_height() + 1.5
        ax4.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:.1f}%", ha="center", va="bottom", color=TEXT_POLAR, fontsize=10, fontweight="bold")

    ax4.set_ylabel("Exit Peak Probability (%)", color=TEXT_POLAR, fontsize=10)
    ax4.set_ylim(0, 75)

    # Annotate recovery
    ax4.text(
        0.5, 0.15,
        f"Defect Recovery Fidelity: {d_rep['recovery_fidelity_pct']:.1f}%\n"
        f"Destructive Interference: CONSERVED\n"
        f"Subspace Invariant: DIM(Sym) = 2n + 2 = 14",
        transform=ax4.transAxes,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_CHARCOAL, edgecolor=GOLD_HARDWARE, alpha=0.9),
        color=TEXT_POLAR,
        fontsize=9,
    )

    out_path = Path("results/charts/glued_trees_speedup_telemetry.png")
    save_deen_figure(fig, out_path)
    print(f"Chart saved to: {out_path}")

if __name__ == "__main__":
    generate_telemetry_chart()
