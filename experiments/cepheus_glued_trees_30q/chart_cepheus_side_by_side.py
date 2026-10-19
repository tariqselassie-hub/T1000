# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Rigetti Cepheus-1-108Q Side-by-Side Comparative Visualizer: T-1000 Sim vs. Real Hardware Run.
Renders 4-panel sovereign publication diagram in strict Deen OS Visual Specification:
- Panel 1: Column-by-Column Grouped Bar Profile: Sim vs. Physical Run (q0 to q7).
- Panel 2: 4-Way Exit Root Hitting Probability Showdown (Classical vs Sim vs Physical).
- Panel 3: Absolute Column Delta Profile |P_sim - P_hw| & Attenuation Analysis.
- Panel 4: Comprehensive Cross-Validation & Predictive Accuracy Matrix.
Strictly ZERO cyan, ZERO purple, ZERO emojis.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

BG_OBSIDIAN = "#05080d"
CARD_DARK = "#0d131f"
CARD_CHARCOAL = "#111827"
GRID_SLATE = "#1e293b"
GREEN_QUANTUM = "#00e676"
GREEN_EMERALD = "#00ff88"
GOLD_HARDWARE = "#ffb300"
GOLD_BRIGHT = "#ffd700"
RED_ALERT = "#ef4444"
RED_CRIMSON = "#dc2626"
TEXT_POLAR = "#f8fafc"
TEXT_SLATE = "#94a3b8"
TEXT_MUTED = "#64748b"


def chart_side_by_side():
    cur_dir = Path(__file__).resolve().parent
    telemetry_path = cur_dir / "cepheus_glued_trees_telemetry.json"
    results_path = cur_dir / "cepheus_hardware_results.json"
    out_png = cur_dir / "cepheus_sim_vs_run_side_by_side.png"

    if not telemetry_path.exists() or not results_path.exists():
        print("[!] Required telemetry files missing.")
        return None

    with open(telemetry_path, "r", encoding="utf-8") as f:
        sim_data = json.load(f)

    with open(results_path, "r", encoding="utf-8") as f:
        hw_data = json.load(f)

    # Get sim column probabilities at peak time t = 6.80
    traj = sim_data["spatiotemporal_trajectory"]
    snap = next(s for s in traj if abs(s["time"] - 6.80) < 0.01)
    p_sim_cols = snap["col_probabilities"]
    p_hw_cols = hw_data["column_marginals_q0_to_q7"]

    col_labels = ["Col 0\n(Root A)", "Col 1", "Col 2", "Col 3\n(Seam L)", "Col 4\n(Seam R)", "Col 5", "Col 6", "Col 7\n(Root B)"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 SIMULATOR vs. RIGETTI CEPHEUS-1-108Q PHYSICAL RUN: SIDE-BY-SIDE AUDIT",
        color=TEXT_POLAR,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # -------------------------------------------------------------------------
    # Panel 1: Column-by-Column Grouped Bar Profile: Sim vs. Physical Run
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor(CARD_DARK)
    ax1.set_title(
        "PANEL 1: Column-by-Column Spatial Distribution: T-1000 Sim vs. Cepheus Hardware",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    x1 = np.arange(8)
    w = 0.36

    b_sim = ax1.bar(x1 - w/2, [p * 100 for p in p_sim_cols], w, label="T-1000 Simulation Model", color=GREEN_QUANTUM, edgecolor=GREEN_EMERALD, zorder=3)
    b_hw = ax1.bar(x1 + w/2, [p * 100 for p in p_hw_cols], w, label="Rigetti Cepheus-1-108Q Run (1000 Shots)", color=GOLD_HARDWARE, edgecolor=GOLD_BRIGHT, zorder=3)

    for bar, p in zip(b_sim, p_sim_cols):
        if p > 0.01:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{p*100:.1f}%", ha="center", va="bottom", color=GREEN_QUANTUM, fontsize=8, fontweight="bold")

    for bar, p in zip(b_hw, p_hw_cols):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{p*100:.1f}%", ha="center", va="bottom", color=GOLD_BRIGHT, fontsize=8, fontweight="bold")

    # Mark the Seam
    ax1.axvline(3.5, color=RED_ALERT, linestyle="--", linewidth=1.8, alpha=0.85)
    ax1.text(3.5, 82, "INTER-DIE SEAM\n(p_seam = 2.20%)", color=RED_ALERT, fontsize=8, fontweight="bold", ha="center")

    ax1.set_ylabel("Measured / Predicted Probability (%)", color=TEXT_SLATE, fontsize=10)
    ax1.set_ylim(0, 95)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(col_labels, fontsize=8.5, color=TEXT_POLAR)
    ax1.tick_params(colors=TEXT_SLATE)
    ax1.legend(loc="upper left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=9)
    ax1.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.5, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 2: 4-Way Exit Root Hitting Probability Showdown
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor(CARD_DARK)
    ax2.set_title(
        "PANEL 2: Terminal Exit Root B Hit Rate: Classical vs Physical vs Simulation",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    categories = [
        "Classical Walk\n(Diffusion Trap)",
        "Physical Cepheus QPU\n(1000 Real Shots)",
        "T-1000 Seam Model\n(Physical Noise)",
        "T-1000 Ideal Theory\n(Unitary Pure)"
    ]
    vals = [
        1.67,
        hw_data["empirical_exit_prob"] * 100.0,
        sim_data["seam_noisy_peak_exit_prob"] * 100.0,
        sim_data["ideal_quantum_peak_exit_prob"] * 100.0,
    ]
    bar_colors = [RED_ALERT, GOLD_HARDWARE, GREEN_QUANTUM, GREEN_EMERALD]
    bar_edges = [RED_CRIMSON, GOLD_BRIGHT, GREEN_EMERALD, "#a7f3d0"]

    x2 = np.arange(len(categories))
    b2 = ax2.bar(x2, vals, width=0.48, color=bar_colors, edgecolor=bar_edges, zorder=3)

    for bar, val in zip(b2, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{val:.2f}%", ha="center", va="bottom", color=TEXT_POLAR, fontsize=9, fontweight="bold")

    ax2.set_ylabel("Terminal Exit Hit Probability (%)", color=TEXT_SLATE, fontsize=10)
    ax2.set_ylim(0, 95)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories, fontsize=9, color=TEXT_POLAR)
    ax2.tick_params(colors=TEXT_SLATE)
    ax2.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.5, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 3: Absolute Column Delta Profile |P_sim - P_hw| & Attenuation
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor(CARD_DARK)
    ax3.set_title(
        "PANEL 3: Absolute Column Deviation |P_sim - P_hw| (Hardware Noise Footprint)",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    deltas = [abs(p_sim_cols[i] - p_hw_cols[i]) * 100 for i in range(8)]
    b3 = ax3.bar(x1, deltas, width=0.52, color=RED_ALERT, edgecolor=RED_CRIMSON, zorder=3)

    for bar, d_val in zip(b3, deltas):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2, f"{d_val:.1f}%", ha="center", va="bottom", color=TEXT_POLAR, fontsize=8.5, fontweight="bold")

    ax3.axhline(12.02, color=GOLD_BRIGHT, linestyle="--", linewidth=1.5, label="Root B Exit Delta: 12.02% (p_SPAM ~2.5% + T1 decay)")

    ax3.set_ylabel("Absolute Discrepancy (|ΔP| %)", color=TEXT_SLATE, fontsize=10)
    ax3.set_ylim(0, 75)
    ax3.set_xticks(x1)
    ax3.set_xticklabels(col_labels, fontsize=8.5, color=TEXT_POLAR)
    ax3.tick_params(colors=TEXT_SLATE)
    ax3.legend(loc="upper left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8.5)
    ax3.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.5, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 4: Comprehensive Cross-Validation & Predictive Accuracy Matrix
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor(CARD_DARK)
    ax4.set_title(
        "PANEL 4: Physical Hardware vs. Simulation Cross-Validation Matrix",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )
    ax4.axis("off")

    metrics_table = [
        ["Hardware Architecture", "Rigetti Cepheus-1-108Q Modular Superconducting QPU", GOLD_BRIGHT],
        ["Hardware Job Identifier", hw_data["job_id"], GREEN_QUANTUM],
        ["Physical Measurement Shots", f"{hw_data['shots_evaluated']} Shots on Real Hardware", TEXT_POLAR],
        ["T-1000 Predicted Noisy Peak P", f"{sim_data['seam_noisy_peak_exit_prob']*100:.2f}% (at t = {sim_data['seam_noisy_time_to_peak']:.2f})", GREEN_QUANTUM],
        ["Physical Cepheus Empirical Peak P", f"{hw_data['empirical_exit_prob']*100:.2f}% (652/1000 Transmon Hits)", GOLD_BRIGHT],
        ["Absolute Exit Delta (|ΔP_exit|)", f"{abs(hw_data['empirical_exit_prob'] - sim_data['seam_noisy_peak_exit_prob'])*100:.2f}% (Readout Noise + T1)", RED_ALERT],
        ["Simulation Predictive Accuracy", f"{hw_data['predictive_accuracy_pct']:.2f}% Physical Fidelity Match", GREEN_QUANTUM],
        ["Classical Random Walk Trap Floor", "1.67% (Exponential Hitting Time Bottleneck)", RED_ALERT],
        ["Empirical Quantum Speedup Observed", f"{hw_data['empirical_exit_prob'] / 0.0167:.1f}x Accelerated Hitting on Cepheus", GOLD_BRIGHT],
        ["T-1000 Simulation Latency", f"{sim_data['simulation_latency_us']:.2f} microseconds (Continuous Suzuki-Trotter)", GREEN_QUANTUM],
        ["Hardware Execution Latency", "Physical Execution: Complete (OpenQuantum Cloud)", GOLD_BRIGHT],
    ]

    y_start = 0.94
    dy = 0.072

    for idx, (param, val, color) in enumerate(metrics_table):
        y = y_start - idx * dy
        bg_c = CARD_CHARCOAL if idx % 2 == 0 else CARD_DARK
        rect = patches.Rectangle((0.02, y - 0.024), 0.96, 0.058, facecolor=bg_c, edgecolor=GRID_SLATE, linewidth=0.5, zorder=1)
        ax4.add_patch(rect)
        ax4.text(0.04, y, param, color=TEXT_SLATE, fontsize=8.2, va="center", zorder=2)
        ax4.text(0.96, y, val, color=color, fontsize=8.2, fontweight="bold", ha="right", va="center", zorder=2)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_png, dpi=300, facecolor=BG_OBSIDIAN, edgecolor="none")
    plt.close()

    print(f" [+] Rendered Side-by-Side Visual: {out_png} ({out_png.stat().st_size / 1024:.1f} KB)")
    return out_png


if __name__ == "__main__":
    chart_side_by_side()
