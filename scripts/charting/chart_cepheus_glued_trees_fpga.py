# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""Generates the Sovereign 3-panel visualization for the Cepheus-108Q Glued-Trees
Hardware Co-Processor Experiment.

Adheres strictly to the Deen OS Sovereign Visual Specification:
Obsidian Black (#05080d), Quantum Green (#00e676), Hardware Amber (#ffb300),
Alert Crimson (#ef4444), Polar Slate (#f8fafc). Zero cyan, zero purple.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "results" / "data" / "cepheus_glued_trees_fpga_telemetry.json"
CHART_PATH = ROOT / "results" / "charts" / "cepheus_glued_trees_fpga_speedup.png"

# Sovereign Palette
BG_ROOT = "#05080d"
BG_CARD = "#0f172a"
LINE_GRID = "#1e293b"
CLR_QUANTUM = "#00e676"   # Emerald Green
CLR_CLASSICAL = "#ffb300" # Hardware Gold / Amber
CLR_NOISY = "#ef4444"     # Alert Crimson
CLR_TEXT = "#f8fafc"      # Polar Slate
CLR_MUTED = "#94a3b8"     # Muted Slate


def main():
    if not DATA_PATH.exists():
        print(f"[!] Data not found at: {DATA_PATH}")
        return

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    traj = data["trajectory"]

    times = [pt["time"] for pt in traj]
    p_classical = [pt["p_exit_classical"] for pt in traj]
    p_noisy = [pt["p_exit_noisy"] for pt in traj]
    p_fpga = [pt["p_exit_fpga_hardware"] for pt in traj]
    speedups = [pt["speedup_vs_classical"] for pt in traj]
    retentions = [pt["fidelity_retention_pct"] for pt in traj]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), facecolor=BG_ROOT)

    # -------------------------------------------------------------------------
    # Panel 1: Exit Probability Trajectory (Roots A -> B Traversal)
    # -------------------------------------------------------------------------
    ax1.set_facecolor(BG_CARD)
    ax1.plot(times, p_fpga, color=CLR_QUANTUM, linewidth=2.8, label="Physical FPGA Co-Processor (1000_QPU on COM6)")
    ax1.plot(times, p_noisy, color=CLR_NOISY, linewidth=2.0, linestyle="--", label="Unmitigated Seam Noise (2.2% Bump-Bond)")
    ax1.plot(times, p_classical, color=CLR_CLASSICAL, linewidth=1.8, linestyle=":", label="Classical Markov Random Walk")

    ax1.set_title("30-QUBIT GLUED-TREES QUANTUM WALK: ROOTS A -> B TRAVERSAL", color=CLR_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Exit Probability P(Root B)", color=CLR_TEXT, fontsize=10)
    ax1.tick_params(colors=CLR_MUTED, labelsize=9)
    ax1.grid(True, color=LINE_GRID, linestyle="-", linewidth=0.7)
    for spine in ax1.spines.values():
        spine.set_color(LINE_GRID)

    legend1 = ax1.legend(facecolor=BG_CARD, edgecolor=LINE_GRID, labelcolor=CLR_TEXT, fontsize=9, loc="upper right")
    legend1.get_frame().set_alpha(0.85)

    # Annotate peak
    peak_p = data["fpga_mitigated_peak_exit_prob"]
    peak_t = data["fpga_time_to_peak"]
    ax1.annotate(
        f"FPGA Peak: P = {peak_p:.3f} at t = {peak_t:.1f}\nSpeedup > {data['quantum_speedup_vs_classical']:.0f}x",
        xy=(peak_t, peak_p),
        xytext=(peak_t + 1.2, peak_p - 0.08),
        color=CLR_QUANTUM,
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(facecolor=CLR_QUANTUM, edgecolor=CLR_QUANTUM, arrowstyle="->", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_ROOT, edgecolor=CLR_QUANTUM, lw=1.2),
    )

    # -------------------------------------------------------------------------
    # Panel 2: Quantum Speedup Factor Over Classical Walk (Log Scale)
    # -------------------------------------------------------------------------
    ax2.set_facecolor(BG_CARD)
    ax2.semilogy(times, speedups, color=CLR_QUANTUM, linewidth=2.2, label="Quantum Speedup Ratio: P_Q(Exit) / P_C(Exit)")
    ax2.axhline(1.0, color=CLR_MUTED, linestyle="--", linewidth=1.0, label="Classical Parity Threshold (1.0x)")

    ax2.set_title("EXPONENTIAL QUANTUM SPEEDUP OVER CLASSICAL MARKOV WALK", color=CLR_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax2.set_ylabel("Speedup Multiplier (Log Scale)", color=CLR_TEXT, fontsize=10)
    ax2.tick_params(colors=CLR_MUTED, labelsize=9)
    ax2.grid(True, color=LINE_GRID, linestyle="-", linewidth=0.7, which="both")
    for spine in ax2.spines.values():
        spine.set_color(LINE_GRID)

    legend2 = ax2.legend(facecolor=BG_CARD, edgecolor=LINE_GRID, labelcolor=CLR_TEXT, fontsize=9, loc="upper left")
    legend2.get_frame().set_alpha(0.85)

    # -------------------------------------------------------------------------
    # Panel 3: Seam Mitigation Retention Percentage
    # -------------------------------------------------------------------------
    ax3.set_facecolor(BG_CARD)
    ax3.plot(times, retentions, color=CLR_CLASSICAL, linewidth=2.2, label="FPGA Seam Fidelity Retention (%)")
    ax3.axhline(90.0, color=CLR_QUANTUM, linestyle=":", linewidth=1.2, label="90% High-Fidelity Target")

    ax3.set_title("HARDWARE SEAM MITIGATION FIDELITY RETENTION ACROSS INTERPOSER", color=CLR_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax3.set_xlabel("Dimensionless Continuous Evolution Time (t)", color=CLR_TEXT, fontsize=10)
    ax3.set_ylabel("Fidelity Retention vs Ideal (%)", color=CLR_TEXT, fontsize=10)
    ax3.set_ylim(0, 105)
    ax3.tick_params(colors=CLR_MUTED, labelsize=9)
    ax3.grid(True, color=LINE_GRID, linestyle="-", linewidth=0.7)
    for spine in ax3.spines.values():
        spine.set_color(LINE_GRID)

    legend3 = ax3.legend(facecolor=BG_CARD, edgecolor=LINE_GRID, labelcolor=CLR_TEXT, fontsize=9, loc="lower right")
    legend3.get_frame().set_alpha(0.85)

    # Attribution Banner
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.06, hspace=0.32)
    fig.text(
        0.5,
        0.015,
        "T-1000 Quantum Simulation Engine | Rigetti Cepheus-1-108Q & Basys 3 FPGA (COM6) | Copyright (c) 2026 Terrence A. Jones SR",
        ha="center",
        fontsize=8,
        color=CLR_MUTED,
        fontfamily="monospace",
    )

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART_PATH, dpi=300, facecolor=BG_ROOT)
    plt.close()
    print(f"[+] Sovereign chart successfully rendered to: {CHART_PATH}")


if __name__ == "__main__":
    main()
