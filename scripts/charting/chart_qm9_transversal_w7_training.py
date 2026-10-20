# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Plots 18-Qubit Transversal [[7, 1, 3]] Steane-Wheel FT-QNN Training Dynamics (Monograph M153).
Compares Clean Logical Circuit vs. Unprotected Noisy Circuit vs. Fault-Tolerant In-Register Ancilla QEC.
Renders under Deen OS Sovereign Visual Specification.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add current workspace scripts/charting to path
sys.path.insert(0, str(Path(__file__).parent))
from deen_theme import (
    apply_deen_theme,
    BG_OBSIDIAN,
    CARD_CHARCOAL,
    GREEN_QUANTUM,
    GOLD_HARDWARE,
    RED_ALERT,
    TEXT_POLAR,
    TEXT_MUTED,
    GRID_SLATE,
)

REPORT_JSON = Path("results/data/qm9_transversal_w7_report.json")
OUTPUT_PLOT = Path("results/plots/qm9_transversal_w7_dynamics.png")


def main():
    if not REPORT_JSON.exists():
        print(f"Error: {REPORT_JSON} not found.")
        sys.exit(1)

    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    apply_deen_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor=BG_OBSIDIAN)

    trials = data["trials"]
    colors = {
        "Ideal Clean Transversal Circuit": GREEN_QUANTUM,
        "Unprotected Noisy Circuit": RED_ALERT,
        "Fault-Tolerant In-Register Ancilla QEC": GOLD_HARDWARE,
    }

    # -------------------------------------------------------------------------
    # Subplot 1: Epoch Loss Trajectory (MSE in eV^2)
    # -------------------------------------------------------------------------
    epochs = list(range(1, data["epochs"] + 1))
    for t in trials:
        name = t["name"]
        color = colors.get(name, TEXT_POLAR)
        linestyle = "--" if "Unprotected" in name else "-"
        linewidth = 2.4 if "Fault-Tolerant" in name else 1.8
        ax1.plot(
            epochs,
            t["epoch_losses"],
            label=f"{name} (Final: {t['final_loss_mse']:.2f})",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker="o" if "Fault-Tolerant" in name else None,
            markersize=4,
        )

    ax1.set_title(
        "[1] QM9 Transversal VQC Loss Convergence (20 Epochs)",
        color=TEXT_POLAR,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax1.set_xlabel("Epoch", color=TEXT_POLAR, fontsize=11)
    ax1.set_ylabel("Mean Squared Error (eV²)", color=TEXT_POLAR, fontsize=11)
    ax1.tick_params(colors=TEXT_MUTED)
    ax1.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)
    ax1.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=9)

    # -------------------------------------------------------------------------
    # Subplot 2: MAE Comparison & Noise Mitigation Bar Chart
    # -------------------------------------------------------------------------
    labels = ["Ideal Clean", "Unprotected Noisy", "Fault-Tolerant Ancilla"]
    maes = [t["final_mae_ev"] for t in trials]
    bar_colors = [GREEN_QUANTUM, RED_ALERT, GOLD_HARDWARE]

    bars = ax2.bar(labels, maes, color=bar_colors, width=0.45, edgecolor=BG_OBSIDIAN, alpha=0.9)
    ax2.set_title(
        "[2] Final MAE (eV) Across 50 QM9 Molecules (Lowest: Fault-Tolerant Ancilla QEC)",
        color=TEXT_POLAR,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax2.set_ylabel("Mean Absolute Error (eV)", color=TEXT_POLAR, fontsize=11)
    ax2.set_ylim(0, max(maes) * 1.18)
    ax2.tick_params(colors=TEXT_MUTED)
    ax2.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6, axis="y")

    # Add text labels on bars
    for bar, t in zip(bars, trials):
        h = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.15,
            f"{h:.3f} eV\n({t['final_mae_kcal_mol']:.1f} kcal/mol)",
            ha="center",
            va="bottom",
            color=TEXT_POLAR,
            fontsize=9,
            fontweight="bold",
        )

    # Top Supertitle
    fig.suptitle(
        f"T-1000 Transversal [[7, 1, 3]] Steane-Wheel FT-QNN Telemetry (Monograph M153)\n"
        f"18 Qubits (14 Data + 4 Active Physical Ancillas) | AVX-VNNI Syndrome Decoder | QM9 (N=50)",
        color=TEXT_POLAR,
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=200, facecolor=BG_OBSIDIAN, edgecolor="none")
    plt.close()
    print(f"[+] Sovereign chart successfully saved to: {OUTPUT_PLOT.resolve()}")


if __name__ == "__main__":
    main()
