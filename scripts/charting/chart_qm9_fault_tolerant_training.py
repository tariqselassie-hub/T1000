# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Plots 18-Qubit Fault-Tolerant QM9 VQC Training Dynamics under Deen OS Sovereign Visual Specification.
Compares Clean Logical Circuit vs. Unprotected Noisy Circuit vs. Fault-Tolerant Mid-Circuit VNNI QEC.
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

REPORT_JSON = Path("results/data/qm9_fault_tolerant_training_report.json")
OUTPUT_PLOT = Path("results/plots/qm9_fault_tolerant_dynamics.png")

def main():
    if not REPORT_JSON.exists():
        print(f"Error: {REPORT_JSON} not found.")
        sys.exit(1)

    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    apply_deen_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG_OBSIDIAN)

    trials = data["trials"]
    colors = {
        "Ideal Clean Logical Circuit": GREEN_QUANTUM,
        "Unprotected Noisy Circuit": RED_ALERT,
        "Fault-Tolerant Mid-Circuit VNNI QEC": GOLD_HARDWARE,
    }

    # Plot 1: Epoch Loss Trajectory (MSE Loss in eV^2)
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
            alpha=0.95,
        )

    ax1.set_title("18Q Dual-Patch Logical VQC: Loss Trajectory", color=TEXT_POLAR, fontsize=13, pad=12, weight="bold")
    ax1.set_xlabel("Training Epoch", color=TEXT_POLAR, fontsize=11)
    ax1.set_ylabel("MSE Loss (eV$^2$)", color=TEXT_POLAR, fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.4, color=GRID_SLATE)
    ax1.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=9)

    # Plot 2: Final Convergence & Noise Robustness Summary
    names = [t["name"].replace("Circuit", "Ckt").replace("Logical", "Log.") for t in trials]
    mae_ev = [t["final_mae_ev"] for t in trials]
    bar_colors = [colors.get(t["name"], TEXT_POLAR) for t in trials]

    bars = ax2.bar(names, mae_ev, color=bar_colors, width=0.45, edgecolor=GRID_SLATE, linewidth=1.2)
    ax2.set_title("Final Regression Accuracy (MAE in eV)", color=TEXT_POLAR, fontsize=13, pad=12, weight="bold")
    ax2.set_ylabel("Mean Absolute Error (eV)", color=TEXT_POLAR, fontsize=11)
    ax2.set_ylim(0, max(mae_ev) * 1.35)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4, color=GRID_SLATE)

    for bar, val, t in zip(bars, mae_ev, trials):
        y_val = bar.get_height()
        corrections = t.get("total_errors_corrected", 0)
        corr_label = f"\n({corrections} QEC Corrs)" if corrections > 0 else ""
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_val + 0.25,
            f"{val:.3f} eV{corr_label}",
            ha="center",
            va="bottom",
            color=TEXT_POLAR,
            fontsize=9,
            weight="bold",
        )

    plt.suptitle(
        "T-1000: 18-Qubit Fault-Tolerant QM9 Training & Mid-Circuit VNNI QEC\n"
        "2x [[9,1,3]] Logical Patches | 262,144 Statevector Amplitudes | B3LYP DFT Molecules",
        color=TEXT_POLAR,
        fontsize=14,
        y=1.02,
        weight="bold",
    )

    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, facecolor=BG_OBSIDIAN, bbox_inches="tight")
    plt.close()
    print(f"Sovereign plot saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
