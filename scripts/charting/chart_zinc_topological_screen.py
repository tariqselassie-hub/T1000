# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
ZINC Molecular Screening: Discrete Gate VQC vs Continuous Topological Walk.
Visualizes:
  1. Quadrant Accuracy & Drug Lead Recall (95.4% vs 55.0%).
  2. Routing Contrast Ratio (22.2:1 vs 2.7:1).
  3. Chemical Latent Space (QED vs SAS) showing topological quadrant classification.
Strictly conforms to Deen OS Sovereign Visual Specification.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deen_theme import (
    apply_deen_theme, save_deen_figure,
    BG_OBSIDIAN, BG_DARK, CARD_CHARCOAL, GRID_SLATE,
    GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_SLATE, TEXT_MUTED
)

def main():
    apply_deen_theme()

    telemetry_path = Path("results/zinc_topological_walk_telemetry.json")
    if not telemetry_path.exists():
        print(f"Error: {telemetry_path} not found.")
        return

    with open(telemetry_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data["models"]
    names = ["Discrete VQC", "Continuous Walk", "Analog Pulse Walk"]
    accuracies = [m["overall_quadrant_accuracy_pct"] for m in models]
    coarse_accs = [m["coarse_qed_accuracy_pct"] for m in models]
    fine_accs = [m["fine_sas_accuracy_pct"] for m in models]
    recalls = [m["lead_recall_pct"] for m in models]
    contrasts = [m["mean_routing_contrast"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor(BG_OBSIDIAN)

    # -------------------------------------------------------------------------
    # Panel 1: Accuracy & Lead Recall
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor(CARD_CHARCOAL)
    x = np.arange(len(names))
    width = 0.20

    r1 = ax1.bar(x - 1.5 * width, coarse_accs, width, label="Coarse (QED)", color=GOLD_HARDWARE, alpha=0.9)
    r2 = ax1.bar(x - 0.5 * width, fine_accs, width, label="Fine (SAS)", color=TEXT_SLATE, alpha=0.9)
    r3 = ax1.bar(x + 0.5 * width, accuracies, width, label="Overall Quadrant", color=GREEN_QUANTUM, alpha=0.9)
    r4 = ax1.bar(x + 1.5 * width, recalls, width, label="Lead Recall", color=GREEN_EMERALD, alpha=0.9)

    ax1.set_ylabel("Accuracy / Recall Rate (%)", color=TEXT_POLAR, fontsize=10)
    ax1.set_title("ZINC Screening Accuracy & Lead Discovery", color=TEXT_POLAR, fontsize=11, fontweight="bold", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, color=TEXT_POLAR, fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.3, color=GRID_SLATE)
    ax1.legend(loc="upper left", framealpha=0.4, facecolor=BG_DARK, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8)

    for bar in r3:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%",
                 ha="center", va="bottom", color=TEXT_POLAR, fontsize=8, fontweight="bold")

    for bar in r4:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.1f}%",
                 ha="center", va="bottom", color=GREEN_EMERALD, fontsize=8, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 2: Routing Contrast Ratio
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor(CARD_CHARCOAL)
    colors = [RED_ALERT, GREEN_QUANTUM, GREEN_EMERALD]
    bars2 = ax2.bar(range(len(names)), contrasts, color=colors, width=0.45, edgecolor=GRID_SLATE)

    ax2.set_ylabel("Mean Extinction Contrast Ratio (P_target : P_compete)", color=TEXT_POLAR, fontsize=10)
    ax2.set_title("Topological Routing Signal-to-Noise Ratio", color=TEXT_POLAR, fontsize=11, fontweight="bold", pad=12)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, color=TEXT_POLAR, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3, color=GRID_SLATE)

    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.8, f"{yval:.1f}:1",
                 ha="center", va="bottom", color=TEXT_POLAR, fontsize=9, fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: Chemical Space (QED vs SAS) with Top Discovered Leads
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor(CARD_CHARCOAL)

    # Load 1000 molecules for background distribution
    csv_path = Path("tests/benchmarks/zinc_molecular/outputs/zinc_test_1k.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        ax3.scatter(df["qed"], df["SAS"], color=TEXT_MUTED, alpha=0.18, s=12, label="ZINC 1k Candidates")

    # Threshold guidelines
    ax3.axvline(0.60, color=GOLD_HARDWARE, linestyle="--", alpha=0.6, label="QED Threshold (0.60)")
    ax3.axhline(3.50, color=GOLD_HARDWARE, linestyle=":", alpha=0.6, label="SAS Threshold (3.50)")

    # Highlight top 5 discovered leads
    top_leads = data.get("top_lead_candidates", [])
    if top_leads:
        q_leads = [lead["qed"] for lead in top_leads]
        s_leads = [lead["sas"] for lead in top_leads]
        ax3.scatter(q_leads, s_leads, color=GREEN_QUANTUM, edgecolor=GOLD_BRIGHT, s=85, linewidth=1.5, zorder=5, label="Discovered Top Leads")

    ax3.text(0.78, 2.0, "QUADRANT 0\n(Prime Leads)", color=GREEN_QUANTUM, fontsize=8, fontweight="bold", ha="center")
    ax3.text(0.78, 4.8, "QUADRANT 1\n(Hard Synthesis)", color=TEXT_SLATE, fontsize=8, ha="center")
    ax3.text(0.40, 2.0, "QUADRANT 2\n(Building Blocks)", color=TEXT_SLATE, fontsize=8, ha="center")
    ax3.text(0.40, 4.8, "QUADRANT 3\n(Deprioritized)", color=RED_ALERT, fontsize=8, ha="center")

    ax3.set_xlabel("Quantitative Drug-Likeness (QED)", color=TEXT_POLAR, fontsize=10)
    ax3.set_ylabel("Synthetic Accessibility (SAS)", color=TEXT_POLAR, fontsize=10)
    ax3.set_title("Latent Space Topological Partitioning", color=TEXT_POLAR, fontsize=11, fontweight="bold", pad=12)
    ax3.grid(True, linestyle="--", alpha=0.3, color=GRID_SLATE)
    ax3.legend(loc="upper left", framealpha=0.4, facecolor=BG_DARK, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=7)

    plt.tight_layout()
    out_asset = Path("assets/zinc_topological_walk_screen.png")
    save_deen_figure(fig, out_asset)
    print(f"Figure saved to {out_asset}")

if __name__ == "__main__":
    main()
