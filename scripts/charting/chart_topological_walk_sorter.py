# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Continuous Topological Walk Sorter Showdown Visualizer.
Plots:
  1. Joint & Subpopulation Accuracy across Classical, Discrete QMUX, and Continuous Walk Sorters.
  2. Ballistic Exit Hitting Probability vs Classical 1/K Dispersion Trap.
  3. Routing Contrast Ratio (Linear vs Extinction).
Strictly adheres to Deen OS Sovereign Visual Specification (Obsidian, Laser Green, Radiant Gold, Alert Red).
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

    json_path = Path("results/topological_walk_sorter_telemetry.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data["models"]
    names = [
        "Classical Walk",
        "Discrete QMUX",
        "Naked Walk",
        "Stabilized Walk"
    ]
    joint_acc = [m["joint_accuracy_pct"] for m in models]
    coarse_acc = [m["coarse_accuracy_pct"] for m in models]
    target_probs = [m["mean_target_exit_prob"] for m in models]
    contrasts = [m["mean_routing_contrast"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG_OBSIDIAN)

    # -------------------------------------------------------------------------
    # Panel 1: Classification Accuracy Showdown (Coarse vs Joint)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor(BG_DARK)
    x = np.arange(len(names))
    width = 0.35

    rects1 = ax1.bar(x - width/2, coarse_acc, width, label="Coarse Acc (%)", color=GOLD_HARDWARE, edgecolor=GRID_SLATE)
    rects2 = ax1.bar(x + width/2, joint_acc, width, label="Joint Acc (%)", color=GREEN_QUANTUM, edgecolor=GRID_SLATE)

    ax1.set_title("Hierarchical Sorting Accuracy", fontsize=13, color=TEXT_POLAR, pad=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=11, color=TEXT_SLATE)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=9, color=TEXT_SLATE)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)
    ax1.legend(loc="upper left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 2: Target Exit Ballistic Hitting Probability
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor(BG_DARK)
    bars = ax2.bar(names, target_probs, width=0.5, color=[RED_ALERT, "#f59e0b", GREEN_EMERALD, GREEN_QUANTUM], edgecolor=GRID_SLATE)
    ax2.axhline(0.25, color=GOLD_BRIGHT, linestyle="--", linewidth=1.5, label="Classical 1/K Dispersion Trap (0.25)")

    ax2.set_title("Ballistic Target Exit Probability", fontsize=13, color=TEXT_POLAR, pad=12)
    ax2.set_ylabel("Probability P(Target Exit)", fontsize=11, color=TEXT_SLATE)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=15, ha="right", fontsize=9, color=TEXT_SLATE)
    ax2.set_ylim(0, 0.65)
    ax2.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)
    ax2.legend(loc="upper left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=9)

    for bar, p in zip(bars, target_probs):
        ax2.text(bar.get_x() + bar.get_width()/2.0, p + 0.015, f"{p:.4f}", ha="center", va="bottom", color=TEXT_POLAR, fontsize=9, weight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: Routing Contrast Ratio (Log Scale)
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor(BG_DARK)
    bars_c = ax3.bar(names, contrasts, width=0.5, color=[RED_ALERT, "#f59e0b", GREEN_EMERALD, GREEN_QUANTUM], edgecolor=GRID_SLATE)

    ax3.set_title("Routing Contrast Ratio (Target / Sibling)", fontsize=13, color=TEXT_POLAR, pad=12)
    ax3.set_ylabel("Contrast Ratio (log scale)", fontsize=11, color=TEXT_SLATE)
    ax3.set_yscale("log")
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=15, ha="right", fontsize=9, color=TEXT_SLATE)
    ax3.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)

    for bar, c in zip(bars_c, contrasts):
        ax3.text(bar.get_x() + bar.get_width()/2.0, c * 1.3, f"{c:.1f}:1", ha="center", va="bottom", color=TEXT_POLAR, fontsize=9, weight="bold")

    plt.tight_layout()
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    out_fig = assets_dir / "topological_walk_sorter_showdown.png"
    save_deen_figure(fig, out_fig)
    print(f"Figure rendered to: {out_fig}")

if __name__ == "__main__":
    main()
