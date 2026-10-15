# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Charting: ZINC Search 1 - Frontier Pareto Drug Candidate Mining.
Strictly adheres to Deen OS visual specification (Deep Obsidian, Emerald Green, Hardware Gold, Alert Red).
Zero emojis. Zero cyan/purple.
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
    TEXT_SLATE,
    TEXT_MUTED,
    GRID_SLATE,
)

def main() -> None:
    apply_deen_theme()

    pareto_path = Path("results/experiments/zinc_pareto_leads.json")
    summary_path = Path("results/experiments/zinc_dual_search_summary.json")

    with open(pareto_path, "r", encoding="utf-8") as f:
        leads = json.load(f)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    ranks = [h["rank"] for h in leads]
    pareto_scores = [h["pareto_score"] for h in leads]
    logps = [h["true_logp"] for h in leads]
    qeds = [h["true_qed"] for h in leads]
    sass = [h["true_sas"] for h in leads]
    pred_logps = [h["pred_logp"] for h in leads]
    pred_qeds = [h["pred_qed"] for h in leads]
    pred_sass = [h["pred_sas"] for h in leads]

    fig = plt.figure(figsize=(18, 12), facecolor=BG_OBSIDIAN)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    # -------------------------------------------------------------
    # SUBPLOT 1: The Frontier Pareto Curve (QED vs SAS)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    style_card(
        ax1,
        title="SEARCH 1: THE PARETO DRUG CANDIDATE FRONTIER",
        subtitle=f"QED (Drug-Likeness) vs SAS (Synthetic Accessibility) | N=100 Leads",
    )

    from matplotlib.colors import LinearSegmentedColormap
    deen_cmap = LinearSegmentedColormap.from_list("deen_gold_green", [GOLD_HARDWARE, GREEN_EMERALD])
    deen_green_gold = LinearSegmentedColormap.from_list("deen_green_gold", [GREEN_FLUX, GOLD_BRIGHT])

    scatter1 = ax1.scatter(
        sass,
        qeds,
        c=logps,
        cmap=deen_green_gold,
        s=60,
        alpha=0.9,
        edgecolors=GOLD_HARDWARE,
        linewidths=0.7,
        zorder=4,
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1, pad=0.02)
    cbar1.set_label("Lipophilicity (logP)", color=TEXT_SLATE, fontsize=8)
    cbar1.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelcolor=TEXT_MUTED)

    # Annotate Top Lead #1
    ax1.scatter([sass[0]], [qeds[0]], color=GREEN_EMERALD, s=150, edgecolors=GOLD_BRIGHT, lw=1.5, zorder=6)
    ax1.annotate(
        f"#1 Golden Lead\nPareto: {pareto_scores[0]:.4f}\nQED: {qeds[0]:.3f} | SAS: {sass[0]:.3f}",
        xy=(sass[0], qeds[0]),
        xytext=(sass[0] + 0.08, qeds[0] - 0.035),
        arrowprops=dict(facecolor=GOLD_BRIGHT, edgecolor=GRID_SLATE, width=1.0, headwidth=5),
        color=TEXT_POLAR,
        fontsize=8,
        bbox=dict(boxstyle="square,pad=0.4", fc=CARD_CHARCOAL, ec=GREEN_QUANTUM, lw=1.0),
    )

    ax1.set_xlabel("Synthetic Accessibility Score (SAS) -> Lower is Easier", fontsize=9, color=TEXT_SLATE)
    ax1.set_ylabel("Quantitative Drug-Likeness (QED) -> Higher is Better", fontsize=9, color=TEXT_SLATE)
    ax1.set_xlim(1.2, 1.8)
    ax1.set_ylim(0.85, 0.94)

    # -------------------------------------------------------------
    # SUBPLOT 2: Ideal Oral Bioavailability Window (logP vs QED)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    style_card(
        ax2,
        title="IDEAL BIOAVAILABILITY COMPLIANCE WINDOW",
        subtitle="Lipinski Rule-of-5 & Veber Oral Parameter Target",
    )

    # Highlight optimal drug-like corridor
    ax2.axvspan(2.0, 3.2, ymin=0.0, ymax=1.0, color=GREEN_QUANTUM, alpha=0.10, label="Optimal Lipophilic Window (2.0 - 3.2)")
    scatter2 = ax2.scatter(
        logps,
        qeds,
        c=pareto_scores,
        cmap=deen_cmap,
        s=50,
        alpha=0.85,
        edgecolors=GRID_SLATE,
        linewidths=0.6,
        label="Top 100 Pareto Leads",
    )
    cbar2 = fig.colorbar(scatter2, ax=ax2, pad=0.02)
    cbar2.set_label("Pareto Score", color=TEXT_SLATE, fontsize=8)
    cbar2.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelcolor=TEXT_MUTED)

    ax2.set_xlabel("Calculated logP (Ideal: 2.0 to 3.0)", fontsize=9, color=TEXT_SLATE)
    ax2.set_ylabel("QED Drug-Likeness Score", fontsize=9, color=TEXT_SLATE)
    ax2.legend(loc="lower left")

    # -------------------------------------------------------------
    # SUBPLOT 3: Model Alignment (QNN Predicted vs Ground-Truth)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    style_card(
        ax3,
        title="12-QUBIT QUANTUM NEURAL NETWORK ALIGNMENT",
        subtitle="True vs Predicted Values across Top 100 Pareto Leads",
    )

    lead_indices = np.arange(1, 26)
    width = 0.35

    ax3.bar(lead_indices - width / 2, qeds[:25], width=width, color=GREEN_QUANTUM, label="Ground-Truth QED", alpha=0.85)
    ax3.bar(lead_indices + width / 2, pred_qeds[:25], width=width, color=GOLD_HARDWARE, label="QNN Predicted QED", alpha=0.85)

    ax3.set_xlabel("Top 25 Lead Rank", fontsize=9, color=TEXT_SLATE)
    ax3.set_ylabel("QED Drug-Likeness Metric", fontsize=9, color=TEXT_SLATE)
    ax3.set_ylim(0.7, 1.0)
    ax3.legend(loc="lower right")

    # -------------------------------------------------------------
    # SUBPLOT 4: Top 6 Golden Leads Leaderboard Readout
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    style_card(
        ax4,
        title="TOP 6 GOLDEN PARETO DRUG CANDIDATES",
        subtitle=f"High Throughput Mined from N={summary['total_screened']:,} Candidates",
    )
    ax4.axis("off")

    top_rows = [
        ["Rank", "Pareto", "QED", "SAS", "logP", "Molecular Architecture / Core Motif"],
        ["#1", f"{pareto_scores[0]:.4f}", f"{qeds[0]:.3f}", f"{sass[0]:.3f}", f"{logps[0]:.3f}", "Dimethoxy Diaryl Benzamide Isostere"],
        ["#2", f"{pareto_scores[1]:.4f}", f"{qeds[1]:.3f}", f"{sass[1]:.3f}", f"{logps[1]:.3f}", "Dimethoxy Diaryl Acetamide Bridge"],
        ["#3", f"{pareto_scores[2]:.4f}", f"{qeds[2]:.3f}", f"{sass[2]:.3f}", f"{logps[2]:.3f}", "Methoxy Fluorophenyl Benzamide"],
        ["#4", f"{pareto_scores[3]:.4f}", f"{qeds[3]:.3f}", f"{sass[3]:.3f}", f"{logps[3]:.3f}", "Methylcarboxamido Fluorobenzamide"],
        ["#5", f"{pareto_scores[4]:.4f}", f"{qeds[4]:.3f}", f"{sass[4]:.3f}", f"{logps[4]:.3f}", "Bis(methoxyphenoxy) Acetamide Core"],
        ["#6", f"{pareto_scores[5]:.4f}", f"{qeds[5]:.3f}", f"{sass[5]:.3f}", f"{logps[5]:.3f}", "Phenoxyacetyl Aminobenzonitrile"],
    ]

    col_widths = [0.10, 0.15, 0.12, 0.12, 0.12, 0.39]
    y_pos = 0.88
    for row_idx, row in enumerate(top_rows):
        x_pos = 0.02
        is_hdr = row_idx == 0
        bg_col = CARD_CHARCOAL if is_hdr else (BG_OBSIDIAN if row_idx % 2 == 0 else "#0c1322")
        txt_col = GOLD_BRIGHT if is_hdr else TEXT_POLAR

        rect = plt.Rectangle((0.0, y_pos - 0.08), 1.0, 0.12, facecolor=bg_col, edgecolor=GRID_SLATE, lw=0.8, transform=ax4.transAxes)
        ax4.add_patch(rect)

        for cell, w in zip(row, col_widths):
            fontweight = "bold" if (is_hdr or cell.startswith("#")) else "normal"
            cell_color = GREEN_EMERALD if (not is_hdr and cell == row[1]) else txt_col
            ax4.text(x_pos, y_pos - 0.03, cell, transform=ax4.transAxes, fontsize=8.5, color=cell_color, fontweight=fontweight)
            x_pos += w
        y_pos -= 0.13

    # Main Super Title
    fig.suptitle(
        "T-1000 SOVEREIGN SEARCH 1: FRONTIER PARETO DRUG CANDIDATE MINING\n"
        f"249,455 ZINC Molecules Evaluated in {summary['inference_wall_clock_ms']:.1f} ms | Peak Pareto: {summary['max_pareto_score']:.4f}",
        fontsize=13,
        fontweight="bold",
        color=TEXT_POLAR,
        y=0.98,
    )

    out_file = Path("results/charts/zinc_pareto_leads.png")
    saved = save_deen_figure(fig, out_file, dpi=300)
    print(f"  -> Generated Search 1 Chart: {saved}")

if __name__ == "__main__":
    main()
