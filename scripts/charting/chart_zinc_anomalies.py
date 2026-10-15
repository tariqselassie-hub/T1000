# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Charting: ZINC Search 3 - Quantum Chemical Anomalies & Outliers.
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
    GOLD_HARDWARE,
    GOLD_BRIGHT,
    RED_ALERT,
    RED_CRIMSON,
    TEXT_POLAR,
    TEXT_SLATE,
    TEXT_MUTED,
    GRID_SLATE,
)

def main() -> None:
    apply_deen_theme()

    anomalies_path = Path("results/experiments/zinc_quantum_anomalies.json")
    summary_path = Path("results/experiments/zinc_dual_search_summary.json")

    with open(anomalies_path, "r", encoding="utf-8") as f:
        anomalies = json.load(f)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    ranks = [h["rank"] for h in anomalies]
    tot_scores = [h["total_anomaly_score"] for h in anomalies]
    qcd_scores = [h["qcd_anomaly"] for h in anomalies]
    hilb_scores = [h["hilbert_dispersion"] for h in anomalies]
    logps = [h["true_logp"] for h in anomalies]
    qeds = [h["true_qed"] for h in anomalies]
    sass = [h["true_sas"] for h in anomalies]

    fig = plt.figure(figsize=(18, 12), facecolor=BG_OBSIDIAN)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    # -------------------------------------------------------------
    # SUBPLOT 1: Anomaly Score Rank Spectrum (Top 100 Outliers)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    style_card(
        ax1,
        title="SEARCH 3: QUANTUM ANOMALY SCORE SPECTRUM",
        subtitle=f"Top 100 Chemical Outliers (Max Anomaly: {summary['max_anomaly_score']:.2f})",
    )

    ax1.plot(ranks, tot_scores, color=RED_ALERT, linewidth=2.0, label="Total Anomaly Score")
    ax1.plot(ranks, qcd_scores, color=GOLD_HARDWARE, linestyle="--", linewidth=1.5, label="QCD Delta Component")
    ax1.scatter([ranks[0], ranks[1]], [tot_scores[0], tot_scores[1]], color=RED_CRIMSON, s=90, edgecolors=GOLD_BRIGHT, zorder=5)

    ax1.annotate(
        f"#1 Extreme Zwitterion\nScore: {tot_scores[0]:.2f}\nlogP: {logps[0]:.2f}",
        xy=(ranks[0], tot_scores[0]),
        xytext=(ranks[0] + 8, tot_scores[0] - 1.2),
        arrowprops=dict(facecolor=GOLD_BRIGHT, edgecolor=GRID_SLATE, width=1.0, headwidth=5),
        color=TEXT_POLAR,
        fontsize=8,
        bbox=dict(boxstyle="square,pad=0.4", fc=CARD_CHARCOAL, ec=RED_ALERT, lw=1.0),
    )

    ax1.set_xlabel("Outlier Rank (1 to 100)", fontsize=9, color=TEXT_SLATE)
    ax1.set_ylabel("Quantum Anomaly Metric (QCD + 2*Hilbert)", fontsize=9, color=TEXT_SLATE)
    ax1.legend(loc="upper right")

    # -------------------------------------------------------------
    # SUBPLOT 2: logP vs SAS Phase Space (Chemical Extremes)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    style_card(
        ax2,
        title="OUTLIER PHASE SPACE (logP vs SAS)",
        subtitle="Zwitterionic / Hyper-Polar Manifold Departure",
    )

    scatter = ax2.scatter(
        logps,
        sass,
        c=tot_scores,
        cmap="hot",
        s=45,
        alpha=0.85,
        edgecolors=CARD_CHARCOAL,
        linewidths=0.5,
    )
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.02)
    cbar.set_label("Anomaly Score", color=TEXT_SLATE, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelcolor=TEXT_MUTED)

    # Reference box for normal drug-like molecules
    ax2.axvspan(-0.5, 4.5, ymin=0.0, ymax=0.5, color=GREEN_QUANTUM, alpha=0.08, label="Standard Drug-Like Window")
    ax2.set_xlabel("Lipophilicity (logP)", fontsize=9, color=TEXT_SLATE)
    ax2.set_ylabel("Synthetic Accessibility (SAS)", fontsize=9, color=TEXT_SLATE)
    ax2.legend(loc="lower right")

    # -------------------------------------------------------------
    # SUBPLOT 3: Component Decomposition (QCD vs Hilbert Dispersion)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    style_card(
        ax3,
        title="QUANTUM COMPONENT DECOMPOSITION",
        subtitle="Observable Delta vs Hilbert State Vector Dispersion Norm",
    )

    ax3.scatter(
        qcd_scores,
        hilb_scores,
        c=RED_ALERT,
        s=36,
        alpha=0.75,
        edgecolors=GOLD_HARDWARE,
        linewidths=0.6,
        label="Top 100 Outliers",
    )
    ax3.set_xlabel("Property Deviation Metric (QCD Delta)", fontsize=9, color=TEXT_SLATE)
    ax3.set_ylabel("Hilbert Statevector Norm ||<Z>||", fontsize=9, color=TEXT_SLATE)
    ax3.legend(loc="upper left")

    # -------------------------------------------------------------
    # SUBPLOT 4: Top 6 Outlier Molecular Matrix Readout
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    style_card(
        ax4,
        title="TOP CHEMICAL ANOMALY LEADERBOARD",
        subtitle=f"Direct Screen of N={summary['total_screened']:,} ZINC Molecules",
    )
    ax4.axis("off")

    top_rows = [
        ["Rank", "Score", "logP", "QED", "SAS", "Key Chemical Motif / Class"],
        ["#1", f"{tot_scores[0]:.2f}", f"{logps[0]:.2f}", f"{qeds[0]:.2f}", f"{sass[0]:.2f}", "Bis-Amidinium Nitro Thioether Zwitterion"],
        ["#2", f"{tot_scores[1]:.2f}", f"{logps[1]:.2f}", f"{qeds[1]:.2f}", f"{sass[1]:.2f}", "Dicarboxylate Diazabicyclo Zwitterion"],
        ["#3", f"{tot_scores[2]:.2f}", f"{logps[2]:.2f}", f"{qeds[2]:.2f}", f"{sass[2]:.2f}", "Carboxylate Benzobis-Amidinium Cation"],
        ["#4", f"{tot_scores[3]:.2f}", f"{logps[3]:.2f}", f"{qeds[3]:.2f}", f"{sass[3]:.2f}", "Methyl-Amidinium Benzoate Zwitterion"],
        ["#5", f"{tot_scores[4]:.2f}", f"{logps[4]:.2f}", f"{qeds[4]:.2f}", f"{sass[4]:.2f}", "Isoxazole Bis-Amidinium Peptide Isostere"],
        ["#6", f"{tot_scores[5]:.2f}", f"{logps[5]:.2f}", f"{qeds[5]:.2f}", f"{sass[5]:.2f}", "Charged Sulfonamido Guanidinium Outlier"],
    ]

    col_widths = [0.10, 0.14, 0.12, 0.12, 0.12, 0.40]
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
            cell_color = RED_ALERT if (not is_hdr and cell == row[1]) else txt_col
            ax4.text(x_pos, y_pos - 0.03, cell, transform=ax4.transAxes, fontsize=8.5, color=cell_color, fontweight=fontweight)
            x_pos += w
        y_pos -= 0.13

    # Main Super Title
    fig.suptitle(
        "T-1000 SOVEREIGN SEARCH 3: QUANTUM CHEMICAL ANOMALY & OUTLIER MINING\n"
        f"249,455 ZINC Molecules Screened in {summary['inference_wall_clock_ms']:.1f} ms | Throughput: {summary['throughput_mols_per_sec']:,.0f} mols/sec",
        fontsize=13,
        fontweight="bold",
        color=TEXT_POLAR,
        y=0.98,
    )

    out_file = Path("results/charts/zinc_quantum_anomalies.png")
    saved = save_deen_figure(fig, out_file, dpi=300)
    print(f"  -> Generated Search 3 Chart: {saved}")

if __name__ == "__main__":
    main()
