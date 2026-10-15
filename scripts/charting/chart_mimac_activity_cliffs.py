# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Charting: MIMAC Search A - Quantum Activity Cliffs & SAR Discontinuities.
Strictly adheres to Deen OS visual specification (Deep Obsidian, Emerald Green, Hardware Gold, Alert Red).
Zero emojis. Zero cyan/purple.
"""

from pathlib import Path
import json
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

    cliffs_path = Path("results/experiments/mimac_quantum_activity_cliffs.json")
    with open(cliffs_path, "r", encoding="utf-8") as f:
        top_cliffs = json.load(f)

    fig = plt.figure(figsize=(18, 12), facecolor=BG_OBSIDIAN)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    ranks = [c["rank"] for c in top_cliffs]
    saliences = [c["cliff_salience"] for c in top_cliffs]
    act_deltas = [c["act_delta"] for c in top_cliffs]
    q_fids = [c["quantum_fidelity"] for c in top_cliffs]
    tanimotos = [c["tanimoto_similarity"] for c in top_cliffs]
    hybrid_sims = [c["hybrid_similarity"] for c in top_cliffs]
    folds = [c["potency_gain_fold"] for c in top_cliffs]
    d_logps = [c["delta_logp"] for c in top_cliffs]
    d_tpsas = [c["delta_tpsa"] for c in top_cliffs]

    # -------------------------------------------------------------
    # SUBPLOT 1: Quantum State Fidelity vs Bioactivity Delta
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    style_card(
        ax1,
        title="QUANTUM FIDELITY VS BIOACTIVITY DISCONTINUITY",
        subtitle="Upper-Right Quadrant: High Structural Identity with Massive Potency Divergence",
    )

    from matplotlib.colors import LinearSegmentedColormap
    deen_cmap = LinearSegmentedColormap.from_list("deen_gold_red", [GOLD_HARDWARE, RED_ALERT])

    scatter1 = ax1.scatter(
        q_fids,
        act_deltas,
        c=saliences,
        cmap=deen_cmap,
        s=80,
        alpha=0.9,
        edgecolors=GOLD_BRIGHT,
        linewidths=0.8,
        zorder=4,
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1, pad=0.02)
    cbar1.set_label("Cliff Salience Metric", color=TEXT_SLATE, fontsize=8)
    cbar1.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelcolor=TEXT_MUTED)

    # Highlight #1 and #2
    ax1.scatter([q_fids[0]], [act_deltas[0]], color=RED_ALERT, s=160, edgecolors=GOLD_BRIGHT, lw=1.5, zorder=6)
    ax1.annotate(
        f"#1 Magic Switch\nJump: +{act_deltas[0]:.2f} ({folds[0]:.1f}x)\nFidelity: {q_fids[0]:.3f}",
        xy=(q_fids[0], act_deltas[0]),
        xytext=(q_fids[0] - 0.12, act_deltas[0] - 0.8),
        arrowprops=dict(facecolor=GOLD_BRIGHT, edgecolor=GRID_SLATE, width=1.0, headwidth=5),
        color=TEXT_POLAR,
        fontsize=8,
        bbox=dict(boxstyle="square,pad=0.4", fc=CARD_CHARCOAL, ec=RED_ALERT, lw=1.0),
    )

    ax1.set_xlabel("10-Qubit Quantum State Fidelity |<psi_A|psi_B>|^2", fontsize=9, color=TEXT_SLATE)
    ax1.set_ylabel("Bioactivity Delta (|Act_B - Act_A|)", fontsize=9, color=TEXT_SLATE)
    ax1.set_xlim(0.60, 1.00)
    ax1.set_ylim(4.5, 8.5)

    # -------------------------------------------------------------
    # SUBPLOT 2: Top 10 Cliff Potency Gain Folds
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    style_card(
        ax2,
        title="POTENCY MULTIPLIER (FOLD INCREASE)",
        subtitle="Activity Leaps Unlocked by Single Functional Group Modifications",
    )

    top10_ranks = np.arange(1, 11)
    bars = ax2.bar(top10_ranks, folds[:10], color=GOLD_HARDWARE, edgecolor=GRID_SLATE, width=0.55, alpha=0.9)
    bars[1].set_color(GREEN_EMERALD) # #2 has 9.3x fold gain
    bars[0].set_color(RED_ALERT)    # #1 has highest salience

    for b, f in zip(bars, folds[:10]):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{f:.1f}x", ha="center", va="bottom", fontsize=8, color=TEXT_POLAR, fontweight="bold")

    ax2.set_xlabel("Cliff Salience Rank (Top 10)", fontsize=9, color=TEXT_SLATE)
    ax2.set_ylabel("Bioactivity Fold Increase", fontsize=9, color=TEXT_SLATE)
    ax2.set_ylim(0, 11)

    # -------------------------------------------------------------
    # SUBPLOT 3: Physicochemical Shift Vectors
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    style_card(
        ax3,
        title="PHYSICOCHEMICAL SHIFT VECTORS (dLogP vs dTPSA)",
        subtitle="Displacement Vector Unlocking Receptor Potency Jump",
    )

    scatter3 = ax3.scatter(
        d_logps,
        d_tpsas,
        c=act_deltas,
        cmap="autumn",
        s=70,
        alpha=0.85,
        edgecolors=GRID_SLATE,
        linewidths=0.7,
    )
    cbar3 = fig.colorbar(scatter3, ax=ax3, pad=0.02)
    cbar3.set_label("Activity Jump (+Units)", color=TEXT_SLATE, fontsize=8)
    cbar3.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelcolor=TEXT_MUTED)

    ax3.axhline(0, color=GRID_SLATE, linestyle="--", linewidth=0.8)
    ax3.axvline(0, color=GRID_SLATE, linestyle="--", linewidth=0.8)
    ax3.set_xlabel("Shift in Lipophilicity (dLogP)", fontsize=9, color=TEXT_SLATE)
    ax3.set_ylabel("Shift in Polar Surface Area (dTPSA, A^2)", fontsize=9, color=TEXT_SLATE)

    # -------------------------------------------------------------
    # SUBPLOT 4: Top 5 Magic Switches Leaderboard Readout
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    style_card(
        ax4,
        title="TOP BIOACTIVITY MAGIC SWITCH LEADERBOARD",
        subtitle="Direct Discovery from 584 Cliff Pairs & 1024-Dim Hilbert Space",
    )
    ax4.axis("off")

    top_rows = [
        ["Rank", "Salience", "Act Jump", "Gain", "Quantum Fid", "Transformation Description"],
        ["#1", f"{saliences[0]:.1f}", f"{top_cliffs[0]['act_parent']:.1f} -> {top_cliffs[0]['act_lead']:.1f}", f"{folds[0]:.1f}x", f"{q_fids[0]:.3f}", "Hydroxymethyl -> Ethyl on Purine Scaffold"],
        ["#2", f"{saliences[1]:.1f}", f"{top_cliffs[1]['act_parent']:.1f} -> {top_cliffs[1]['act_lead']:.1f}", f"{folds[1]:.1f}x", f"{q_fids[1]:.3f}", "Meta-Amino -> Hydroxymethyl on Benzylamine"],
        ["#3", f"{saliences[2]:.1f}", f"{top_cliffs[2]['act_parent']:.1f} -> {top_cliffs[2]['act_lead']:.1f}", f"{folds[2]:.1f}x", f"{q_fids[2]:.3f}", "Ethyl -> Chloro on Substituted Benzylamine"],
        ["#4", f"{saliences[3]:.1f}", f"{top_cliffs[3]['act_parent']:.1f} -> {top_cliffs[3]['act_lead']:.1f}", f"{folds[3]:.1f}x", f"{q_fids[3]:.3f}", "Ethyl -> Hydroxymethyl Core Scaffold Shift"],
        ["#5", f"{saliences[4]:.1f}", f"{top_cliffs[4]['act_parent']:.1f} -> {top_cliffs[4]['act_lead']:.1f}", f"{folds[4]:.1f}x", f"{q_fids[4]:.3f}", "Fluoro -> Hydroxymethyl Exchange on Arene"],
    ]

    col_widths = [0.10, 0.14, 0.16, 0.12, 0.16, 0.32]
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
        "MIMAC SEARCH A: QUANTUM ACTIVITY CLIFFS & SAR DISCONTINUITIES\n"
        "325 Molecules | 584 Cliff Pairs | 10-Qubit Hilbert Space State Fidelity Mapping",
        fontsize=13,
        fontweight="bold",
        color=TEXT_POLAR,
        y=0.98,
    )

    out_file = Path("results/charts/mimac_quantum_activity_cliffs.png")
    saved = save_deen_figure(fig, out_file, dpi=300)
    print(f"  -> Generated Search A Chart: {saved}")

if __name__ == "__main__":
    main()
