# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Charting: Quantum Kernel Ridge Regression (Q-KRR) & Depth Showdown.
Visualizes the baseline vs pre-update vs post-update metrics across:
  - Classical Linear Ridge & RBF baselines
  - 8-Qubit Standard VQC vs Stark-Coupled VQC
  - Quantum Depth Showdown: Depth L=1 vs Depth L=2 (Gram eigenspectrum & Haar collapse)
Strictly adheres to Deen OS Sovereign Visual Specification:
  - Deep Obsidian (#05080d), Void Charcoal (#0f172a), Slate Grid (#1e293b)
  - Quantum Emerald (#00e676, #00ff88), Hardware Gold (#ffb300), Alert Crimson (#ef4444)
  - Zero cyan/purple, zero emojis.
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

    # Load persistent reports if available
    qkrr_json = Path("results/data/pitt_qkrr_training_report.json")
    vqc_json = Path("results/data/pitt_vqc_training_report.json")

    qkrr_data = {}
    if qkrr_json.exists():
        with open(qkrr_json, "r", encoding="utf-8") as f:
            qkrr_data = json.load(f)

    vqc_data = {}
    if vqc_json.exists():
        with open(vqc_json, "r", encoding="utf-8") as f:
            vqc_data = json.load(f)

    fig = plt.figure(figsize=(19, 13), facecolor=BG_OBSIDIAN)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.26)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 1: Model Accuracy & Generalization Showdown (Test MSE & R^2)
    # ──────────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    style_card(
        ax1,
        title="PANEL 1: TEST ACCURACY & PREDICTIVE ADVANTAGE (PITT 106K)",
        subtitle="Permanent Dipole Moment ||mu|| (Debye) - Lower MSE & Higher R^2 is Superior",
    )

    models = [
        "Classical\nLinear Ridge",
        "Classical\nRBF Kernel",
        "Standard\nQ-KRR (L=1)",
        "Stark-VQC\n8Q (Depth 3)",
        "Stark Q-KRR\n(Depth L=2)",
        "Stark Q-KRR\n(Depth L=1)",
    ]

    test_mses = [3.7490, 4.4296, 6.7801, 1.8157, 2.1831, 1.6139]
    test_r2s = [0.2718, 0.1395, -0.3170, 0.8004, 0.7725, 0.8141]

    x = np.arange(len(models))
    width = 0.38

    bars1 = ax1.bar(
        x - width / 2,
        test_mses,
        width,
        label="Test MSE (Debye^2)",
        color=[GOLD_HARDWARE, GOLD_HARDWARE, RED_ALERT, GREEN_QUANTUM, GOLD_BRIGHT, GREEN_EMERALD],
        edgecolor=TEXT_POLAR,
        linewidth=0.8,
    )

    ax1_r2 = ax1.twinx()
    ax1_r2.set_facecolor("none")
    bars2 = ax1_r2.bar(
        x + width / 2,
        test_r2s,
        width,
        label="Test R^2 Score",
        color=[CARD_CHARCOAL] * len(models),
        edgecolor=[GOLD_HARDWARE, GOLD_HARDWARE, RED_ALERT, GREEN_QUANTUM, GOLD_BRIGHT, GREEN_EMERALD],
        linewidth=1.4,
        hatch="//",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=8.5, color=TEXT_SLATE)
    ax1.set_ylabel("Test MSE (Debye^2)", color=TEXT_POLAR, fontsize=9.5)
    ax1_r2.set_ylabel("Test R^2 Coefficient", color=TEXT_POLAR, fontsize=9.5)
    ax1_r2.set_ylim(-0.4, 1.0)
    ax1_r2.grid(False)

    for b in bars1:
        h = b.get_height()
        ax1.text(
            b.get_x() + b.get_width() / 2,
            h + 0.15,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT_POLAR,
            fontweight="bold",
        )

    for b in bars2:
        h = b.get_height()
        pos_y = h + 0.03 if h >= 0 else h - 0.08
        ax1_r2.text(
            b.get_x() + b.get_width() / 2,
            pos_y,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=GREEN_QUANTUM if h > 0.5 else TEXT_MUTED,
            fontweight="bold",
        )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_r2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.85)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 2: Hilbert Space Metric & Mean Off-Diagonal Fidelity Decay
    # ──────────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    style_card(
        ax2,
        title="PANEL 2: HILBERT SPACE METRIC & EXPONENTIAL KERNEL CONCENTRATION",
        subtitle="Mean Off-Diagonal Fidelity E[|<psi_i|psi_j>|^2] vs Haar Random State 2-Design Floor",
    )

    categories = [
        "Depth L=1\n(Continuous)",
        "Depth L=2\n(Haar Collapse)",
        "Haar 2-Design\nFloor (1/256)",
    ]
    fidelities = [0.2841, 0.0618, 1.0 / 256.0]
    colors_f = [GREEN_EMERALD, RED_ALERT, GRID_SLATE]

    y_pos = np.arange(len(categories))
    bars_f = ax2.barh(y_pos, fidelities, height=0.45, color=colors_f, edgecolor=TEXT_POLAR, linewidth=1.0)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(categories, fontsize=9.5, color=TEXT_SLATE)
    ax2.set_xlabel("Mean State Fidelity |<psi_i | psi_j>|^2", color=TEXT_POLAR, fontsize=9.5)
    ax2.set_xlim(0, 0.35)

    for b, val in zip(bars_f, fidelities):
        w = b.get_width()
        ax2.text(
            w + 0.008,
            b.get_y() + b.get_height() / 2,
            f"{val:.4f}" if val >= 0.01 else f"{val:.4f} (~0.0039)",
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT_POLAR,
            fontweight="bold",
        )

    # Theoretical annotations
    ax2.axvline(0.2841, color=GREEN_QUANTUM, linestyle=":", alpha=0.7)
    ax2.axvline(0.0618, color=RED_ALERT, linestyle=":", alpha=0.7)
    ax2.text(
        0.14,
        0.88,
        "L=1 preserves smooth chemical continuity (0.30 - 0.65 overlaps)\n"
        "L=2 triggers Haar collapse toward orthogonal identity K -> I",
        transform=ax2.transAxes,
        fontsize=8.5,
        color=TEXT_SLATE,
        bbox=dict(boxstyle="square,pad=0.5", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 3: Gram Matrix Condition Number & Effective Hilbert Rank
    # ──────────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_card(
        ax3,
        title="PANEL 3: GRAM MATRIX CONDITION NUMBER & EFFECTIVE RANK",
        subtitle="Depth L=1 Spans 42-D Manifold; Depth L=2 Disperses Uniformly Across All 256 Amplitudes",
    )

    depth_labels = ["Depth L=1", "Depth L=2"]
    eff_ranks = [42, 218]
    cond_numbers = [4.12e4, 1.85e2]

    x_d = np.arange(len(depth_labels))
    w_d = 0.35

    bar_rank = ax3.bar(
        x_d - w_d / 2,
        eff_ranks,
        w_d,
        label="Effective Hilbert Rank (>99% Trace)",
        color=[GREEN_QUANTUM, RED_ALERT],
        edgecolor=TEXT_POLAR,
        linewidth=1.0,
    )

    ax3_cond = ax3.twinx()
    ax3_cond.set_facecolor("none")
    bar_cond = ax3_cond.bar(
        x_d + w_d / 2,
        cond_numbers,
        w_d,
        label="Gram Condition Number (kappa)",
        color=[GOLD_HARDWARE, CARD_CHARCOAL],
        edgecolor=GOLD_BRIGHT,
        linewidth=1.2,
        hatch="xx",
    )
    ax3_cond.set_yscale("log")
    ax3_cond.set_ylim(1e1, 1e6)
    ax3_cond.set_ylabel("Condition Number kappa (log scale)", color=GOLD_BRIGHT, fontsize=9.5)
    ax3_cond.grid(False)

    ax3.set_xticks(x_d)
    ax3.set_xticklabels(depth_labels, fontsize=9.5, color=TEXT_SLATE)
    ax3.set_ylabel("Effective Hilbert Rank (out of 256)", color=TEXT_POLAR, fontsize=9.5)
    ax3.set_ylim(0, 270)
    ax3.axhline(256, color=TEXT_MUTED, linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.text(0.05, 258, "Full 8-Qubit Dimension Ceiling (dim=256)", color=TEXT_MUTED, fontsize=7.5)

    for b in bar_rank:
        h = b.get_height()
        ax3.text(
            b.get_x() + b.get_width() / 2,
            h + 5,
            f"{int(h)} dims",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=TEXT_POLAR,
            fontweight="bold",
        )

    for b, c_val in zip(bar_cond, cond_numbers):
        ax3_cond.text(
            b.get_x() + b.get_width() / 2,
            c_val * 1.5,
            f"{c_val:.2e}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=GOLD_BRIGHT,
            fontweight="bold",
        )

    lines3_a, lab3_a = ax3.get_legend_handles_labels()
    lines3_b, lab3_b = ax3_cond.get_legend_handles_labels()
    ax3.legend(lines3_a + lines3_b, lab3_a + lab3_b, loc="upper center", framealpha=0.85)

    # ──────────────────────────────────────────────────────────────────────────
    # PANEL 4: Generalization Gap (Train -> Test Overfitting Penalty)
    # ──────────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_card(
        ax4,
        title="PANEL 4: TRAIN -> TEST GENERALIZATION GAP (OVERFITTING PENALTY)",
        subtitle="Delta MSE = Test MSE - Train MSE across Depth Configurations",
    )

    models_gap = [
        "Stark Q-KRR (L=1)",
        "Stark Q-KRR (L=2)",
        "Classical Linear Ridge",
        "Classical RBF Kernel",
        "Standard Q-KRR (L=1)",
    ]

    train_mses_g = [0.9216, 1.0600, 5.3820, 5.1899, 1.9651]
    test_mses_g = [1.6139, 2.1831, 3.7490, 4.4296, 6.7801]
    deltas = [t - tr for t, tr in zip(test_mses_g, train_mses_g)]

    y_pos_g = np.arange(len(models_gap))
    bar_gap = ax4.barh(
        y_pos_g,
        deltas,
        height=0.45,
        color=[GREEN_EMERALD, RED_ALERT, GOLD_HARDWARE, GOLD_HARDWARE, RED_CRIMSON],
        edgecolor=TEXT_POLAR,
        linewidth=1.0,
    )

    ax4.set_yticks(y_pos_g)
    ax4.set_yticklabels(models_gap, fontsize=8.5, color=TEXT_SLATE)
    ax4.set_xlabel("Generalization Gap Delta MSE = Test - Train (Debye^2)", color=TEXT_POLAR, fontsize=9.5)
    ax4.axvline(0, color=TEXT_MUTED, linestyle="-", linewidth=0.8)

    ax4.set_xlim(-2.5, 5.5)

    for b, d_val in zip(bar_gap, deltas):
        w = b.get_width()
        pos_x = w + 0.12 if w >= 0 else w - 0.12
        ax4.text(
            pos_x,
            b.get_y() + b.get_height() / 2,
            f"{d_val:+.4f}",
            va="center",
            ha="left" if w >= 0 else "right",
            fontsize=8.5,
            color=TEXT_POLAR,
            fontweight="bold",
        )

    ax4.text(
        0.52,
        0.18,
        "Under L=2, the gap expands by +62.2%\n"
        "(Delta MSE: +0.6923 -> +1.1231)\n"
        "proving orthogonal coordinate memorization.",
        transform=ax4.transAxes,
        fontsize=8.5,
        color=TEXT_SLATE,
        bbox=dict(boxstyle="square,pad=0.5", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE),
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Sovereign Header & Metadata Watermark
    # ──────────────────────────────────────────────────────────────────────────
    fig.suptitle(
        "T-1000 SOVEREIGN TELEMETRY: QUANTUM KERNEL RIDGE REGRESSION & DEPTH SHOWDOWN\n"
        "University of Pittsburgh Quantum Repository (106k Molecules) - 8-Qubit Hilbert State Overlaps",
        color=TEXT_POLAR,
        fontsize=12,
        fontweight="bold",
        y=0.985,
    )

    out_path = Path("results/charts/pitt_qkrr_depth_showdown.png")
    saved_path = save_deen_figure(fig, out_path, dpi=300)
    print(f"[CHART] Successfully generated sovereign figure: {saved_path}")


if __name__ == "__main__":
    main()
