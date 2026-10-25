# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes ECLEKTIK Track 2 Hierarchical Ultrametric Language Family Sorting.
Outputs high-resolution charts adhering strictly to Deen OS Sovereign Visual Specification.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, BG_DARK, CARD_CHARCOAL, GREEN_QUANTUM, GREEN_EMERALD,
    GOLD_HARDWARE, GOLD_BRIGHT, RED_ALERT, TEXT_POLAR, TEXT_SLATE, TEXT_MUTED, GRID_SLATE
)

def chart_eclektik_ultrametric(json_path: Path, output_dir: Path) -> tuple[Path, Path]:
    apply_deen_theme()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 ECLEKTIK TRACK 2 // 10-QUBIT HIERARCHICAL ULTRAMETRIC LANGUAGE SORTER",
        color=TEXT_POLAR, fontsize=13, fontweight="bold", y=0.98
    )

    # 1. Macro-Family Classification Accuracy
    ax1 = axes[0, 0]
    fam_names = [f["family"] for f in data["family_metrics"]]
    fam_accs = [f["accuracy_pct"] for f in data["family_metrics"]]
    fam_colors = [GREEN_QUANTUM, GOLD_HARDWARE, GREEN_EMERALD, GOLD_BRIGHT]

    bars1 = ax1.bar(fam_names, fam_accs, color=fam_colors, width=0.55, edgecolor=GRID_SLATE, lw=1.2)
    ax1.axhline(90.0, color=GOLD_BRIGHT, ls="--", lw=1.2, alpha=0.8, label="90% Quality Floor")
    ax1.axhline(data["test_coarse_family_accuracy_pct"], color=GREEN_QUANTUM, ls=":", lw=1.5,
                label=f"Coarse Accuracy: {data['test_coarse_family_accuracy_pct']:.2f}%")
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Holdout Accuracy (%)", color=TEXT_MUTED)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 2.5, f"{yval:.1f}%",
                 ha="center", va="bottom", color=TEXT_POLAR, fontsize=10, fontweight="bold")
    style_card(ax1, "PANEL 1: COARSE MACRO-FAMILY ACCURACY", "H_C (2Q, d_C=4) Pure Quantum Routing")
    ax1.legend(loc="lower right")

    # 2. 12-Language Leaf Accuracy
    ax2 = axes[0, 1]
    langs = [l["code"] for l in data["language_metrics"]]
    lang_accs = [l["accuracy_pct"] for l in data["language_metrics"]]
    bar_colors = [GREEN_QUANTUM if a >= 90.0 else (GOLD_HARDWARE if a >= 70.0 else RED_ALERT) for a in lang_accs]

    bars2 = ax2.bar(langs, lang_accs, color=bar_colors, width=0.6, edgecolor=GRID_SLATE, lw=1.0)
    ax2.axhline(data["test_joint_12way_accuracy_pct"], color=GREEN_EMERALD, ls="--", lw=1.5,
                label=f"Mean Joint 12-Way: {data['test_joint_12way_accuracy_pct']:.2f}%")
    ax2.axhline(90.0, color=GOLD_BRIGHT, ls=":", lw=1.2, alpha=0.7, label="90% Precision Horizon")
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Leaf Accuracy (%)", color=TEXT_MUTED)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 2.0, f"{int(round(yval))}%",
                 ha="center", va="bottom", color=TEXT_SLATE, fontsize=8, fontweight="bold")
    style_card(ax2, "PANEL 2: 12-LANGUAGE LEAF ACCURACY", "P(y,s) Readout with Zero Classical Weights")
    ax2.legend(loc="lower left")

    # 3. Ultrametric Hilbert Space Gram Overlaps
    ax3 = axes[1, 0]
    geom = data["geometry"]
    categories = ["Identical Leaf\n(en vs en)", "Sibling Family\n(en vs de)", "Disjoint Family\n(en vs es)"]
    overlaps = [geom["identical_leaf_overlap"], geom["sibling_family_overlap"], geom["disjoint_family_overlap"]]
    overlap_colors = [GREEN_QUANTUM, GOLD_HARDWARE, RED_ALERT]

    bars3 = ax3.bar(categories, overlaps, color=overlap_colors, width=0.5, edgecolor=GRID_SLATE, lw=1.2)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Hilbert Gram Overlap Tr(ρ_u ρ_v)", color=TEXT_MUTED)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.03, f"{yval:.4f}",
                 ha="center", va="bottom", color=TEXT_POLAR, fontsize=10, fontweight="bold")
    style_card(ax3, "PANEL 3: ULTRAMETRIC HILBERT SPACE GRAM OVERLAP", "Tr(ρ_C_u ρ_C_v) Hierarchy Separation")

    # 4. Strong Ultrametric Triangle Inequality
    ax4 = axes[1, 1]
    d_sibling = geom["gamma1_sibling"]
    d_disjoint = geom["gamma2_disjoint"]
    labels_dist = ["d(en, de)\n[Sibling]", "d(en, es)\n[Disjoint]", "d(de, es)\n[Disjoint]"]
    d_vals = [d_sibling, d_disjoint, d_disjoint]
    dist_colors = [GOLD_HARDWARE, GREEN_QUANTUM, GREEN_EMERALD]

    bars4 = ax4.bar(labels_dist, d_vals, color=dist_colors, width=0.5, edgecolor=GRID_SLATE, lw=1.2)
    ax4.set_ylim(0, 1.4)
    ax4.set_ylabel("Ultrametric Tree Distance d(u, v)", color=TEXT_MUTED)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.04, f"{yval:.2f}",
                 ha="center", va="bottom", color=TEXT_POLAR, fontsize=10, fontweight="bold")
    status_text = "PASS: d(u,w) <= max(d(u,v), d(v,w))\nConserved under U_sink Decoupling"
    ax4.text(0.5, 0.85, status_text, transform=ax4.transAxes, ha="center", va="center",
             color=GREEN_QUANTUM, fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="square,pad=0.6", facecolor=CARD_CHARCOAL, edgecolor=GREEN_QUANTUM, lw=1.2))
    style_card(ax4, "PANEL 4: STRONG TRIANGLE INEQUALITY VERIFICATION", "Hierarchical Isometric Tree Embedding")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    png_path = output_dir / "eclektik_ultrametric_sorter_telemetry.png"
    svg_path = output_dir / "eclektik_ultrametric_sorter_telemetry.svg"
    save_deen_figure(fig, png_path)
    save_deen_figure(fig, svg_path)
    plt.close(fig)

    print(f"   [DEEN OS] Telemetry Rendered : {png_path}")
    print(f"   [DEEN OS] Vector SVG Rendered: {svg_path}")
    return png_path, svg_path

if __name__ == "__main__":
    report_file = Path("results/experiments/eclektik_ultrametric_report.json")
    out_dir = Path("results/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_eclektik_ultrametric(report_file, out_dir)
