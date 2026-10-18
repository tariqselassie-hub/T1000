# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visual Telemetry: Cascaded Multi-Tier Topological Glued-Tree Valve.
Illustrates sequential ballistic wavepacket routing across multi-tier weld bottlenecks,
highlighting focusing at intermediate roots, final exit arrival, and dark-subspace extinction.
Strictly adheres to Deen OS Sovereign Visual Specification.
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
    TEXT_MUTED,
)

def generate_telemetry_chart():
    apply_deen_theme()

    report_path = Path("results/experiments/cascaded_tree_valve_report.json")
    assert report_path.exists(), f"Missing report: {report_path}"

    with open(report_path, "r") as f:
        data = json.load(f)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG_OBSIDIAN)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Target Hit %: Quantum Cascaded Valve vs Classical Random Walk Floor
    style_card(
        ax1,
        title="CASCADED TARGET HIT: QUANTUM VS CLASSICAL",
        subtitle="Dual-Weld Ballistic Peak Hit % vs Multi-Tier Random Dispersal Floor",
    )
    labels_2x2 = [f"{r['depth1']}+{r['depth2']}" for r in data["benchmarks_2x2"]]
    q_hit_2x2 = [r["quantum_target_peak_prob"] * 100.0 for r in data["benchmarks_2x2"]]
    c_hit_2x2 = [r["classical_target_hit_pct"] for r in data["benchmarks_2x2"]]

    labels_2x4 = [f"{r['depth1']}+{r['depth2']}" for r in data["benchmarks_2x4"]]
    q_hit_2x4 = [r["quantum_target_peak_prob"] * 100.0 for r in data["benchmarks_2x4"]]
    c_hit_2x4 = [r["classical_target_hit_pct"] for r in data["benchmarks_2x4"]]

    x_idx_2x2 = np.arange(len(labels_2x2))
    x_idx_2x4 = np.arange(len(labels_2x4))

    ax1.plot(x_idx_2x2, q_hit_2x2, color=GREEN_QUANTUM, marker="o", linewidth=2.4, label="2x2 Quantum Valve Target (4 Exits)")
    ax1.plot(x_idx_2x2, c_hit_2x2, color=GOLD_BRIGHT, linestyle="--", marker="x", linewidth=1.5, label="2x2 Classical Walk (~25%)")
    ax1.plot(x_idx_2x4, q_hit_2x4, color=GREEN_FLUX, marker="s", linewidth=2.4, label="2x4 Quantum Valve Target (8 Exits)")
    ax1.plot(x_idx_2x4, c_hit_2x4, color=RED_ALERT, linestyle="--", marker="^", linewidth=1.5, label="2x4 Classical Walk (~12.5%)")

    ax1.set_xticks(x_idx_2x2)
    ax1.set_xticklabels(labels_2x2, color=TEXT_POLAR)
    ax1.set_xlabel("Tier Depths (n1 + n2)", color=TEXT_POLAR, fontsize=10)
    ax1.set_ylabel("Target Exit Hit (%)", color=TEXT_POLAR, fontsize=10)
    ax1.set_ylim(5, 75)
    ax1.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=TEXT_MUTED, labelcolor=TEXT_POLAR)

    # 2. Sequential Wavepacket Trajectory across Dual Welds
    style_card(
        ax2,
        title="SEQUENTIAL WAVE PACKET TRAJECTORY (n1=3, n2=3)",
        subtitle="Coherent Cascading: Entrance -> Intermediate Root -> Final Exit",
    )
    traj = data["routing_trajectory"]
    t = [s["time"] for s in traj]
    p_in = [s["p_entrance"] * 100.0 for s in traj]
    p_mid = [s["p_target_intermediate"] * 100.0 for s in traj]
    p_exit = [s["p_target_final_exit"] * 100.0 for s in traj]
    p_sib = [s["p_sibling_exits"] * 100.0 for s in traj]
    p_cross = [s["p_cross_class_exits"] * 100.0 for s in traj]

    ax2.plot(t, p_in, color=TEXT_MUTED, linestyle=":", linewidth=1.5, label="ROOT_IN (Col 0)")
    ax2.plot(t, p_mid, color=GOLD_HARDWARE, linewidth=2.0, label="Intermediate Root MID[0]")
    ax2.plot(t, p_exit, color=GREEN_QUANTUM, linewidth=2.6, label="Final Exit OUT[0, 1] (Target)")
    ax2.plot(t, p_sib, color=GOLD_BRIGHT, linestyle="--", linewidth=1.5, label="Sibling Exit OUT[0, 0] (Leak)")
    ax2.plot(t, p_cross, color=RED_ALERT, linestyle="-.", linewidth=1.5, label="Cross-Class Exits OUT[1, *]")

    ax2.set_xlabel("Evolution Time (t)", color=TEXT_POLAR, fontsize=10)
    ax2.set_ylabel("Probability Density (%)", color=TEXT_POLAR, fontsize=10)
    ax2.set_ylim(-2, 105)
    ax2.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=TEXT_MUTED, labelcolor=TEXT_POLAR)

    # 3. Routing Contrast Ratio across Both Tiers
    style_card(
        ax3,
        title="CASCADED EXTINCTION CONTRAST RATIO",
        subtitle="Multi-Tier Dark-Subspace Suppression Ratio P(Target) / P(Non-Target)",
    )
    contrast_2x2 = [r["quantum_routing_contrast"] for r in data["benchmarks_2x2"]]
    contrast_2x4 = [r["quantum_routing_contrast"] for r in data["benchmarks_2x4"]]

    ax3.plot(x_idx_2x2, contrast_2x2, color=GOLD_BRIGHT, marker="o", linewidth=2.4, label="2x2 Contrast (~2,489:1)")
    ax3.plot(x_idx_2x4, contrast_2x4, color=GREEN_EMERALD, marker="s", linewidth=2.4, label="2x4 Contrast (~831:1)")
    ax3.axhline(1.0, color=RED_ALERT, linestyle=":", label="Classical Unbiased Baseline (1:1)")

    ax3.set_yscale("log")
    ax3.set_xticks(x_idx_2x2)
    ax3.set_xticklabels(labels_2x2, color=TEXT_POLAR)
    ax3.set_xlabel("Tier Depths (n1 + n2)", color=TEXT_POLAR, fontsize=10)
    ax3.set_ylabel("Contrast Ratio (Log Scale)", color=TEXT_POLAR, fontsize=10)
    ax3.legend(loc="center right", facecolor=CARD_CHARCOAL, edgecolor=TEXT_MUTED, labelcolor=TEXT_POLAR)

    # 4. Architecture Specifications & Key Metrics Summary
    style_card(
        ax4,
        title="CASCADED TOPOLOGICAL ARCHITECTURE & METRICS",
        subtitle="Dual-Weld Deep Quantum Router Telemetry Summary",
    )
    ax4.axis("off")
    specs = [
        ("Architecture", "Cascaded Multi-Tier Glued-Tree Valve"),
        ("Hierarchy Model", "Tier 1 (Coarse K1) -> Tier 2 (Fine K2)"),
        ("Terminal Outputs", "K1 x K2 = 4 (2x2) or 8 (2x4) distinct roots"),
        ("2x2 Contrast Ratio", f"{data['benchmarks_2x2'][-1]['quantum_routing_contrast']:.1f} : 1 (Extinction: 99.96%)"),
        ("2x4 Contrast Ratio", f"{data['benchmarks_2x4'][-1]['quantum_routing_contrast']:.1f} : 1 (Extinction: 99.88%)"),
        ("Dual-Weld Transit", "Zero intermediate measurement collapse"),
        ("Classical Hit Limit", "1 / (K1 x K2) with Omega(2^{(n1+n2)/2}) trapping"),
        ("Ballistic Group Vel", "v_g \u2248 1.33 columns / unit time across tiers"),
    ]

    y_pos = 0.88
    for label, val in specs:
        ax4.text(0.05, y_pos, f"{label}:", color=GOLD_BRIGHT, fontsize=10, fontweight="bold")
        ax4.text(0.53, y_pos, str(val), color=TEXT_POLAR, fontsize=10)
        y_pos -= 0.11

    out_path = Path("results/charts/cascaded_tree_valve_telemetry.png")
    save_deen_figure(fig, out_path)
    print(f"Chart saved to: {out_path}")

if __name__ == "__main__":
    generate_telemetry_chart()
