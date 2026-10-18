# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visual Telemetry: Multi-Exit Topological Glued-Tree Valve & Quantum Router.
Illustrates deterministic ballistic wavepacket routing into target exit trees via
Aharonov-Bohm phase holonomy and dark-subspace extinction across K-way welded trees.
Strictly conforms to Deen OS Sovereign Visual Specification.
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

    report_path = Path("results/experiments/glued_tree_valve_report.json")
    assert report_path.exists(), f"Missing report: {report_path}"

    with open(report_path, "r") as f:
        data = json.load(f)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG_OBSIDIAN)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Target Hit Probability: Quantum Valve vs Classical 1/K Dispersion
    style_card(
        ax1,
        title="TARGET ROUTING PROBABILITY: QUANTUM VS CLASSICAL",
        subtitle="Ballistic Peak Hit % vs Classical 1/K Dispersion Floor",
    )
    depths_k2 = [r["depth"] for r in data["benchmarks_k2"]]
    q_hit_k2 = [r["quantum_target_peak_prob"] * 100.0 for r in data["benchmarks_k2"]]
    c_hit_k2 = [r["classical_target_hit_pct"] for r in data["benchmarks_k2"]]

    depths_k4 = [r["depth"] for r in data["benchmarks_k4"]]
    q_hit_k4 = [r["quantum_target_peak_prob"] * 100.0 for r in data["benchmarks_k4"]]
    c_hit_k4 = [r["classical_target_hit_pct"] for r in data["benchmarks_k4"]]

    ax1.plot(depths_k2, q_hit_k2, color=GREEN_QUANTUM, marker="o", linewidth=2.2, label="K=2 Quantum Valve Target")
    ax1.plot(depths_k2, c_hit_k2, color=GOLD_BRIGHT, linestyle="--", marker="x", linewidth=1.5, label="K=2 Classical Walk (~50%)")
    ax1.plot(depths_k4, q_hit_k4, color=GREEN_FLUX, marker="s", linewidth=2.2, label="K=4 Quantum Valve Target")
    ax1.plot(depths_k4, c_hit_k4, color=RED_ALERT, linestyle="--", marker="^", linewidth=1.5, label="K=4 Classical Walk (~25%)")

    ax1.set_xlabel("Tree Depth (n)", color=TEXT_POLAR, fontsize=10)
    ax1.set_ylabel("Target Exit Hit (%)", color=TEXT_POLAR, fontsize=10)
    ax1.set_ylim(15, 88)
    ax1.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=TEXT_MUTED, labelcolor=TEXT_POLAR)

    # 2. Extinction Contrast Ratio (Target vs Non-Target Exits)
    style_card(
        ax2,
        title="TOPOLOGICAL EXTINCTION CONTRAST RATIO",
        subtitle="Aharonov-Bohm Dark-Subspace Suppression P(Target) / P(Non-Target)",
    )
    contrast_k2 = [r["quantum_routing_contrast"] for r in data["benchmarks_k2"]]
    contrast_k4 = [r["quantum_routing_contrast"] for r in data["benchmarks_k4"]]

    ax2.plot(depths_k2, contrast_k2, color=GOLD_BRIGHT, marker="o", linewidth=2.4, label="K=2 Valve Contrast (~1,708:1)")
    ax2.plot(depths_k4, contrast_k4, color=GREEN_EMERALD, marker="s", linewidth=2.4, label="K=4 Valve Contrast (~522:1)")
    ax2.axhline(1.0, color=RED_ALERT, linestyle=":", label="Classical Unbiased Baseline (1:1)")

    ax2.set_yscale("log")
    ax2.set_xlabel("Tree Depth (n)", color=TEXT_POLAR, fontsize=10)
    ax2.set_ylabel("Routing Contrast Ratio (Log Scale)", color=TEXT_POLAR, fontsize=10)
    ax2.legend(loc="center right", facecolor=CARD_CHARCOAL, edgecolor=TEXT_MUTED, labelcolor=TEXT_POLAR)

    # 3. K = 4 Detailed Spatiotemporal Exit Wavepacket Dynamics
    style_card(
        ax3,
        title="K=4 DYNAMIC ROUTING TRAJECTORY (TARGET = EXIT 2)",
        subtitle="Selective Flow into Exit 2 with Quenched Exits 0, 1, 3",
    )
    traj = data["routing_trajectory_k4"]
    t = [s["time"] for s in traj]
    p_in = [s["p_entrance"] * 100.0 for s in traj]
    p_targ = [s["p_target_exit"] * 100.0 for s in traj]
    p_comp = [s["p_non_target_sum"] * 100.0 for s in traj]
    p_weld = [s["p_weld_interface"] * 100.0 for s in traj]

    ax3.plot(t, p_in, color=TEXT_MUTED, linestyle=":", linewidth=1.5, label="ROOT_IN")
    ax3.plot(t, p_weld, color=GOLD_HARDWARE, linestyle="--", linewidth=1.5, label="Weld Interface (2^n Leaves)")
    ax3.plot(t, p_targ, color=GREEN_QUANTUM, linewidth=2.6, label="ROOT_OUT[2] (Target Hit)")
    ax3.plot(t, p_comp, color=RED_ALERT, linewidth=1.8, label="ROOT_OUT[0,1,3] (Non-Target Sum)")

    ax3.set_xlabel("Evolution Time (t)", color=TEXT_POLAR, fontsize=10)
    ax3.set_ylabel("Probability Density (%)", color=TEXT_POLAR, fontsize=10)
    ax3.set_ylim(-2, 105)
    ax3.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=TEXT_MUTED, labelcolor=TEXT_POLAR)

    # 4. Architecture & Physical Invariants Summary
    style_card(
        ax4,
        title="QUANTUM VALVE ARCHITECTURE & SUMMARY",
        subtitle="Pure AVX2 Topological Telemetry Invariants",
    )
    ax4.axis("off")
    specs = [
        ("Architecture", "Multi-Exit Topological Glued-Tree Valve"),
        ("Symmetric Reduction", "O(K * 2^n) Graph -> (K+1)(n+1) 1D Chain"),
        ("K=2 Contrast Ratio", f"{data['benchmarks_k2'][-1]['quantum_routing_contrast']:.1f} : 1 (Extinction: 99.94%)"),
        ("K=4 Contrast Ratio", f"{data['benchmarks_k4'][-1]['quantum_routing_contrast']:.1f} : 1 (Extinction: 99.81%)"),
        ("K=4 Peak Target Hit", f"{data['benchmarks_k4'][0]['quantum_target_peak_prob']*100.0:.2f}% (vs 25.0% Classical)"),
        ("Ballistic Group Vel", "v_g \u2248 1.33 columns / unit time"),
        ("Classical Trap Limit", "Omega(2^{n/2}) with 1/K random dispersion"),
        ("Aharonov-Bohm Phase", "phi_y = \u03c0 on non-target (Dark State)"),
    ]

    y_pos = 0.88
    for label, val in specs:
        ax4.text(0.05, y_pos, f"{label}:", color=GOLD_BRIGHT, fontsize=10, fontweight="bold")
        ax4.text(0.53, y_pos, str(val), color=TEXT_POLAR, fontsize=10)
        y_pos -= 0.11

    out_path = Path("results/charts/glued_tree_valve_telemetry.png")
    save_deen_figure(fig, out_path)
    print(f"Chart saved to: {out_path}")

if __name__ == "__main__":
    generate_telemetry_chart()
