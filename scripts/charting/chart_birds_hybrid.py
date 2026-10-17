# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visual Telemetry: Waterbirds 11k Robustness & Subpopulation Generalization.
Compares Raw Amplitude-Encoded 32x32 baseline against the MobileNetV3 + 10Q Angle Transfer QNN.
Strictly conforms to the Deen OS Sovereign Visual Specification.
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

    raw_report_path = Path("data/birds/birds_qnn_report.json")
    hybrid_report_path = Path("data/birds/birds_hybrid_qnn_report.json")

    with open(hybrid_report_path, "r") as f:
        hybrid_data = json.load(f)

    with open(raw_report_path, "r") as f:
        raw_data = json.load(f)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG_OBSIDIAN)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Training & Test Accuracy Progression
    style_card(
        ax1,
        title="HYBRID VQC ACCURACY TRAJECTORY",
        subtitle="Convergence across 15 Epochs with Two-Speed AdamW",
    )
    epochs = [h["epoch"] for h in hybrid_data["history"]]
    trn_acc = [h["train_accuracy_pct"] for h in hybrid_data["history"]]
    tst_acc = [h["test_accuracy_pct"] for h in hybrid_data["history"]]
    wga = [h["worst_group_accuracy_pct"] for h in hybrid_data["history"]]

    ax1.plot(epochs, trn_acc, color=GREEN_EMERALD, marker="o", linewidth=2.0, label="Train Accuracy %")
    ax1.plot(epochs, tst_acc, color=GOLD_BRIGHT, marker="s", linewidth=1.8, label="Test Accuracy %")
    ax1.plot(epochs, wga, color=RED_ALERT, marker="^", linestyle="--", linewidth=1.6, label="Worst-Group Accuracy (WGA) %")
    ax1.axhline(50.0, color=TEXT_MUTED, linestyle=":", alpha=0.5, label="Chance Floor (50%)")
    ax1.set_xlabel("Epoch", color=TEXT_MUTED)
    ax1.set_ylabel("Accuracy %", color=TEXT_MUTED)
    ax1.set_ylim(30, 90)
    ax1.legend(loc="lower right")

    # 2. 4-Group Accuracy Breakdown: Raw Pixels vs Hybrid Transfer
    style_card(
        ax2,
        title="4-GROUP GENERALIZATION BREAKDOWN",
        subtitle="Raw Pixels (Shortcut Collapse) vs Hybrid Transfer QNN",
    )
    groups = ["G0: Land/Land", "G1: Land/Water", "G2: Water/Land", "G3: Water/Water"]
    raw_snap = raw_data["best_checkpoint_evaluation"]
    hyb_snap = hybrid_data["best_checkpoint_evaluation"]

    raw_vals = [
        raw_snap["g0_land_land"]["accuracy_pct"],
        raw_snap["g1_land_water"]["accuracy_pct"],
        raw_snap["g2_water_land"]["accuracy_pct"],
        raw_snap["g3_water_water"]["accuracy_pct"],
    ]
    hyb_vals = [
        hyb_snap["g0_land_land"]["accuracy_pct"],
        hyb_snap["g1_land_water"]["accuracy_pct"],
        hyb_snap["g2_water_land"]["accuracy_pct"],
        hyb_snap["g3_water_water"]["accuracy_pct"],
    ]

    x = np.arange(len(groups))
    width = 0.35
    ax2.bar(x - width/2, raw_vals, width, label="Raw Pixels (Amplitude)", color=CARD_CHARCOAL, edgecolor=TEXT_MUTED)
    ax2.bar(x + width/2, hyb_vals, width, label="Hybrid Transfer (10Q Angle)", color=GREEN_QUANTUM, edgecolor=GREEN_EMERALD)
    ax2.set_ylabel("Accuracy %", color=TEXT_MUTED)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, rotation=15, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 100)
    ax2.axhline(50.0, color=TEXT_MUTED, linestyle=":", alpha=0.5)
    ax2.legend(loc="upper right")

    # 3. Loss Dynamics (Train vs Test)
    style_card(
        ax3,
        title="HYBRID LOSS DYNAMICS",
        subtitle="Softmax Cross-Entropy Loss under Balanced Minibatching",
    )
    trn_loss = [h["train_loss"] for h in hybrid_data["history"]]
    tst_loss = [h["test_loss"] for h in hybrid_data["history"]]
    ax3.plot(epochs, trn_loss, color=GREEN_FLUX, linewidth=2.0, marker="o", label="Train Loss")
    ax3.plot(epochs, tst_loss, color=GOLD_HARDWARE, linewidth=1.8, marker="s", label="Test Loss")
    ax3.set_xlabel("Epoch", color=TEXT_MUTED)
    ax3.set_ylabel("Cross-Entropy Loss", color=TEXT_MUTED)
    ax3.legend(loc="upper right")

    # 4. L1 Bloch Coherence Scaling
    style_card(
        ax4,
        title="QUANTUM SUPERPOSITION DYNAMICS",
        subtitle="Mean L1 Off-Diagonal Coherence Trajectory",
    )
    coh = [h["mean_coherence_l1"] for h in hybrid_data["history"]]
    ax4.plot(epochs, coh, color=GREEN_EMERALD, linewidth=2.2, marker="D", label="Mean L1 Coherence")
    ax4.set_xlabel("Epoch", color=TEXT_MUTED)
    ax4.set_ylabel("Coherence Metric", color=TEXT_MUTED)
    ax4.legend(loc="lower right")

    out_path = Path("results/charts/waterbirds_hybrid_qnn_telemetry.png")
    save_deen_figure(fig, out_path, dpi=300)
    print(f"Sovereign telemetry figure saved to: {out_path}")

if __name__ == "__main__":
    generate_telemetry_chart()
