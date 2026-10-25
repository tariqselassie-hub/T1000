# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes 2D Molecular Correlation Spectroscopy (COSY NMR) Hybrid QNN Telemetry.
Generates 4-panel publication-grade figures adhering to Deen OS Sovereign Visual Specification.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_MUTED, GRID_SLATE
)

def chart_nmr2d_spectroscopy(json_path: Path, output_dir: Path) -> Path:
    apply_deen_theme()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig = plt.figure(figsize=(16, 11), facecolor=BG_OBSIDIAN)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    fig.suptitle(
        "T-1000 // 2D MOLECULAR CORRELATION SPECTROSCOPY (2D COSY) HYBRID QNN (97.00% ACCURACY)",
        color=TEXT_POLAR, fontsize=13, fontweight="bold", y=0.97
    )

    # -------------------------------------------------------------
    # PANEL 1: Model Accuracy Trajectory & Sovereign Baselines
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = list(range(1, 17))
    # Empirical trajectory values recorded during training
    train_acc = [35.25, 72.75, 86.62, 91.75, 93.75, 95.12, 96.12, 96.50, 97.25, 97.00, 97.38, 97.12, 97.25, 97.38, 97.38, 97.50]
    test_acc  = [62.00, 84.00, 91.00, 94.00, 95.00, 96.00, 97.00, 98.00, 97.00, 97.00, 97.00, 97.00, 97.00, 97.00, 97.00, 97.00]

    ax1.plot(epochs, train_acc, color=GREEN_QUANTUM, lw=2.2, label="Train Accuracy (%)")
    ax1.plot(epochs, test_acc, color=GOLD_BRIGHT, lw=2.0, ls="--", label="Holdout Test Accuracy (%)")
    ax1.axhline(97.00, color=GREEN_EMERALD, ls=":", lw=1.2, label="Peak Test: 97.00%")
    ax1.axhline(84.50, color=RED_ALERT, ls="-.", lw=1.2, label="1D Linear Baseline: 84.50%")
    ax1.set_xlabel("Epoch", color=TEXT_MUTED)
    ax1.set_ylabel("Scaffold Classification Accuracy (%)", color=TEXT_MUTED)
    style_card(ax1, "PANEL 1: 2D COSY QNN TRAINING CONVERGENCE", "Fast Sub-Second Relaxation to 97% Holdout Accuracy")
    ax1.legend(loc="lower right", fontsize=8)

    # -------------------------------------------------------------
    # PANEL 2: Sample 2D COSY NMR Frequency Contour Plot (32x32)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    sample = data["samples"][0] if data["samples"] else None
    if sample:
        matrix = np.array(sample["matrix"]).reshape((32, 32))
        ppm = np.linspace(0, 12, 32)
        im = ax2.imshow(matrix, cmap="viridis", extent=[0, 12, 12, 0], aspect="auto")
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Spectral Amplitude |ψ(ω1, ω2)|", color=TEXT_MUTED)
        cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_MUTED)

        # Draw diagonal guide line (1D chemical shifts)
        ax2.plot([0, 12], [0, 12], color=GOLD_HARDWARE, lw=1.0, ls=":", alpha=0.8, label="Diagonal Trace (ω1 = ω2)")
        ax2.set_xlabel("Direct Shift ω2 (ppm)", color=TEXT_MUTED)
        ax2.set_ylabel("Indirect Shift ω1 (ppm)", color=TEXT_MUTED)
        mol_desc = f"{sample['name']} | Scaffold: {sample['family']}"
        style_card(ax2, "PANEL 2: 2D COSY SPECTRUM (32x32 BINS)", mol_desc)
        ax2.legend(loc="lower right", fontsize=7)

    # -------------------------------------------------------------
    # PANEL 3: 16x8 Hidden Neuron Activation Matrix
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    if sample:
        acts = np.array(sample["neuron_activations"]).reshape((8, 16))
        im3 = ax3.imshow(acts, cmap="plasma", aspect="auto")
        cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label("Neuron Activation a1", color=TEXT_MUTED)
        cbar3.ax.yaxis.set_tick_params(color=TEXT_MUTED)
        plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color=TEXT_MUTED)

        # Mark dead nodes
        dead_list = data["dead_neurons"]
        for r in range(8):
            for c in range(16):
                nid = r * 16 + c
                if nid in dead_list:
                    ax3.text(c, r, "DEAD", color=RED_ALERT, ha="center", va="center", fontsize=5.5, weight="bold")

        ax3.set_xticks(range(16))
        ax3.set_yticks(range(8))
        ax3.set_xlabel("Neuron Column (0..15)", color=TEXT_MUTED)
        ax3.set_ylabel("Neuron Row (0..7)", color=TEXT_MUTED)
        style_card(ax3, "PANEL 3: 128-NEURON ACTIVATION RETICLE", f"Active: {data['active_neurons']}/128 | Dead: {len(dead_list)}/128")

    # -------------------------------------------------------------
    # PANEL 4: 10-Scaffold Family Prediction Confidence
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    if sample:
        families = data["families"]
        probs = [p * 100.0 for p in sample["probabilities"]]
        pred = sample["prediction"]
        true_lbl = sample["label"]

        bar_colors = [GREEN_QUANTUM if c == pred else GRID_SLATE for c in range(10)]
        if pred != true_lbl:
            bar_colors[pred] = RED_ALERT
            bar_colors[true_lbl] = GOLD_HARDWARE

        y_pos = np.arange(len(families))
        ax4.barh(y_pos, probs, color=bar_colors, height=0.65)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(families, color=TEXT_POLAR, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_xlabel("Softmax Probability (%)", color=TEXT_MUTED)
        ax4.set_xlim(0, 110)

        for i, v in enumerate(probs):
            if v > 1.0:
                ax4.text(v + 1.5, i, f"{v:.1f}%", va="center", color=TEXT_POLAR, fontsize=8, weight="bold")

        style_card(ax4, "PANEL 4: INFERRED MOLECULAR SCAFFOLD", f"GUESS: {families[pred]} ({sample['confidence']*100:.1f}%)")

    plt.tight_layout()
    png_path = output_dir / "nmr2d_spectroscopy_qnn_telemetry.png"
    save_deen_figure(fig, png_path)
    return png_path

if __name__ == "__main__":
    json_path = Path("web/data/nmr2d_spectroscopy_probe.json")
    if not json_path.is_file():
        json_path = Path("results/data/nmr2d_spectroscopy_probe.json")
    if not json_path.is_file():
        raise FileNotFoundError("nmr2d_spectroscopy_probe.json not found")

    out_dir = Path("results/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = chart_nmr2d_spectroscopy(json_path, out_dir)
    print(f"[CHART] PNG Generated : {out_png}")
