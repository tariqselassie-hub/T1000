# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes MNIST Hybrid QCNN Neural Health, Neuron Firing Distribution, and Layer-to-Layer Guess Pipeline.
Generates publication-quality charts adhering strictly to Deen OS Sovereign Visual Specification.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, CARD_CHARCOAL, GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_MUTED, GRID_SLATE
)

def chart_mnist_neural_probe(json_path: Path, csv_path: Path, output_dir: Path) -> Path:
    apply_deen_theme()
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    df = pd.read_csv(csv_path) if csv_path.is_file() else None

    fig = plt.figure(figsize=(16, 11), facecolor=BG_OBSIDIAN)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    fig.suptitle(
        f"T-1000 HYBRID QCNN // NEURAL PROBE & FIRING DIAGNOSTICS ({data['test_accuracy_pct']:.2f}% TEST ACCURACY)",
        color=TEXT_POLAR, fontsize=13, fontweight="bold", y=0.97
    )

    # -------------------------------------------------------------
    # PANEL 1: Model Accuracy Trajectory & Sovereign Baselines
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    if df is not None:
        epochs = df["epoch"]
        ax1.plot(epochs, df["train_acc_pct"], color=GREEN_QUANTUM, lw=2.2, label="Train Accuracy (%)")
        ax1.plot(epochs, df["test_acc_pct"], color=GOLD_BRIGHT, lw=2.0, ls="--", label="Test Accuracy (%)")
        final_test = df["test_acc_pct"].iloc[-1]
        ax1.axhline(final_test, color=GREEN_EMERALD, ls=":", lw=1.2, alpha=0.9, label=f"Holdout: {final_test:.2f}%")
        ax1.axhline(84.50, color=RED_ALERT, ls="-.", lw=1.2, alpha=0.8, label="Linear VQC: 84.50%")
        ax1.axhline(81.25, color=GOLD_HARDWARE, ls=":", lw=1.0, alpha=0.7, label="HDC-10k: 81.25%")
        ax1.axhline(78.59, color=TEXT_MUTED, ls="--", lw=1.0, alpha=0.5, label="BNN 4096-bit: 78.59%")
        ax1.set_xlabel("Epoch", color=TEXT_MUTED)
        ax1.set_ylabel("Classification Accuracy (%)", color=TEXT_MUTED)
    style_card(ax1, "PANEL 1: ACCURACY & SOVEREIGN BENCHMARKS", "10-Qubit Hilbert + 65 Observables Breakthrough")
    ax1.legend(loc="lower right", fontsize=8)

    # -------------------------------------------------------------
    # PANEL 2: 128 Hidden Neuron Firing Rates & Health Distribution
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    stats = data["neuron_stats"]
    neuron_ids = [s["id"] for s in stats]
    firing_rates = [s["firing_rate"] * 100.0 for s in stats]
    is_dead = [s["is_dead"] for s in stats]

    colors = [RED_ALERT if dead else GREEN_QUANTUM for dead in is_dead]
    bars = ax2.bar(neuron_ids, firing_rates, color=colors, width=0.8, alpha=0.85)

    dead_ids = data["dead_neurons"]
    ax2.axhline(0, color=GRID_SLATE, lw=1.0)
    mean_rate = np.mean(firing_rates)
    ax2.axhline(mean_rate, color=GOLD_BRIGHT, ls="--", lw=1.2, label=f"Mean Firing Rate ({mean_rate:.1f}%)")

    # Annotate dead neurons
    for dead_id in dead_ids:
        ax2.scatter([dead_id], [0], color=RED_ALERT, s=40, zorder=5, marker="x")
        ax2.annotate(f"N{dead_id}\nDEAD", (dead_id, 3), color=RED_ALERT, fontsize=7, ha="center", weight="bold")

    ax2.set_xlabel("Neuron Index (0..127)", color=TEXT_MUTED)
    ax2.set_ylabel("Test Set Firing Frequency (%)", color=TEXT_MUTED)
    style_card(
        ax2,
        f"PANEL 2: 128-NEURON HEALTH DISTRIBUTION ({data['active_neurons_count']}/128 Active)",
        f"Dead Neurons: {data['dead_neurons_count']} {dead_ids} | Active: {data['active_neurons_count']}"
    )
    ax2.legend(loc="upper right", fontsize=8)

    # -------------------------------------------------------------
    # PANEL 3: 16x8 Hidden Neuron Activation Heatmap (Grid Matrix)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    # Reshape firing rates into 8x16 grid
    grid_rates = np.array(firing_rates).reshape((8, 16))
    im = ax3.imshow(grid_rates, cmap="viridis", aspect="auto", vmin=0, vmax=100)
    cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label("Firing Rate (%)", color=TEXT_MUTED)
    cbar.ax.yaxis.set_tick_params(color=TEXT_MUTED)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_MUTED)

    for r in range(8):
        for c in range(16):
            nid = r * 16 + c
            if nid in dead_ids:
                ax3.text(c, r, "DEAD", color=RED_ALERT, ha="center", va="center", fontsize=6, weight="bold")
            elif grid_rates[r, c] > 70:
                ax3.text(c, r, f"{int(grid_rates[r, c])}", color="#ffffff", ha="center", va="center", fontsize=6)

    ax3.set_xticks(range(16))
    ax3.set_yticks(range(8))
    ax3.set_xlabel("Neuron Column (0..15)", color=TEXT_MUTED)
    ax3.set_ylabel("Neuron Row (0..7)", color=TEXT_MUTED)
    style_card(ax3, "PANEL 3: 16x8 HIDDEN LAYER NEURAL RETICLE", "Spatial Activation Matrix & Dead Neuron Localization")

    # -------------------------------------------------------------
    # PANEL 4: Sample Forward Guess with Prediction Confidence
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    sample = data["samples"][0] if data["samples"] else None
    if sample:
        probs = [p * 100.0 for p in sample["probabilities"]]
        classes = list(range(10))
        pred = sample["prediction"]
        true_lbl = sample["label"]

        bar_colors = [GREEN_QUANTUM if c == pred else GRID_SLATE for c in classes]
        if pred != true_lbl:
            bar_colors[pred] = RED_ALERT
            bar_colors[true_lbl] = GOLD_HARDWARE

        bars4 = ax4.bar(classes, probs, color=bar_colors, width=0.65)
        for c, p in enumerate(probs):
            if p > 1.0:
                ax4.text(c, p + 1.5, f"{p:.1f}%", ha="center", color=TEXT_POLAR, fontsize=8, weight="bold")

        ax4.set_xticks(classes)
        ax4.set_xlabel("Output Digit Class (0..9)", color=TEXT_MUTED)
        ax4.set_ylabel("Softmax Probability (%)", color=TEXT_MUTED)
        ax4.set_ylim(0, 110)

        status_text = f"GUESS: [{pred}] | TRUE: [{true_lbl}] | CONFIDENCE: {sample['confidence']*100:.1f}%"
        style_card(ax4, f"PANEL 4: INFERENCE GUESS & PROBABILITY PROFILE", status_text)

    plt.tight_layout()
    png_path = output_dir / "mnist_hybrid_qnn_neural_probe.png"
    save_deen_figure(fig, png_path)
    return png_path

if __name__ == "__main__":
    json_file = Path("web/data/mnist_neural_probe.json")
    if not json_file.is_file():
        json_file = Path("results/data/mnist_neural_probe.json")
    if not json_file.is_file():
        raise FileNotFoundError("mnist_neural_probe.json not found")

    csv_candidates = [
        Path("results/hosts/unknown-cpu/experiments/mnist_hybrid_qnn_training.csv"),
        Path("results/experiments/mnist_hybrid_qnn_training.csv"),
        Path("target/bench-scratch/unknown-cpu/results/experiments/mnist_hybrid_qnn_training.csv"),
    ]
    csv_file = Path("none")
    for p in csv_candidates:
        if p.is_file():
            csv_file = p
            break

    out_dir = Path("results/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = chart_mnist_neural_probe(json_file, csv_file, out_dir)
    print(f"[CHART] PNG Generated : {png_path}")
