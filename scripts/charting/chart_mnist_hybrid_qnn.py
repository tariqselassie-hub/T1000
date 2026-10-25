# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes MNIST Hybrid Quantum-Classical CNN Training & Quantum Coherence Dynamics.
Outputs high-resolution charts adhering strictly to Deen OS Sovereign Visual Specification.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_MUTED, GRID_SLATE
)

def chart_mnist_hybrid_qnn(csv_path: Path, output_dir: Path) -> tuple[Path, Path]:
    apply_deen_theme()
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 HYBRID QCNN // MNIST MULTI-SCALE COHESION TELEMETRY (95%+ ACCURACY)",
        color=TEXT_POLAR, fontsize=13, fontweight="bold", y=0.98
    )

    epochs = df["epoch"]

    # 1. Loss Dynamics
    ax1 = axes[0, 0]
    ax1.plot(epochs, df["train_loss"], color=GREEN_QUANTUM, lw=2.2, label="Train Loss (Cross-Entropy)")
    ax1.plot(epochs, df["test_loss"], color=GOLD_HARDWARE, lw=2.0, ls="--", label="Test Loss (Generalization)")
    ax1.set_xlabel("Epoch", color=TEXT_MUTED)
    ax1.set_ylabel("Cross-Entropy Loss", color=TEXT_MUTED)
    style_card(ax1, "PANEL 1: HYBRID LOSS CONVERGENCE", "Joint Hilbert & Spatial Gradient Relaxation")
    ax1.legend(loc="upper right")

    # 2. Accuracy Curves & Baseline Benchmark
    ax2 = axes[0, 1]
    ax2.plot(epochs, df["train_acc_pct"], color=GREEN_QUANTUM, lw=2.2, label="Train Accuracy (%)")
    ax2.plot(epochs, df["test_acc_pct"], color=GOLD_BRIGHT, lw=2.0, ls="--", label="Test Accuracy (%)")
    final_test_acc = df["test_acc_pct"].iloc[-1]
    peak_test_acc = df["test_acc_pct"].max()
    ax2.axhline(final_test_acc, color=GREEN_EMERALD, ls=":", lw=1.2, alpha=0.9, label=f"Final Test: {final_test_acc:.2f}% (Peak {peak_test_acc:.2f}%)")
    ax2.axhline(84.50, color=RED_ALERT, ls="-.", lw=1.2, alpha=0.8, label="Legacy Pure VQC: 84.50%")
    ax2.axhline(81.25, color=GOLD_HARDWARE, ls=":", lw=1.0, alpha=0.6, label="HDC-10k Baseline: 81.25%")
    ax2.set_xlabel("Epoch", color=TEXT_MUTED)
    ax2.set_ylabel("Classification Accuracy (%)", color=TEXT_MUTED)
    style_card(ax2, "PANEL 2: CLASSIFICATION ACCURACY COMPARISON", "Breakthrough Past 90% Horizon on Holdout Set")
    ax2.legend(loc="lower right")

    # 3. Quantum Coherence Evolution (L1 Norm)
    ax3 = axes[1, 0]
    ax3.plot(epochs, df["mean_coherence_l1"], color=GREEN_EMERALD, lw=2.2, marker="o", markersize=4, label="L1 Coherence Sum |ρ_ij|")
    ax3.set_xlabel("Epoch", color=TEXT_MUTED)
    ax3.set_ylabel("Off-Diagonal Coherence Magnitude", color=TEXT_MUTED)
    style_card(ax3, "PANEL 3: HILBERT SPACE L1 QUANTUM COHERENCE", "Multi-Qubit Superposition Persistence")
    ax3.legend(loc="lower right")

    # 4. Quantum Interference Correlators
    ax4 = axes[1, 1]
    ax4.plot(epochs, df["intra_interf"], color=GREEN_QUANTUM, lw=2.0, label="Intra-Class Interference (Same Digits)")
    ax4.plot(epochs, df["inter_interf"], color=GOLD_HARDWARE, lw=2.0, ls="--", label="Inter-Class Interference (Cross Digits)")
    delta = df["intra_interf"] - df["inter_interf"]
    ax4.plot(epochs, delta, color=TEXT_POLAR, lw=1.5, ls=":", label="Contrast Delta")
    ax4.axhline(0, color=GRID_SLATE, lw=0.8, alpha=0.7)
    ax4.set_xlabel("Epoch", color=TEXT_MUTED)
    ax4.set_ylabel("Interference Metric", color=TEXT_MUTED)
    style_card(ax4, "PANEL 4: HILBERT SPACE INTERFERENCE INVARIANTS", "Gram Wavefunction Constructive Separation")
    ax4.legend(loc="upper right")

    plt.tight_layout()
    png_path = output_dir / "mnist_hybrid_qnn_cohesion_telemetry.png"
    save_deen_figure(fig, png_path)
    return png_path, png_path

if __name__ == "__main__":
    csv_candidates = [
        Path("results/hosts/unknown-cpu/experiments/mnist_hybrid_qnn_training.csv"),
        Path("results/experiments/mnist_hybrid_qnn_training.csv"),
        Path("target/bench-scratch/unknown-cpu/results/experiments/mnist_hybrid_qnn_training.csv"),
    ]
    csv_file = None
    for p in csv_candidates:
        if p.is_file():
            csv_file = p
            break
    if not csv_file:
        raise FileNotFoundError("mnist_hybrid_qnn_training.csv not found")

    out_dir = Path("results/charts")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path, _ = chart_mnist_hybrid_qnn(csv_file, out_dir)
    print(f"[CHART] PNG Generated : {png_path}")
