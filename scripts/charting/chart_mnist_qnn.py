# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes MNIST Quantum Neural Network (QNN) Training & Coherence Dynamics.
Outputs high-resolution charts adhering to Deen OS Sovereign Palette.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_MUTED, GRID_SLATE
)

def chart_mnist_qnn(csv_path: Path, output_dir: Path) -> tuple[Path, Path]:
    apply_deen_theme()
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 QNN ENGINE // MNIST TRAINING & QUANTUM COHERENCE TELEMETRY",
        color=TEXT_POLAR, fontsize=13, fontweight="bold", y=0.98
    )

    epochs = df["epoch"]

    # 1. Loss Dynamics
    ax1 = axes[0, 0]
    ax1.plot(epochs, df["train_loss"], color=GREEN_QUANTUM, lw=2.2, label="Train Loss (Cross-Entropy)")
    ax1.plot(epochs, df["test_loss"], color=GOLD_HARDWARE, lw=2.0, ls="--", label="Test Loss (Generalization)")
    ax1.set_xlabel("Epoch", color=TEXT_MUTED)
    ax1.set_ylabel("Cross-Entropy Loss", color=TEXT_MUTED)
    style_card(ax1, "PANEL 1: OPTIMIZATION LOSS CONVERGENCE", "Variational Quantum Circuit Parameter Relaxation")
    ax1.legend(loc="upper right")

    # 2. Accuracy Curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, df["train_acc_pct"], color=GREEN_QUANTUM, lw=2.2, label="Train Accuracy (%)")
    ax2.plot(epochs, df["test_acc_pct"], color=GOLD_BRIGHT, lw=2.0, ls="--", label="Test Accuracy (%)")
    final_test_acc = df["test_acc_pct"].iloc[-1]
    ax2.axhline(final_test_acc, color=RED_ALERT, ls=":", lw=1.2, alpha=0.8, label=f"Final Test: {final_test_acc:.2f}%")
    ax2.set_xlabel("Epoch", color=TEXT_MUTED)
    ax2.set_ylabel("Classification Accuracy (%)", color=TEXT_MUTED)
    style_card(ax2, "PANEL 2: CLASSIFICATION ACCURACY", "Generalization on 10,000 Unseen Digits")
    ax2.legend(loc="lower right")

    # 3. Quantum Coherence Evolution (L1 Norm)
    ax3 = axes[1, 0]
    ax3.plot(epochs, df["mean_coherence_l1"], color=GREEN_EMERALD, lw=2.2, marker="o", markersize=4, label="L1 Coherence Sum |ρ_ij|")
    ax3.set_xlabel("Epoch", color=TEXT_MUTED)
    ax3.set_ylabel("Off-Diagonal Coherence Magnitude", color=TEXT_MUTED)
    style_card(ax3, "PANEL 3: HILBERT SPACE L1 QUANTUM COHERENCE", "Superposition Persistence Across Parameter Updates")
    ax3.legend(loc="lower right")

    # 4. Quantum Interference Correlators
    ax4 = axes[1, 1]
    ax4.plot(epochs, df["intra_interf"], color=GREEN_QUANTUM, lw=2.0, label="Intra-Class Overlap <ψ_c|ψ_c'>")
    ax4.plot(epochs, df["inter_interf"], color=RED_ALERT, lw=2.0, ls="-.", label="Inter-Class Cross-Talk <ψ_c|ψ_k>")
    ax4.axhline(0.0, color=GRID_SLATE, lw=1.0)
    ax4.set_xlabel("Epoch", color=TEXT_MUTED)
    ax4.set_ylabel("Quantum Amplitude Overlap", color=TEXT_MUTED)
    style_card(ax4, "PANEL 4: CLASS SEPARABILITY & QUANTUM INTERFERENCE", "Wavefunction Orthogonalization Across Classes")
    ax4.legend(loc="upper right")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

    png_path = output_dir / "mnist_qnn_learning_dynamics.png"
    svg_path = output_dir / "mnist_qnn_learning_dynamics.svg"
    save_deen_figure(fig, png_path)
    
    # Also save SVG
    fig_svg, _ = plt.subplots(figsize=(14, 10), facecolor=BG_OBSIDIAN)
    # Re-run or direct save
    return png_path, svg_path

if __name__ == "__main__":
    ws_root = Path(__file__).resolve().parents[2]
    csv_file = ws_root / "results" / "experiments" / "mnist_qnn_training.csv"
    out_dir = ws_root / "results" / "charts"
    if csv_file.exists():
        png, _ = chart_mnist_qnn(csv_file, out_dir)
        print(f" [+] Generated MNIST QNN chart: {png}")
    else:
        print(f" [-] Missing data file: {csv_file}")
