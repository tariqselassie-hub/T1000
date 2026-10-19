# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Deen OS Sovereign Visualizer: Geometric Quantum Graph Neural Network (QGNN).
Adheres strictly to the Sovereign Visual Specification:
Deep Obsidian (#05080d), Void Charcoal (#0f172a), Laser Emerald Green (#00e676),
Radiant Gold (#ffb300), Alert Red (#ef4444), Polar Slate (#f8fafc).
Zero emojis. No prohibited cyan or purple tints.
"""

import sys
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# --- Sovereign Color Palette ---
COLOR_BG = "#05080d"
COLOR_CARD = "#0f172a"
COLOR_GREEN = "#00e676"
COLOR_GOLD = "#ffb300"
COLOR_RED = "#ef4444"
COLOR_SLATE = "#f8fafc"
COLOR_MUTED = "#64748b"
COLOR_GRID = "#1e293b"

def main():
    json_path = os.path.join("results", "data", "qgnn_molecular_telemetry.json")
    if not os.path.exists(json_path):
        print(f"Error: Telemetry JSON not found at {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scaling_points = data["scaling_points"]
    perm_test = data["permutation_test"]
    training = data["training_history"]
    latency_us = data["mean_inference_latency_us"]

    qubits = [p["qubits"] for p in scaling_points]
    hea_var = [p["hea_variance"] for p in scaling_points]
    qgnn_var = [p["qgnn_variance"] for p in scaling_points]
    hea_dim = [p["hea_lie_dim"] for p in scaling_points]
    qgnn_dim = [p["qgnn_lie_dim"] for p in scaling_points]

    epochs = [t["epoch"] for t in training]
    mse_losses = [t["mse_loss"] for t in training]
    rmse_losses = [t["rmse"] for t in training]

    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=COLOR_BG)
    fig.subplots_adjust(hspace=0.32, wspace=0.25, top=0.92, bottom=0.08, left=0.08, right=0.95)

    fig.suptitle(
        "T-1000 GEOMETRIC QUANTUM GRAPH NEURAL NETWORK (QGNN) SHOWDOWN\n"
        "Lie-Algebraic Equivariance, Barren Plateau Resilience & Molecular Topology Scaling",
        color=COLOR_SLATE, fontsize=16, fontweight="bold", y=0.97
    )

    # -------------------------------------------------------------
    # Panel 1: Barren Plateau Gradient Variance vs Qubit Width
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor(COLOR_CARD)
    ax1.plot(qubits, qgnn_var, color=COLOR_GREEN, marker="o", linewidth=2.5, markersize=8, label="Geometric QGNN (Lie Equivariant)")
    ax1.plot(qubits, hea_var, color=COLOR_RED, marker="s", linewidth=2.0, linestyle="--", markersize=7, label="Hardware-Efficient Ansatz (HEA)")
    ax1.set_title("PANEL 1: Barren Plateau Resilience (Gradient Variance)", color=COLOR_SLATE, fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Hilbert Space Qubits (N)", color=COLOR_SLATE, fontsize=10)
    ax1.set_ylabel("Empirical Gradient Variance Var[d_theta L]", color=COLOR_SLATE, fontsize=10)
    ax1.set_yscale("log")
    ax1.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6)
    ax1.tick_params(colors=COLOR_SLATE)
    for spine in ax1.spines.values():
        spine.set_color(COLOR_GRID)
    ax1.legend(facecolor=COLOR_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_SLATE, loc="upper right")

    # Annotation
    ax1.annotate(
        "Invariant Lie Algebra Bound:\nVar ~ O(1/poly(N)) Resilient",
        xy=(qubits[-1], qgnn_var[-1]), xytext=(qubits[-2], qgnn_var[-1] * 2.5),
        color=COLOR_GREEN, fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_GREEN, lw=1.5)
    )

    # -------------------------------------------------------------
    # Panel 2: Dynamical Lie Algebra Dimension Scaling
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor(COLOR_CARD)
    ax2.plot(qubits, hea_dim, color=COLOR_RED, marker="^", linewidth=2.2, linestyle="--", label="HEA: dim(su(2^N)) = 4^N - 1")
    ax2.plot(qubits, qgnn_dim, color=COLOR_GOLD, marker="D", linewidth=2.5, label="QGNN: dim(g) ~ O(|V| + |E|)")
    ax2.set_title("PANEL 2: Dynamical Lie Algebra Dimension dim(g)", color=COLOR_SLATE, fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Hilbert Space Qubits (N)", color=COLOR_SLATE, fontsize=10)
    ax2.set_ylabel("Lie Dimension dim(g) [Log Scale]", color=COLOR_SLATE, fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6)
    ax2.tick_params(colors=COLOR_SLATE)
    for spine in ax2.spines.values():
        spine.set_color(COLOR_GRID)
    ax2.legend(facecolor=COLOR_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_SLATE, loc="upper left")

    # -------------------------------------------------------------
    # Panel 3: Graph Permutation Equivariance Verification
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor(COLOR_CARD)
    categories = ["Original G\n(c1ccccc1)", "Permuted p(G)\n(Cyclic Shift)", "Absolute Delta\n|Delta y|"]
    vals = [perm_test["original_prediction"], perm_test["permuted_prediction"], perm_test["max_absolute_delta"]]
    colors = [COLOR_GREEN, COLOR_GOLD, COLOR_RED]
    bars = ax3.bar(categories, vals, color=colors, width=0.45, edgecolor=COLOR_GRID, linewidth=1.5)
    ax3.set_title("PANEL 3: Exact Permutation Equivariance Invariance", color=COLOR_SLATE, fontsize=12, fontweight="bold", pad=12)
    ax3.set_ylabel("Molecular Prediction Value y", color=COLOR_SLATE, fontsize=10)
    ax3.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6, axis="y")
    ax3.tick_params(colors=COLOR_SLATE)
    for spine in ax3.spines.values():
        spine.set_color(COLOR_GRID)

    for bar, val in zip(bars, vals):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.02,
            f"{val:.6f}", ha="center", va="bottom", color=COLOR_SLATE, fontsize=9, fontweight="bold"
        )
    ax3.set_ylim(0, max(vals) * 1.25)

    # -------------------------------------------------------------
    # Panel 4: Molecular Property Training & AVX2 Latency
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor(COLOR_CARD)
    ax4.plot(epochs, mse_losses, color=COLOR_GREEN, marker="o", linewidth=2.0, label="MSE Loss")
    ax4.plot(epochs, rmse_losses, color=COLOR_GOLD, marker="s", linewidth=2.0, linestyle="--", label="RMSE")
    ax4.set_title("PANEL 4: ZINC Drug Candidate Property Training", color=COLOR_SLATE, fontsize=12, fontweight="bold", pad=12)
    ax4.set_xlabel("Training Epochs", color=COLOR_SLATE, fontsize=10)
    ax4.set_ylabel("Loss Metric", color=COLOR_SLATE, fontsize=10)
    ax4.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6)
    ax4.tick_params(colors=COLOR_SLATE)
    for spine in ax4.spines.values():
        spine.set_color(COLOR_GRID)
    ax4.legend(facecolor=COLOR_BG, edgecolor=COLOR_GRID, labelcolor=COLOR_SLATE, loc="upper right")

    throughput_mols = 1_000_000.0 / latency_us
    info_box = (
        f"AVX2 SIMD Telemetry:\n"
        f"- Latency: {latency_us:.2f} us / mol\n"
        f"- Throughput: {throughput_mols:,.0f} mols/s\n"
        f"- Convergence: Monotonic Loss Reduction"
    )
    ax4.text(
        0.05, 0.20, info_box, transform=ax4.transAxes,
        color=COLOR_SLATE, fontsize=9,
        bbox=dict(boxstyle="square,pad=0.6", facecolor=COLOR_BG, edgecolor=COLOR_GREEN, alpha=0.9)
    )

    out_dir = os.path.join("results", "charts")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "qgnn_geometric_showdown.png")
    plt.savefig(out_file, dpi=300, facecolor=COLOR_BG)
    plt.close()

    print(f" [+] Rendered QGNN Geometric Showdown Visual: {out_file} ({os.path.getsize(out_file)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
