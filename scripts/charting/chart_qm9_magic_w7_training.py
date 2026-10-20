# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Universal FT-QNN Training: Magic State Distillation & Transversal [[7, 1, 3]] Steane-Wheel QEC (Monograph M156)
Renders a 4-panel publication visual strictly adhering to the Deen OS Sovereign Visual Specification:
- Root Canvas: Deep Obsidian (#05080d)
- Card Panels: Void Charcoal (#0f172a)
- Quantum Telemetry: Laser Emerald Green (#00e676)
- Distillation & Clocks: Radiant Gold (#ffb300)
- Noise & Thresholds: Crimson Red (#ef4444)
- Text & Readouts: Polar Slate (#f8fafc)
Zero cyan, zero purple, zero emojis.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    json_path = os.path.join("results", "data", "qm9_magic_w7_training_report.json")
    if not os.path.exists(json_path):
        print(f"Error: Telemetry file {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    trials = data["trials"]
    epochs = np.arange(1, data["epochs"] + 1)
    cubic_ratio = data["cubic_suppression_ratio"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#05080d")

    colors = {
        "Clean Ideal FT-QNN": "#00e676",
        "Unmitigated Noisy": "#ef4444",
        "Universal FT-QNN (W7+Magic)": "#ffb300",
    }
    markers = {
        "Clean Ideal FT-QNN": "o",
        "Unmitigated Noisy": "x",
        "Universal FT-QNN (W7+Magic)": "s",
    }

    # Panel 1: Training Loss Convergence
    ax1 = axes[0, 0]
    ax1.set_facecolor("#0f172a")
    for t in trials:
        name = t["name"]
        losses = t["epoch_losses"]
        c = colors.get(name, "#38bdf8")
        m = markers.get(name, "o")
        ax1.plot(epochs, losses, color=c, marker=m, linewidth=2.2, label=f"{name} (Final: {losses[-1]:.3f})")
    ax1.set_xlabel("Training Epoch", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Normalized Loss (MSE)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_title("Universal FT-QNN Convergence across 20 Epochs", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", facecolor="#111827", edgecolor="#334155", fontsize=9, labelcolor="#f8fafc")
    ax1.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax1.tick_params(colors="#e2e8f0")

    # Panel 2: Bravyi-Kitaev Cubic Error Suppression
    ax2 = axes[0, 1]
    ax2.set_facecolor("#0f172a")
    eps_in = np.logspace(-3, -1, 100)
    eps_out_linear = eps_in
    eps_out_cubic = np.minimum(eps_in, 35.0 * (eps_in ** 3))

    ax2.plot(eps_in * 100, eps_out_linear * 100, color="#ef4444", linestyle="--", linewidth=2.0, label="Unencoded Physical Error")
    ax2.plot(eps_in * 100, eps_out_cubic * 100, color="#ffb300", linewidth=2.5, label=r"Bravyi-Kitaev 15-to-1 ($\epsilon_{\mathrm{out}} \approx 35\epsilon_{\mathrm{in}}^3$)")
    ax2.axvline(1.5, color="#00e676", linestyle=":", linewidth=1.8, label=f"QM9 Operating Point (p=1.5%, {cubic_ratio:.1f}x Gain)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Physical Input Error Rate (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Output Logical Error Rate (%)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_title("Bravyi-Kitaev Magic State Distillation Factory", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(loc="upper left", facecolor="#111827", edgecolor="#334155", fontsize=9, labelcolor="#f8fafc")
    ax2.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax2.tick_params(colors="#e2e8f0")

    # Panel 3: In-Register QEC & Magic State Injections
    ax3 = axes[1, 0]
    ax3.set_facecolor("#0f172a")
    names3 = [t["name"].replace("Universal FT-QNN (W7+Magic)", "Universal FT-QNN\n(W7+Magic)") for t in trials]
    injections = [t["total_magic_injections"] for t in trials]
    accepted = [t["accepted_distillations"] for t in trials]
    recoveries = [t["total_errors_corrected"] for t in trials]

    x = np.arange(len(names3))
    w = 0.25
    ax3.bar(x - w, injections, width=w, label="Magic Injections", color="#334155", edgecolor="#475569")
    ax3.bar(x, accepted, width=w, label="Accepted Distillations", color="#ffb300", edgecolor="#b45309")
    ax3.bar(x + w, recoveries, width=w, label="AVX-VNNI Recoveries", color="#00e676", edgecolor="#047857")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names3, color="#e2e8f0", fontsize=9)
    ax3.set_ylabel("Event Count", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_title("In-Register Active Ancilla QEC & Distillation Statistics", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax3.legend(loc="upper right", facecolor="#111827", edgecolor="#334155", fontsize=9, labelcolor="#f8fafc")
    ax3.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax3.tick_params(colors="#e2e8f0")

    # Panel 4: Final Prediction Accuracy (MAE in eV)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#0f172a")
    maes = [t["final_mae_ev"] for t in trials]
    bars4 = ax4.bar(names3, maes, color=["#00e676", "#ef4444", "#ffb300"], width=0.45, edgecolor="#1e293b", linewidth=1.5)
    ax4.set_ylabel("Final MAE (eV)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax4.set_title("QM9 HOMO-LUMO Gap Prediction Accuracy", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax4.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax4.tick_params(colors="#e2e8f0")

    for bar, val in zip(bars4, maes):
        ax4.annotate(f"{val:.3f} eV",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", color="#f8fafc", fontweight="bold", fontsize=10)

    plt.suptitle("T-1000: Universal FT-QNN Training with Magic State Distillation & W7 QEC (Monograph M156)\nEastin-Knill Bypass via Bravyi-Kitaev 15-to-1 Teleportation in 18-Qubit L3 Cache",
                 color="#f8fafc", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_plot_dir = os.path.join("results", "plots")
    os.makedirs(out_plot_dir, exist_ok=True)
    out_plot_path = os.path.join(out_plot_dir, "qm9_magic_w7_training_dynamics.png")
    plt.savefig(out_plot_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Publication chart generated: {out_plot_path}")

if __name__ == "__main__":
    main()
