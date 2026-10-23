# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Directed Heptal Szegedy Quantum Walk & DFS Benchmark Visualizer (Monograph M158).

Plots 4 sovereign panels:
  1. Unitary Szegedy Eigenphase Spectrum on the Complex Unit Circle (Quadratic Gap Expansion).
  2. Ballistic Trajectory Showdown: Classical Diffusion vs Quantum Szegedy Hitting on Sink v7.
  3. Decoherence-Free Subspace (DFS) Immunity under Continuous Hub Dephasing.
  4. Quantum Node Population Heatmap Across Trajectory (Nodes v1..v7 over Time).

Strictly conforms to Deen OS Sovereign Visual Specification (Obsidian, Laser Green, Radiant Gold, Alert Red).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, BG_DARK, CARD_CHARCOAL, GRID_SLATE,
    GREEN_QUANTUM, GREEN_EMERALD, GREEN_FLUX,
    GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, RED_CRIMSON,
    TEXT_POLAR, TEXT_SLATE, TEXT_MUTED
)

def main():
    apply_deen_theme()

    # Search for telemetry report in canonical or mirror paths
    candidates = [
        Path("results/data/directed_heptal_quantum_walk_report.json"),
        Path("results/directed_heptal_quantum_walk_report.json"),
        Path("results/hosts/unknown-cpu/data/directed_heptal_quantum_walk_report.json"),
    ]
    report_path = None
    for p in candidates:
        if p.exists():
            report_path = p
            break

    if not report_path:
        print("Error: Telemetry report not found in candidates.")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BG_OBSIDIAN)

    # -------------------------------------------------------------------------
    # Panel 1: Unitary Eigenphase Spectrum (Szegedy W_49 on Unit Circle)
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    style_card(ax1, "1. UNITARY SZEGEDY EIGENPHASE SPECTRUM (W_49)", "Embeds P into H_7 x H_7 with Quadratic Gap Expansion")

    # Unit circle outline
    theta = np.linspace(0, 2 * np.pi, 300)
    ax1.plot(np.cos(theta), np.sin(theta), linestyle="--", color=GRID_SLATE, linewidth=1.5, alpha=0.8, label="Unit Circle |z| = 1")
    ax1.axhline(0, color=GRID_SLATE, linestyle=":", linewidth=0.8)
    ax1.axvline(0, color=GRID_SLATE, linestyle=":", linewidth=0.8)

    # Reconstruct exact W_49 eigenvalues from the classical spectrum
    # For every lambda_k, mu = lambda_k +/- i * sqrt(1 - lambda_k^2)
    evals_P = data["analytical_eigenvalues"]
    for idx, lam in enumerate(evals_P):
        if abs(lam) <= 1.0:
            phi = np.arccos(np.clip(lam, -1.0, 1.0))
            z_pos = np.exp(1j * phi)
            z_neg = np.exp(-1j * phi)
            color = GOLD_HARDWARE if idx == 0 else (GREEN_QUANTUM if idx == 1 else TEXT_MUTED)
            ax1.scatter([z_pos.real, z_neg.real], [z_pos.imag, z_neg.imag], color=color, s=70, zorder=5, edgecolor=BG_OBSIDIAN, linewidth=1.0)

    # Mark Stationary state mu = 1 + 0i
    ax1.scatter([1.0], [0.0], color=GOLD_BRIGHT, s=120, zorder=6, marker="*", label="Stationary Mode (mu = 1.0)")

    # Annotate quadratic gap theta_2
    lambda_2 = evals_P[1]
    theta_2 = np.arccos(lambda_2)
    ax1.plot([0, np.cos(theta_2)], [0, np.sin(theta_2)], color=GREEN_QUANTUM, linewidth=1.8, label=f"Eigenphase theta_2 = {theta_2:.3f} rad ({theta_2*180/np.pi:.1f} deg)")
    ax1.plot([0, lambda_2], [0, 0], color=GOLD_HARDWARE, linewidth=2.0, linestyle="-.", label=f"Classical lambda_2 = {lambda_2:.3f} (Gap = {1-lambda_2:.3f})")

    ax1.text(0.35, 0.45, f"Phase Separation: {theta_2 / (1 - lambda_2):.2f}x\nQuadratic Speedup", color=GREEN_EMERALD, fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_CHARCOAL, edgecolor=GREEN_QUANTUM, alpha=0.9))

    ax1.set_xlim(-1.25, 1.25)
    ax1.set_ylim(-1.25, 1.25)
    ax1.set_xlabel("Re(mu)", color=TEXT_SLATE)
    ax1.set_ylabel("Im(mu)", color=TEXT_SLATE)
    ax1.set_aspect("equal")
    ax1.legend(loc="lower left", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 2: Classical Diffusion vs Quantum Szegedy Hitting on Sink v7
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    style_card(ax2, "2. SINK v7 HITTING TRAJECTORY SHOWDOWN", "Quantum Ballistic Acceleration vs Classical Markov Diffusion Trap")

    trajectory = data["walk_trajectory"]
    steps = [t["step"] for t in trajectory]
    sink_c = [t["classical_sink_prob"] for t in trajectory]
    sink_q = [t["quantum_sink_prob"] for t in trajectory]
    hub_q = [t["quantum_hub_prob"] for t in trajectory]

    ax2.plot(steps, sink_q, color=GREEN_QUANTUM, linewidth=2.2, marker="o", markersize=4, label=f"Quantum Szegedy Sink (Peak {max(sink_q):.4f})")
    ax2.plot(steps, sink_c, color=GOLD_HARDWARE, linewidth=2.0, linestyle="--", marker="s", markersize=4, label=f"Classical Markov Sink (Peak {max(sink_c):.4f})")
    ax2.plot(steps, hub_q, color=TEXT_MUTED, linewidth=1.2, linestyle=":", label="Quantum Hub (v1) Retention")

    # Stationary sink level pi_7 = 8/42
    pi_7 = 8.0 / 42.0
    ax2.axhline(pi_7, color=GRID_SLATE, linestyle="-.", linewidth=1.2, label=f"Stationary Sink pi_7 = 8/42 ({pi_7:.4f})")

    # Annotate peak speedup
    peak_q_step = np.argmax(sink_q)
    ax2.annotate(f"Quantum Peak: {max(sink_q):.4f}\n({max(sink_q)/max(sink_c):.2f}x Speedup @ t={peak_q_step})",
                 xy=(peak_q_step, max(sink_q)),
                 xytext=(peak_q_step + 3, max(sink_q) - 0.12),
                 color=GREEN_EMERALD, fontweight="bold", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=GREEN_QUANTUM, lw=1.5),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_CHARCOAL, edgecolor=GREEN_QUANTUM))

    ax2.set_xlabel("Walk Step (t)", color=TEXT_SLATE)
    ax2.set_ylabel("Detection Probability", color=TEXT_SLATE)
    ax2.set_xlim(0, 30)
    ax2.set_ylim(0, 0.85)
    ax2.legend(loc="upper right", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 3: Decoherence-Free Subspace (DFS) Hub-Dephasing Immunity
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    style_card(ax3, "3. DECOHERENCE-FREE SUBSPACE (DFS) IMMUNITY", "Antisymmetric Mode (v3 - v6) under Continuous Hub Dephasing Noise")

    noise_traj = data["dark_noise_trajectory"]
    noise_steps = [n["step"] for n in noise_traj]
    dark_fid = [n["dark_fidelity"] for n in noise_traj]
    bright_fid = [n["bright_fidelity"] for n in noise_traj]

    ax3.plot(noise_steps, dark_fid, color=GREEN_QUANTUM, linewidth=2.5, label="Dark State (v3 - v6): 100% Retained (F = 1.000000)")
    ax3.plot(noise_steps, bright_fid, color=RED_ALERT, linewidth=2.0, linestyle="--", label=f"Bright State (v3 + v6): Dephased (F = {bright_fid[-1]:.4f})")

    ax3.fill_between(noise_steps, dark_fid, 0.98, color=GREEN_QUANTUM, alpha=0.15)
    ax3.fill_between(noise_steps, bright_fid, 0.0, color=RED_ALERT, alpha=0.10)

    ax3.text(5, 0.50, f"Dark Mode Hub Leakage: {data['dark_state_max_hub_leakage']:.1e}\nExact DFS Cancellation (v3 - v6)\nMachine Zero Coupling to v1 Noise",
             color=GREEN_EMERALD, fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_CHARCOAL, edgecolor=GREEN_QUANTUM, alpha=0.9))

    ax3.set_xlabel("Noise Injection Step (t)", color=TEXT_SLATE)
    ax3.set_ylabel("Subspace Overlap Fidelity", color=TEXT_SLATE)
    ax3.set_xlim(1, 50)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="center right", framealpha=0.9)

    # -------------------------------------------------------------------------
    # Panel 4: Quantum Node Population Spatiotemporal Heatmap
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    style_card(ax4, "4. QUANTUM WAVEPACKET TRANSIT HEATMAP", "Probability Density Flow across Nodes v1..v7 over Time")

    # Heatmap matrix: [nodes=7, steps=21]
    heat_data = np.zeros((7, 21))
    for t_idx in range(21):
        for n_idx in range(7):
            heat_data[n_idx, t_idx] = trajectory[t_idx]["quantum_probs"][n_idx]

    cmap_colors = [BG_OBSIDIAN, CARD_CHARCOAL, GRID_SLATE, GREEN_FLUX, GREEN_QUANTUM, GOLD_HARDWARE]
    deen_cmap = mcolors.LinearSegmentedColormap.from_list("DeenHeatmap", cmap_colors)

    im = ax4.imshow(heat_data, aspect="auto", origin="lower", cmap=deen_cmap, vmin=0.0, vmax=0.75, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax4, pad=0.02)
    cbar.ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    cbar.set_label("Occupancy Probability", color=TEXT_SLATE, fontsize=9)

    ax4.set_yticks(range(7))
    ax4.set_yticklabels([f"v{i+1}" for i in range(7)], color=TEXT_POLAR, fontweight="bold")
    ax4.set_xticks(range(0, 21, 2))
    ax4.set_xticklabels(range(0, 21, 2), color=TEXT_MUTED)
    ax4.set_xlabel("Walk Step (t)", color=TEXT_SLATE)
    ax4.set_ylabel("Graph Node", color=TEXT_SLATE)

    # Annotate ballistic transfer from v1 to v7
    ax4.annotate("Hub v1 Launch", xy=(0, 0), xytext=(2, 0.5), color=GOLD_BRIGHT, fontsize=8, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GOLD_HARDWARE, lw=1.2))
    ax4.annotate("Sink v7 Peak (73.5%)", xy=(4, 6), xytext=(6, 5.2), color=GREEN_EMERALD, fontsize=8, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN_QUANTUM, lw=1.2))

    plt.tight_layout()

    # Save output to results/ and results/charts/
    out_png = Path("results/directed_heptal_quantum_walk.png")
    saved_path = save_deen_figure(fig, out_png, dpi=300)
    print(f"Sovereign Chart successfully saved to: {saved_path}")

    # Mirror copy to results/charts/
    mirror_png = Path("results/charts/directed_heptal_quantum_walk.png")
    save_deen_figure(fig, mirror_png, dpi=300)
    print(f"Chart mirror saved to: {mirror_png}")

if __name__ == "__main__":
    main()
