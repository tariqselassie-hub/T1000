# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
RCS Physical Noise Chaos & Coherence Transition Visualization.
Plots:
  1. Depth vs Cross-Linear XEB Fidelity (Ideal convergence vs Unprotected vs Stabilized).
  2. Quantum Overlap Fidelity Cliff across depth.
  3. Depolarizing Noise Probability vs XEB Retention and KS Goodness-of-Fit boundary.
Adheres strictly to Deen OS visual specification (Deep Obsidian, Laser Green, Radiant Gold, Alert Red).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from deen_theme import (
    apply_deen_theme, save_deen_figure,
    BG_OBSIDIAN, BG_DARK, CARD_CHARCOAL, GRID_SLATE,
    GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_SLATE, TEXT_MUTED
)

def main():
    apply_deen_theme()

    json_path = Path("results/rcs_physical_noise_chaos.json")
    if not json_path.exists():
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    depth_data = data["depth_sweep"]
    noise_data = data["noise_sweep"]

    # Filter depth points for 14Q and 16Q
    d14 = [p for p in depth_data if p["qubits"] == 14]
    d16 = [p for p in depth_data if p["qubits"] == 16]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG_OBSIDIAN)

    # -------------------------------------------------------------------------
    # Panel 1: Depth vs Cross-Linear XEB (Convergence to PT Haar Asymptote)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.set_facecolor(BG_DARK)
    depths_14 = [p["depth"] for p in d14]
    ideal_xeb_14 = [p["ideal_xeb"] for p in d14]
    unprot_xeb_14 = [p["unprot_xeb"] for p in d14]
    stab_xeb_14 = [p["stab_xeb"] for p in d14]

    ax1.axhline(1.0, color=GOLD_HARDWARE, linestyle="--", linewidth=1.2, label="Haar Porter-Thomas Limit (1.0)")
    ax1.plot(depths_14, ideal_xeb_14, marker="o", color=TEXT_POLAR, linewidth=2.0, label="Ideal Unitary (14Q)")
    ax1.plot(depths_14, unprot_xeb_14, marker="s", color=RED_ALERT, linewidth=2.0, label="Unprotected Noise (T1/T2*)")
    ax1.plot(depths_14, stab_xeb_14, marker="^", color=GREEN_QUANTUM, linewidth=2.0, label="Stabilized (XY4 + Manifold)")

    ax1.set_title("Cross-Linear XEB vs Depth (14Q)", fontsize=13, color=TEXT_POLAR, pad=12)
    ax1.set_xlabel("Circuit Depth (Layers)", fontsize=11, color=TEXT_SLATE)
    ax1.set_ylabel("Linear XEB Fidelity (F_XEB)", fontsize=11, color=TEXT_SLATE)
    ax1.set_yscale("log")
    ax1.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)
    ax1.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 2: Quantum Overlap Fidelity Cliff (14Q & 16Q)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor(BG_DARK)

    unprot_fid_14 = [p["unprot_fidelity"] for p in d14]
    stab_fid_14 = [p["stab_fidelity"] for p in d14]
    unprot_fid_16 = [p["unprot_fidelity"] for p in d16]
    stab_fid_16 = [p["stab_fidelity"] for p in d16]

    ax2.plot(depths_14, unprot_fid_14, marker="s", linestyle=":", color=RED_ALERT, linewidth=1.8, label="14Q Unprotected")
    ax2.plot(depths_14, stab_fid_14, marker="o", color=GREEN_QUANTUM, linewidth=2.0, label="14Q Stabilized (+5,100%)")
    ax2.plot(depths_14, unprot_fid_16, marker="x", linestyle=":", color="#f87171", linewidth=1.8, label="16Q Unprotected")
    ax2.plot(depths_14, stab_fid_16, marker="D", color=GREEN_EMERALD, linewidth=2.0, label="16Q Stabilized (+13,500%)")

    ax2.set_title("Quantum State Overlap Fidelity Cliff", fontsize=13, color=TEXT_POLAR, pad=12)
    ax2.set_xlabel("Circuit Depth (Layers)", fontsize=11, color=TEXT_SLATE)
    ax2.set_ylabel("Fidelity vs Ideal |<psi_ideal|psi>|^2", fontsize=11, color=TEXT_SLATE)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)
    ax2.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=9)

    # -------------------------------------------------------------------------
    # Panel 3: Depolarizing Noise Probability vs XEB & KS Conformance
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.set_facecolor(BG_DARK)

    depol_p = [p["depol_prob"] for p in noise_data]
    unprot_noise_xeb = [p["unprot_cross_xeb"] for p in noise_data]
    stab_noise_xeb = [p["stab_cross_xeb"] for p in noise_data]
    ks_passed = [p["unprot_ks_passed"] for p in noise_data]

    ax3.plot(depol_p, unprot_noise_xeb, marker="s", color=RED_ALERT, linewidth=2.0, label="Unprotected Cross-XEB")
    ax3.plot(depol_p, stab_noise_xeb, marker="^", color=GREEN_QUANTUM, linewidth=2.0, label="Stabilized Cross-XEB")

    # Mark the KS threshold critical boundary
    ax3.axvline(0.010, color=GOLD_HARDWARE, linestyle="--", linewidth=1.5, label="KS Test Failure Horizon (p=0.01)")

    ax3.set_title("Physical Noise Sensitivity & KS Threshold (14Q, D=16)", fontsize=13, color=TEXT_POLAR, pad=12)
    ax3.set_xlabel("Depolarizing Error Rate (p)", fontsize=11, color=TEXT_SLATE)
    ax3.set_ylabel("Cross-Linear XEB", fontsize=11, color=TEXT_SLATE)
    ax3.set_xscale("log")
    ax3.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)
    ax3.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=9)

    plt.tight_layout()
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    out_fig = assets_dir / "rcs_physical_noise_chaos.png"
    save_deen_figure(fig, out_fig)
    print(f"Figure rendered to: {out_fig}")

if __name__ == "__main__":
    main()
