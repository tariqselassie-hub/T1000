# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes 2D Quantum Spin Lattice Phase Transitions & Critical Entanglement Peaks.
Adheres strictly to the Deen OS Sovereign Palette.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_MUTED, GRID_SLATE
)

def chart_lattice_phase(json_path: Path, output_dir: Path) -> Path:
    apply_deen_theme()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hx_ratio = [frame["h_x"] / max(abs(frame["j_coupling"]), 1e-6) for frame in data]
    energies = [frame["energy"] for frame in data]
    magnetizations = [frame["mean_magnetization"] for frame in data]
    bipartite_entropies = [frame["bipartite_entropy"] for frame in data]
    site_entropies = [frame["site_entropy"][0] for frame in data]

    # Find empirical critical point (peak entanglement)
    max_entropy_idx = int(np.argmax(bipartite_entropies))
    crit_hx = hx_ratio[max_entropy_idx]
    crit_entropy = bipartite_entropies[max_entropy_idx]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 ENGINE // 2D QUANTUM SPIN LATTICE: PHASE TRANSITION & ENTANGLEMENT CRITICALITY",
        color=TEXT_POLAR, fontsize=13, fontweight="bold", y=0.98
    )

    # 1. Ground State Energy <H>
    ax1 = axes[0]
    ax1.plot(hx_ratio, energies, color=GOLD_HARDWARE, lw=2.4, marker="o", markersize=5, label="Ground State Energy <H>")
    ax1.axvline(crit_hx, color=RED_ALERT, ls="--", lw=1.5, label=f"Critical Point (hx/|J| ≈ {crit_hx:.3f})")
    ax1.set_xlabel("Transverse Field Ratio h_x / |J|", color=TEXT_MUTED)
    ax1.set_ylabel("Ground State Energy <H_Ising>", color=TEXT_MUTED)
    style_card(ax1, "PANEL 1: GROUND STATE ENERGY CURVE", "Variational Quantum Relaxation (HVA Ansatz)")
    ax1.legend(loc="lower left")

    # 2. Spontaneous Magnetization Breakdown <M_z>
    ax2 = axes[1]
    ax2.plot(hx_ratio, magnetizations, color=GREEN_QUANTUM, lw=2.4, marker="s", markersize=5, label="Spontaneous Order <M_z>")
    ax2.axvline(crit_hx, color=RED_ALERT, ls="--", lw=1.5, label=f"Empirical Phase Boundary: {crit_hx:.3f}")
    ax2.axvline(3.044, color=GOLD_BRIGHT, ls=":", lw=1.2, label="Thermodynamic 2D Ising (h_c ≈ 3.044)")
    ax2.axhline(0.0, color=GRID_SLATE, lw=1.0)
    ax2.set_xlabel("Transverse Field Ratio h_x / |J|", color=TEXT_MUTED)
    ax2.set_ylabel("Local Magnetization Order <Z_i>", color=TEXT_MUTED)
    style_card(ax2, "PANEL 2: SPONTANEOUS ORDER COLLAPSE", "Ferromagnetic Alignment to Paramagnetic Disorder")
    ax2.legend(loc="upper right")

    # 3. Critical Bipartite Entanglement Entropy Spike
    ax3 = axes[2]
    ax3.plot(hx_ratio, bipartite_entropies, color=GREEN_EMERALD, lw=2.4, marker="D", markersize=6, label="Bipartite Half-Cut Entropy S(A:B)")
    ax3.plot(hx_ratio, site_entropies, color=GOLD_HARDWARE, lw=1.8, ls="--", marker="^", markersize=5, label="Single-Site Entropy S(ρ_0)")
    ax3.axvline(crit_hx, color=RED_ALERT, ls="--", lw=1.5, label=f"Critical Entropy Peak: {crit_entropy:.3f} nats")
    ax3.set_xlabel("Transverse Field Ratio h_x / |J|", color=TEXT_MUTED)
    ax3.set_ylabel("Von Neumann Entropy (nats)", color=TEXT_MUTED)
    style_card(ax3, "PANEL 3: BIPARTITE ENTANGLEMENT SPIKE", "Quantum Criticality Detection via Mutual Information")
    ax3.legend(loc="lower right")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

    png_path = output_dir / "lattice_phase_transition.png"
    save_deen_figure(fig, png_path)
    return png_path

if __name__ == "__main__":
    ws_root = Path(__file__).resolve().parents[2]
    json_file = ws_root / "results" / "experiments" / "lattice_phase_transition.json"
    out_dir = ws_root / "results" / "charts"
    if json_file.exists():
        png = chart_lattice_phase(json_file, out_dir)
        print(f" [+] Generated lattice phase transition chart: {png}")
    else:
        print(f" [-] Missing data file: {json_file}")
