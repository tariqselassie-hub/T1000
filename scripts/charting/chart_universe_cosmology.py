#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Quantum Universe & Cosmological Spacetime Simulation Visualizer.
Plots 4 panels strictly adhering to the Deen OS Sovereign Visual Specification:
- Root Canvas: Deep Obsidian (#05080d)
- Card Panels: Void Charcoal (#0f172a)
- Quantum Signals: Laser Emerald (#00e676)
- Hardware / Gold: Radiant Amber (#ffb300)
- Alert / Horizon: Crimson Red (#ef4444)
- Gridlines: Muted Slate Grid (#1e293b)
- Typography: Polar Slate (#f8fafc)
Zero emojis. Zero prohibited colors (no cyan, purple, magenta).
"""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOR_BG_ROOT = "#05080d"
COLOR_BG_CARD = "#0f172a"
COLOR_EMERALD = "#00e676"
COLOR_AMBER = "#ffb300"
COLOR_CRIMSON = "#ef4444"
COLOR_GRID = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_MUTED = "#94a3b8"

def load_telemetry(json_path: Path) -> dict:
    if not json_path.exists():
        scratch_path = Path("target/bench-scratch/unknown-cpu") / json_path
        if scratch_path.exists():
            json_path = scratch_path
        else:
            raise FileNotFoundError(f"Telemetry file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def render_cosmology_charts(data: dict, output_png: Path):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=COLOR_BG_ROOT)
    fig.subplots_adjust(hspace=0.28, wspace=0.22)

    epochs = data.get("epochs", [])
    times = [e["cosmic_time"] for e in epochs]
    scale_factors = [e["scale_factor"] for e in epochs]
    hubbles = [e["hubble_parameter"] for e in epochs]
    particles = [e["total_particle_density"] for e in epochs]
    entropies = [e["horizon_entanglement_entropy"] for e in epochs]

    final_spectrum = data.get("final_spectrum", {})
    mode_reports = final_spectrum.get("mode_reports", [])
    momenta = [m["momentum_k"] for m in mode_reports]
    n_ks = [m["particle_number"] for m in mode_reports]
    power_specs = [m["power_spectrum"] for m in mode_reports]

    # -------------------------------------------------------------
    # Panel 1: Cosmic Expansion & Comoving Hubble Horizon
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor(COLOR_BG_CARD)

    ax1.plot(times, scale_factors, color=COLOR_AMBER, linewidth=2.4, label="Scale Factor a(t)")
    comoving_horizons = [1.0 / max(1e-4, a * h) for a, h in zip(scale_factors, hubbles)]
    ax1.plot(times, comoving_horizons, color=COLOR_EMERALD, linewidth=2.0, linestyle="--", label="Comoving Horizon (aH)^(-1)")

    ax1.set_title("Cosmic Expansion & Comoving Hubble Horizon", fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)
    ax1.set_xlabel("Cosmic Time t", fontsize=10, color=COLOR_TEXT)
    ax1.set_ylabel("Scale Factor / Horizon Radius", fontsize=10, color=COLOR_TEXT)
    ax1.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax1.tick_params(colors=COLOR_MUTED)
    ax1.legend(loc="upper left", facecolor=COLOR_BG_ROOT, edgecolor=COLOR_GRID, fontsize=9)

    # -------------------------------------------------------------
    # Panel 2: Bogoliubov Created Particle Density Spectrum
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor(COLOR_BG_CARD)

    bars = ax2.bar(
        [f"k={k:.2f}" for k in momenta[:12]],
        n_ks[:12],
        color=COLOR_EMERALD,
        edgecolor=COLOR_GRID,
        width=0.55,
        label="Created Particles n_k = |beta_k|^2"
    )

    ax2.set_title("Bogoliubov Vacuum Particle Creation Spectrum", fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)
    ax2.set_xlabel("Comoving Momentum Mode (k)", fontsize=10, color=COLOR_TEXT)
    ax2.set_ylabel("Created Quanta Number Density", fontsize=10, color=COLOR_TEXT)
    ax2.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID, axis="y")
    ax2.tick_params(colors=COLOR_MUTED)
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(loc="upper right", facecolor=COLOR_BG_ROOT, edgecolor=COLOR_GRID, fontsize=9)

    # -------------------------------------------------------------
    # Panel 3: Primordial Perturbation Power Spectrum P(k)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor(COLOR_BG_CARD)

    ax3.plot(momenta, power_specs, marker="o", color=COLOR_EMERALD, linewidth=2.2, label="Calculated P(k)")
    h_final = final_spectrum.get("hubble_parameter", 1.2)
    hz_asymptote = (h_final / (2.0 * np.pi)) ** 2
    ax3.axhline(hz_asymptote, color=COLOR_AMBER, linestyle=":", linewidth=2.0, label=f"Harrison-Zeldovich Flat (H/2pi)^2: {hz_asymptote:.4f}")

    ax3.set_title("Primordial Perturbation Power Spectrum (CMB Seed)", fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)
    ax3.set_xlabel("Comoving Momentum Mode k", fontsize=10, color=COLOR_TEXT)
    ax3.set_ylabel("Power Spectrum P(k)", fontsize=10, color=COLOR_TEXT)
    ax3.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax3.tick_params(colors=COLOR_MUTED)
    ax3.legend(loc="upper right", facecolor=COLOR_BG_ROOT, edgecolor=COLOR_GRID, fontsize=9)

    # -------------------------------------------------------------
    # Panel 4: Cosmological Horizon Entanglement & Area Law
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor(COLOR_BG_CARD)

    ax4.plot(times, entropies, color=COLOR_EMERALD, linewidth=2.4, label="Entanglement Entropy S_vN(t)")
    areas = [a ** 2 for a in scale_factors]
    max_s = max(entropies) if entropies else 1.0
    max_a2 = max(areas) if areas else 1.0
    norm_areas = [ar * (max_s / max_a2) for ar in areas]
    ax4.plot(times, norm_areas, color=COLOR_AMBER, linewidth=1.8, linestyle="--", label="Normalized Horizon Area A_H(t)")

    ax4.set_title("Cosmological Horizon Entanglement vs. Area Law", fontsize=13, fontweight="bold", color=COLOR_TEXT, pad=12)
    ax4.set_xlabel("Cosmic Time t", fontsize=10, color=COLOR_TEXT)
    ax4.set_ylabel("Von Neumann Entanglement Entropy", fontsize=10, color=COLOR_TEXT)
    ax4.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax4.tick_params(colors=COLOR_MUTED)
    ax4.legend(loc="upper left", facecolor=COLOR_BG_ROOT, edgecolor=COLOR_GRID, fontsize=9)

    fig.suptitle("T-1000 Sovereign Telemetry: Quantum Universe Simulation & Spacetime Emergence",
                 fontsize=15, fontweight="bold", color=COLOR_TEXT, y=0.98)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300, bbox_inches="tight", facecolor=COLOR_BG_ROOT)
    plt.close()
    print(f"Cosmology chart successfully saved to: {output_png}")

if __name__ == "__main__":
    json_path = Path("results/experiments/universe_cosmology.json")
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    out_png = Path("results/charts/universe_cosmology_telemetry.png")
    if len(sys.argv) > 2:
        out_png = Path(sys.argv[2])

    data = load_telemetry(json_path)
    render_cosmology_charts(data, out_png)
