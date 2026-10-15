# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes 2D Magnon Quench Dynamics & Spin-Wave Dispersion Across Crystal Lattice.
Adheres strictly to the Deen OS Sovereign Palette.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, BG_DARK, GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_MUTED, GRID_SLATE
)

def chart_lattice_quench(json_path: Path, output_dir: Path) -> Path:
    apply_deen_theme()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    timesteps = [frame.get("step", 0) for frame in data]
    # Assuming dt = 0.05 from simulation
    times = [step * 0.05 for step in timesteps]
    mean_mag = [frame["mean_magnetization"] for frame in data]
    bipartite_entropy = [frame["bipartite_entropy"] for frame in data]

    # Extract site-specific magnetization profiles
    width = data[0]["width"]
    height = data[0]["height"]
    center_idx = (height // 2) * width + (width // 2)
    corner_idx = 0
    edge_idx = (height // 2) * width

    center_mag = [frame["magnetization"][center_idx] for frame in data]
    edge_mag = [frame["magnetization"][edge_idx] for frame in data]
    corner_mag = [frame["magnetization"][corner_idx] for frame in data]

    fig = plt.figure(figsize=(18, 9), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 ENGINE // 2D MAGNON QUENCH & SPIN-WAVE DISPERSION TELEMETRY",
        color=TEXT_POLAR, fontsize=13, fontweight="bold", y=0.98
    )

    # 1. Left Column: Temporal Dynamics (2 Subplots)
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
    ax1.plot(times, center_mag, color=RED_ALERT, lw=2.4, label="Center Defect Site (Excitations)")
    ax1.plot(times, edge_mag, color=GOLD_HARDWARE, lw=2.0, ls="--", label="Mid-Edge Site (Wavefront Arrival)")
    ax1.plot(times, corner_mag, color=GREEN_QUANTUM, lw=2.0, ls="-.", label="Corner Boundary Site (Reflection)")
    ax1.axhline(0.0, color=GRID_SLATE, lw=1.0)
    ax1.set_xlabel("Physical Time t (s)", color=TEXT_MUTED)
    ax1.set_ylabel("Local Magnetization <Z_i(t)>", color=TEXT_MUTED)
    style_card(ax1, "PANEL 1: MAGNON SPIN-WAVE PROPAGATION", "Trotterized Real-Time Unitary Evolution")
    ax1.legend(loc="lower right")

    ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
    ax2.plot(times, bipartite_entropy, color=GREEN_EMERALD, lw=2.4, marker="o", markersize=4, label="Bipartite Half-Cut Entropy S(A:B)")
    ax2.plot(times, mean_mag, color=GOLD_BRIGHT, lw=1.8, ls="--", label="Total Order <M_z(t)>")
    ax2.set_xlabel("Physical Time t (s)", color=TEXT_MUTED)
    ax2.set_ylabel("Entanglement Entropy S(t) (nats)", color=TEXT_MUTED)
    style_card(ax2, "PANEL 2: NON-EQUILIBRIUM ENTANGLEMENT GROWTH", "Lieb-Robinson Bound & Information Spreading")
    ax2.legend(loc="upper left")

    # 2. Right Column: 2D Spatial Heatmaps at 4 Snapshots
    # Custom Sovereign Colormap: Alert Red (Down -1.0) -> Void Charcoal (0.0) -> Laser Green (Up +1.0)
    deen_cmap = LinearSegmentedColormap.from_list(
        "deen_spin_cmap",
        [RED_ALERT, BG_DARK, GREEN_QUANTUM],
        N=256
    )

    snapshot_indices = [0, len(data) // 4, len(data) // 2, len(data) - 1]
    titles = ["t = 0.00 s (Localized Defect)", "t = 0.25 s (Radial Dispersion)", "t = 0.50 s (Edge Reflection)", "t = 1.00 s (Interference)"]

    for i, idx in enumerate(snapshot_indices):
        row = i // 2
        col = 2 + (i % 2)
        ax = plt.subplot2grid((2, 4), (row, col))
        grid_mag = np.array(data[idx]["magnetization"]).reshape((height, width))
        im = ax.imshow(grid_mag, cmap=deen_cmap, vmin=-1.0, vmax=1.0, origin="lower")
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.set_facecolor(BG_DARK)
        for spine in ax.spines.values():
            spine.set_color(GRID_SLATE)
        ax.set_title(titles[i], color=TEXT_POLAR, fontsize=9, fontweight="bold", pad=6)

        # Draw grid borders
        ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
        ax.grid(which="minor", color=GRID_SLATE, linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", size=0)

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

    png_path = output_dir / "lattice_quench_dynamics.png"
    save_deen_figure(fig, png_path)
    return png_path

if __name__ == "__main__":
    ws_root = Path(__file__).resolve().parents[2]
    json_file = ws_root / "results" / "experiments" / "lattice_quench_dynamics.json"
    out_dir = ws_root / "results" / "charts"
    if json_file.exists():
        png = chart_lattice_quench(json_file, out_dir)
        print(f" [+] Generated lattice quench dynamics chart: {png}")
    else:
        print(f" [-] Missing data file: {json_file}")
