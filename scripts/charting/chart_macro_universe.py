# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Macroscopic Universe & Cosmic Web Visualizer
Renders the macroscopic emergence of the universe:
- Zel'dovich gravitational displacement caustics
- Cosmic web filaments, galaxy cluster nodes, and voids
- Expanding cosmological event horizon R_H(t)
- Matter density contrast field slice delta(x)

Strictly adheres to Deen OS Sovereign Visual Specification:
Deep Obsidian (#05080d), Void Charcoal (#0f172a), Laser Emerald (#00e676),
Radiant Amber (#ffb300), Crimson Red (#ef4444), Polar Slate (#f8fafc).
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHARTS_DIR = PROJECT_ROOT / "results" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = CHARTS_DIR / "macro_universe_cosmic_web.png"

# Sovereign Color Palette
COLOR_BG_ROOT = "#05080d"
COLOR_BG_CARD = "#0f172a"
COLOR_EMERALD = "#00e676"
COLOR_AMBER = "#ffb300"
COLOR_CRIMSON = "#ef4444"
COLOR_GRID = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_TEXT_DIM = "#94a3b8"

def simulate_zeldovich_cosmic_web(n_particles=400, box_size=20.0, scale_a=20.0, hubble_h=1.2):
    """Computes Zel'dovich approximation for macroscopic cosmic web formation."""
    np.random.seed(42)
    
    # Golden spiral distribution in sphere
    indices = np.arange(0, n_particles, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_particles)
    theta = np.pi * (1 + 5**0.5) * indices
    r = box_size * 0.45 * (indices / n_particles)**(1/3)
    
    qx = r * np.sin(phi) * np.cos(theta)
    qy = r * np.sin(phi) * np.sin(theta)
    qz = r * np.cos(phi)
    
    # Primordial wave modes (Harrison-Zeldovich scale-invariant spectrum)
    n_modes = 16
    psi_x = np.zeros(n_particles)
    psi_y = np.zeros(n_particles)
    psi_z = np.zeros(n_particles)
    
    for m in range(1, n_modes + 1):
        km = 2.0 * np.pi * m / box_size
        amp = (1.0 / (m**1.5)) * 0.35
        phase_x = m * 1.37
        phase_y = m * 2.19
        phase_z = m * 3.71
        
        psi_x += amp * km * np.sin(km * qx + phase_x)
        psi_y += amp * km * np.sin(km * qy + phase_y)
        psi_z += amp * km * np.sin(km * qz + phase_z)
        
    # Growth factor D(a)
    growth = (scale_a - 1.0) / 10.0 + 0.05
    
    # Eulerian displaced coordinates x = a * (q - D * Psi)
    x = qx - growth * psi_x * 0.4
    y = qy - growth * psi_y * 0.4
    z = qz - growth * psi_z * 0.4
    
    # Density contrast delta ~ growth * div(Psi)
    psi_mag = np.sqrt(psi_x**2 + psi_y**2 + psi_z**2)
    delta = growth * psi_mag * 1.8 - 0.2
    
    # Horizon radius R_H
    r_horizon = box_size * 0.48
    
    return qx, qy, x, y, z, delta, r_horizon

def plot_macro_universe():
    qx, qy, x, y, z, delta, r_horizon = simulate_zeldovich_cosmic_web()
    
    fig = plt.figure(figsize=(16, 12), facecolor=COLOR_BG_ROOT)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25, left=0.07, right=0.95, top=0.91, bottom=0.07)
    
    # -------------------------------------------------------------
    # Panel 1: Macroscopic Cosmic Web & Filamentary Network
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], facecolor=COLOR_BG_CARD)
    ax1.set_title("1. Macroscopic Cosmic Web & Filamentary Network", color=COLOR_TEXT, fontsize=12, fontweight="bold", pad=12)
    
    # Draw Event Horizon Sphere boundary
    horizon_circle = plt.Circle((0, 0), r_horizon, fill=False, color=COLOR_CRIMSON, linestyle="--", linewidth=1.5, alpha=0.85, label="Event Horizon R_H(t)")
    ax1.add_patch(horizon_circle)
    
    # Draw Filaments connecting high density nodes
    high_density_idx = np.where(delta > 0.3)[0]
    for i in range(len(high_density_idx)):
        idx_i = high_density_idx[i]
        for j in range(i + 1, min(i + 8, len(high_density_idx))):
            idx_j = high_density_idx[j]
            dist = np.sqrt((x[idx_i] - x[idx_j])**2 + (y[idx_i] - y[idx_j])**2)
            if dist < 4.2:
                alpha = max(0.15, min(0.7, 1.0 - dist / 4.2))
                ax1.plot([x[idx_i], x[idx_j]], [y[idx_i], y[idx_j]], color=COLOR_EMERALD, alpha=alpha, linewidth=0.9, zorder=2)
                
    # Plot Galaxy Clusters and Halos
    clusters = delta > 1.686
    filaments = (delta > 0.3) & ~clusters
    voids = delta <= 0.3
    
    ax1.scatter(x[voids], y[voids], s=12, color=COLOR_TEXT_DIM, alpha=0.35, label="Void Tracer Particles", zorder=1)
    ax1.scatter(x[filaments], y[filaments], s=35, color=COLOR_EMERALD, alpha=0.8, edgecolors="none", label="Filament Bridges", zorder=3)
    ax1.scatter(x[clusters], y[clusters], s=95, color=COLOR_AMBER, alpha=0.95, edgecolors=COLOR_TEXT, linewidth=1.0, label="Galaxy Superclusters (δ > 1.686)", zorder=4)
    
    ax1.set_xlim(-12, 12)
    ax1.set_ylim(-12, 12)
    ax1.set_xlabel("Comoving X [Mpc / h]", color=COLOR_TEXT_DIM, fontsize=10)
    ax1.set_ylabel("Comoving Y [Mpc / h]", color=COLOR_TEXT_DIM, fontsize=10)
    ax1.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
    ax1.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6)
    for spine in ax1.spines.values():
        spine.set_color(COLOR_GRID)
    ax1.legend(loc="upper right", facecolor=COLOR_BG_CARD, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, fontsize=8)
    
    # -------------------------------------------------------------
    # Panel 2: Primordial Density Contrast Field Slice delta(x, y)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor=COLOR_BG_CARD)
    ax2.set_title("2. Primordial Matter Density Contrast Field δ(x, y)", color=COLOR_TEXT, fontsize=12, fontweight="bold", pad=12)
    
    grid_res = 120
    gx = np.linspace(-10, 10, grid_res)
    gy = np.linspace(-10, 10, grid_res)
    X, Y = np.meshgrid(gx, gy)
    
    # Synthesize continuous density field from wave superposition
    Z = np.zeros_like(X)
    for m in range(1, 12):
        km = 2.0 * np.pi * m / 20.0
        amp = (1.0 / (m**1.2)) * 0.4
        Z += amp * np.sin(km * X + m * 1.3) * np.cos(km * Y + m * 2.1)
    Z = Z * 2.2 + 0.1
    
    # Sovereign Emerald/Amber colormap
    colors = [COLOR_BG_ROOT, "#0d2818", COLOR_EMERALD, COLOR_AMBER, COLOR_TEXT]
    nodes = [0.0, 0.25, 0.55, 0.85, 1.0]
    cmap_cosmic = LinearSegmentedColormap.from_list("sovereign_cosmic", list(zip(nodes, colors)))
    
    im2 = ax2.imshow(Z, extent=[-10, 10, -10, 10], origin="lower", cmap=cmap_cosmic, aspect="auto")
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Overdensity δ = (ρ - ρ̄) / ρ̄", color=COLOR_TEXT_DIM, fontsize=9)
    cbar2.ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=8)
    cbar2.outline.set_edgecolor(COLOR_GRID)
    
    ax2.set_xlabel("Spatial Coordinate X [Mpc / h]", color=COLOR_TEXT_DIM, fontsize=10)
    ax2.set_ylabel("Spatial Coordinate Y [Mpc / h]", color=COLOR_TEXT_DIM, fontsize=10)
    ax2.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
    for spine in ax2.spines.values():
        spine.set_color(COLOR_GRID)
        
    # -------------------------------------------------------------
    # Panel 3: Cosmological Horizon vs Structure Scale Growth
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], facecolor=COLOR_BG_CARD)
    ax3.set_title("3. Cosmological Horizon R_H(t) vs Physical Structure Scale", color=COLOR_TEXT, fontsize=12, fontweight="bold", pad=12)
    
    t_vals = np.linspace(0.0, 2.5, 100)
    hubble = 1.2
    scale_a_t = np.exp(hubble * t_vals)
    comoving_h = 1.0 / (scale_a_t * hubble)
    phys_horizon = 1.0 / hubble  # constant in de Sitter
    
    k_modes = [0.314, 0.628, 1.256, 2.513]
    colors_modes = [COLOR_EMERALD, COLOR_AMBER, "#38bdf8", COLOR_CRIMSON]
    
    ax3.plot(t_vals, comoving_h, color=COLOR_CRIMSON, linewidth=2.4, label="Comoving Hubble Radius r_c = (aH)⁻¹", zorder=4)
    for i, km in enumerate(k_modes):
        phys_lambda = 2.0 * np.pi / (km * scale_a_t)
        ax3.plot(t_vals, [2.0 * np.pi / km] * len(t_vals), linestyle="--", linewidth=1.2, color=colors_modes[i % 2], alpha=0.75, label=f"Mode λ = {2.0*np.pi/km:.1f} Mpc")
        
    ax3.set_yscale("log")
    ax3.set_xlabel("Cosmic Time t [H⁻¹]", color=COLOR_TEXT_DIM, fontsize=10)
    ax3.set_ylabel("Comoving Horizon / Scale [Mpc / h]", color=COLOR_TEXT_DIM, fontsize=10)
    ax3.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
    ax3.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6)
    for spine in ax3.spines.values():
        spine.set_color(COLOR_GRID)
    ax3.legend(loc="upper right", facecolor=COLOR_BG_CARD, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, fontsize=8)
    
    # -------------------------------------------------------------
    # Panel 4: Halo Mass / Density Function & Spherical Collapse
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], facecolor=COLOR_BG_CARD)
    ax4.set_title("4. Halo Density Distribution (Gunn-Gott Spherical Collapse)", color=COLOR_TEXT, fontsize=12, fontweight="bold", pad=12)
    
    delta_bins = np.linspace(-1.0, 3.5, 35)
    counts, bins = np.histogram(delta, bins=delta_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    
    ax4.bar(bin_centers[bin_centers < 0.3], counts[bin_centers < 0.3], width=0.11, color=COLOR_GRID, edgecolor=COLOR_TEXT_DIM, alpha=0.7, label="Voids (δ < 0.3)")
    ax4.bar(bin_centers[(bin_centers >= 0.3) & (bin_centers < 1.686)], counts[(bin_centers >= 0.3) & (bin_centers < 1.686)], width=0.11, color=COLOR_EMERALD, edgecolor=COLOR_BG_ROOT, alpha=0.85, label="Filaments & Sheets")
    ax4.bar(bin_centers[bin_centers >= 1.686], counts[bin_centers >= 1.686], width=0.11, color=COLOR_AMBER, edgecolor=COLOR_BG_ROOT, alpha=0.95, label="Virialized Halos (δ > 1.686)")
    
    ax4.axvline(1.686, color=COLOR_CRIMSON, linestyle="--", linewidth=1.8, label="Linear Collapse Barrier δ_c = 1.686")
    
    ax4.set_xlabel("Density Perturbation δ", color=COLOR_TEXT_DIM, fontsize=10)
    ax4.set_ylabel("Particle / Tracer Count", color=COLOR_TEXT_DIM, fontsize=10)
    ax4.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
    ax4.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6)
    for spine in ax4.spines.values():
        spine.set_color(COLOR_GRID)
    ax4.legend(loc="upper right", facecolor=COLOR_BG_CARD, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, fontsize=8)
    
    # Main Header
    fig.suptitle("T-1000 // MACROSCOPIC COSMIC WEB & HORIZON SPHERE SIMULATION", color=COLOR_TEXT, fontsize=15, fontweight="heavy", y=0.96)
    
    plt.savefig(OUTPUT_PATH, dpi=180, facecolor=COLOR_BG_ROOT)
    plt.close()
    print(f"Macro Universe chart saved successfully to: {OUTPUT_PATH}")

if __name__ == "__main__":
    plot_macro_universe()

