# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Water (H2O) 2D Energy & Linearization Barrier Visualizer
Renders the complete ab-initio potential energy surface, 180° planar linearization barrier,
Walsh molecular orbital correlation, and vibrational normal modes.

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
OUTPUT_PATH = CHARTS_DIR / "water_pes_manifold.png"
JSON_PATH = PROJECT_ROOT / "results" / "hosts" / "unknown-cpu" / "experiments" / "water_pes.json"

# Sovereign Color Palette
COLOR_BG_ROOT = "#05080d"
COLOR_BG_CARD = "#0f172a"
COLOR_EMERALD = "#00e676"
COLOR_AMBER = "#ffb300"
COLOR_CRIMSON = "#ef4444"
COLOR_GRID = "#1e293b"
COLOR_TEXT = "#f8fafc"
COLOR_TEXT_DIM = "#94a3b8"

def load_telemetry():
    if not JSON_PATH.exists():
        print(f"Warning: {JSON_PATH} not found, generating theoretical model...")
        return None
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_water_pes():
    data = load_telemetry()
    
    fig = plt.figure(figsize=(16, 12), facecolor=COLOR_BG_ROOT)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.25, left=0.07, right=0.95, top=0.91, bottom=0.07)
    
    # -------------------------------------------------------------
    # Panel 1: 2D Potential Energy Surface E(r_OH, theta_HOH)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0], facecolor=COLOR_BG_CARD)
    ax1.set_title("1. 2D Potential Energy Surface E(r_OH, θ_HOH) [STO-3G RHF]", color=COLOR_TEXT, fontsize=12, fontweight="bold", pad=12)
    
    if data and "grid" in data:
        r_steps = data["grid"]["r_steps"]
        th_steps = data["grid"]["theta_steps"]
        pts = data["grid"]["points"]
        
        r_vals = np.array([p["r_oh"] for p in pts]).reshape(r_steps, th_steps)
        th_vals = np.array([p["theta_deg"] for p in pts]).reshape(r_steps, th_steps)
        e_vals = np.array([p["total_energy"] for p in pts]).reshape(r_steps, th_steps)
        
        e_min = np.min(e_vals)
        rel_e_kcal = (e_vals - e_min) * 627.509
        
        colors = [COLOR_BG_ROOT, "#0d2818", COLOR_EMERALD, COLOR_AMBER, COLOR_CRIMSON]
        cmap_pes = LinearSegmentedColormap.from_list("sovereign_pes", list(zip([0, 0.15, 0.45, 0.75, 1.0], colors)))
        
        cf = ax1.contourf(th_vals, r_vals, rel_e_kcal, levels=25, cmap=cmap_pes, alpha=0.95)
        cs = ax1.contour(th_vals, r_vals, rel_e_kcal, levels=[5, 15, 30, 50, 75, 100], colors=COLOR_TEXT_DIM, linewidths=0.8, alpha=0.6)
        ax1.clabel(cs, inline=True, fontsize=8, fmt="%d kcal")
        
        cbar = plt.colorbar(cf, ax=ax1, fraction=0.046, pad=0.04)
        cbar.set_label("Relative Energy ΔE [kcal/mol]", color=COLOR_TEXT_DIM, fontsize=9)
        cbar.ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=8)
        cbar.outline.set_edgecolor(COLOR_GRID)
        
        # Mark Equilibrium and Linear Saddle Point
        eq_pt = data["grid"]["min_energy_point"]
        ax1.scatter([eq_pt["theta_deg"]], [eq_pt["r_oh"]], color=COLOR_AMBER, s=120, edgecolors=COLOR_TEXT, linewidth=1.5, zorder=5, label=f"Equilibrium (θ={eq_pt['theta_deg']:.1f}°, r={eq_pt['r_oh']:.2f} Å)")
        
        lin_pt = data["grid"]["linearization_point"]
        ax1.scatter([180.0], [lin_pt["r_oh"]], color=COLOR_CRIMSON, s=100, marker="X", edgecolors=COLOR_TEXT, linewidth=1.2, zorder=5, label=f"Linear Saddle (180.0°, r={lin_pt['r_oh']:.2f} Å)")
        
        # Draw Minimum Energy Path (MEP)
        mep_theta = np.linspace(eq_pt["theta_deg"], 180.0, 30)
        mep_r = eq_pt["r_oh"] + (lin_pt["r_oh"] - eq_pt["r_oh"]) * ((mep_theta - eq_pt["theta_deg"]) / (180.0 - eq_pt["theta_deg"]))**2
        ax1.plot(mep_theta, mep_r, color=COLOR_TEXT, linestyle="--", linewidth=1.5, alpha=0.85, label="Minimum Energy Path")
        
    ax1.set_xlabel("Bond Angle θ_HOH [Degrees]", color=COLOR_TEXT_DIM, fontsize=10)
    ax1.set_ylabel("Bond Length r_OH [Ångström]", color=COLOR_TEXT_DIM, fontsize=10)
    ax1.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
    for spine in ax1.spines.values():
        spine.set_color(COLOR_GRID)
    ax1.legend(loc="upper left", facecolor=COLOR_BG_CARD, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, fontsize=8)
    
    # -------------------------------------------------------------
    # Panel 2: 180° Linearization Barrier & Inversion Double Well
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], facecolor=COLOR_BG_CARD)
    ax2.set_title("2. 180° Planar Inversion Barrier ΔE_lin (Double Well)", color=COLOR_TEXT, fontsize=12, fontweight="bold", pad=12)
    
    if data and "linearization" in data:
        lin = data["linearization"]
        w_states = lin["walsh_states"]
        th_line = np.array([w["theta_deg"] for w in w_states])
        e_line = np.array([w["total_energy"] for w in w_states])
        e_eq = lin["equilibrium_energy"]
        barrier_kcal = (e_line - e_eq) * 627.509
        
        # Symmetrize to show full inversion double-well potential
        th_full = np.concatenate([-(180.0 - th_line[::-1]) + 180.0, th_line])
        # Mirror around 180°: left well at 104.5°, barrier at 180°, right well at 255.5° (or inverted umbrella)
        th_mirror = 360.0 - th_line[::-1]
        th_umbrella = np.concatenate([th_line, th_mirror])
        barrier_umbrella = np.concatenate([barrier_kcal, barrier_kcal[::-1]])
        
        ax2.plot(th_umbrella, barrier_umbrella, color=COLOR_EMERALD, linewidth=2.5, label="Adiabatic Bending Potential V(θ)")
        ax2.axvline(180.0, color=COLOR_CRIMSON, linestyle="--", linewidth=1.5, alpha=0.85, label="Linear Saddle Barrier (180.0°)")
        
        # Fill barrier height region
        ax2.fill_between(th_umbrella, 0, barrier_umbrella, color=COLOR_EMERALD, alpha=0.12)
        
        # Barrier height annotation
        barr_cm1 = lin["barrier_cm1"]
        barr_ev = lin["barrier_ev"]
        ax2.annotate(f"Barrier Height:\n{barr_cm1:.1f} cm⁻¹\n({barr_ev:.2f} eV / {lin['barrier_kcal_mol']:.1f} kcal/mol)",
                     xy=(180.0, lin["barrier_kcal_mol"]), xytext=(180.0, lin["barrier_kcal_mol"] * 0.45),
                     arrowprops=dict(arrowstyle="->", color=COLOR_AMBER, lw=1.2),
                     color=COLOR_AMBER, fontsize=9, fontweight="bold", ha="center",
                     bbox=dict(boxstyle="round,pad=0.4", fc=COLOR_BG_ROOT, ec=COLOR_GRID))
        
        # Vibrational harmonic levels in well
        zpe_kcal = data["vibrations"]["total_zpe_kcal_mol"]
        ax2.axhline(zpe_kcal, color=COLOR_AMBER, linestyle=":", linewidth=1.2, label=f"ZPE Level ({zpe_kcal:.1f} kcal/mol)")
        
    ax2.set_xlabel("Bond Angle θ_HOH [Degrees]", color=COLOR_TEXT_DIM, fontsize=10)
    ax2.set_ylabel("Inversion Energy ΔE [kcal/mol]", color=COLOR_TEXT_DIM, fontsize=10)
    ax2.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
    ax2.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6)
    for spine in ax2.spines.values():
        spine.set_color(COLOR_GRID)
    ax2.legend(loc="upper center", facecolor=COLOR_BG_CARD, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, fontsize=8)
    
    # -------------------------------------------------------------
    # Panel 3: Walsh Molecular Orbital Correlation (Degeneracy at 180°)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0], facecolor=COLOR_BG_CARD)
    ax3.set_title("3. Walsh Diagram: 3a₁ vs 1b₁ Lone Pair Degeneracy at 180°", color=COLOR_TEXT, fontsize=12, fontweight="bold", pad=12)
    
    if data and "linearization" in data:
        w_states = data["linearization"]["walsh_states"]
        th_w = np.array([w["theta_deg"] for w in w_states])
        orb_3a1 = np.array([w["orbital_3a1"] for w in w_states])
        orb_1b1 = np.array([w["orbital_1b1"] for w in w_states])
        
        ax3.plot(th_w, orb_3a1, color=COLOR_AMBER, linewidth=2.2, label="3a₁ Lone Pair (s-p Hybrid, in-plane)")
        ax3.plot(th_w, orb_1b1, color=COLOR_EMERALD, linewidth=2.2, label="1b₁ Pure Non-Bonding Lone Pair (out-of-plane)")
        
        # Highlight intersection at 180°
        ax3.scatter([180.0], [orb_3a1[-1]], color=COLOR_TEXT, s=110, edgecolors=COLOR_AMBER, linewidth=2.0, zorder=5)
        ax3.annotate("Degenerate 1π_u Shell\n(Gap = 0.000 eV)", xy=(180.0, orb_3a1[-1]), xytext=(150.0, orb_3a1[-1] + 0.05),
                     arrowprops=dict(arrowstyle="->", color=COLOR_TEXT, lw=1.2),
                     color=COLOR_TEXT, fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc=COLOR_BG_ROOT, ec=COLOR_GRID))
        
    ax3.set_xlabel("Bond Angle θ_HOH [Degrees]", color=COLOR_TEXT_DIM, fontsize=10)
    ax3.set_ylabel("Orbital Energy ε_i [Hartree]", color=COLOR_TEXT_DIM, fontsize=10)
    ax3.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
    ax3.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6)
    for spine in ax3.spines.values():
        spine.set_color(COLOR_GRID)
    ax3.legend(loc="center left", facecolor=COLOR_BG_CARD, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, fontsize=8)
    
    # -------------------------------------------------------------
    # Panel 4: Normal Vibrational Spectrum & Dipole Moment Collapse
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1], facecolor=COLOR_BG_CARD)
    ax4.set_title("4. Vibrational Spectrum & Dipole Collapse |μ(θ)| → 0", color=COLOR_TEXT, fontsize=12, fontweight="bold", pad=12)
    
    if data and "linearization" in data and "vibrations" in data:
        w_states = data["linearization"]["walsh_states"]
        th_w = np.array([w["theta_deg"] for w in w_states])
        dipole_w = np.array([w["dipole_moment_debye"] for w in w_states])
        
        # Plot Dipole moment collapse on primary axis
        ax4.plot(th_w, dipole_w, color=COLOR_CRIMSON, linewidth=2.4, label="Dipole Moment |μ(θ)| [Debye]")
        ax4.scatter([180.0], [0.0], color=COLOR_AMBER, s=90, zorder=5, label="μ = 0.00 D at D_inf_h (Inversion Center)")
        ax4.set_ylabel("Dipole Moment |μ| [Debye]", color=COLOR_CRIMSON, fontsize=10)
        ax4.tick_params(axis="y", labelcolor=COLOR_CRIMSON, labelsize=9)
        
        # Inset / twin axis for normal mode frequencies
        ax4_sub = ax4.twinx()
        modes = data["vibrations"]["modes"]
        mode_names = [f"ν_{m['mode_index']}\n({m['symmetry_label']})" for m in modes]
        mode_freqs = [m["frequency_cm1"] for m in modes]
        
        # Small bar readout in inset or overlay
        for i, m in enumerate(modes):
            ax4.text(85, 1.8 - i * 0.35, f"{m['name']}: {m['frequency_cm1']:.1f} cm⁻¹ ({m['frequency_thz']:.1f} THz)",
                     color=COLOR_EMERALD, fontsize=8, family="monospace",
                     bbox=dict(boxstyle="square,pad=0.2", fc=COLOR_BG_ROOT, ec=COLOR_GRID))
            
        ax4_sub.set_ylim(0, 5000)
        ax4_sub.set_ylabel("Harmonic Frequency [cm⁻¹]", color=COLOR_EMERALD, fontsize=10)
        ax4_sub.tick_params(axis="y", labelcolor=COLOR_EMERALD, labelsize=9)
        
    ax4.set_xlabel("Bond Angle θ_HOH [Degrees]", color=COLOR_TEXT_DIM, fontsize=10)
    ax4.tick_params(colors=COLOR_TEXT_DIM, labelsize=9)
    ax4.grid(True, color=COLOR_GRID, linestyle=":", alpha=0.6)
    for spine in ax4.spines.values():
        spine.set_color(COLOR_GRID)
    ax4.legend(loc="upper right", facecolor=COLOR_BG_CARD, edgecolor=COLOR_GRID, labelcolor=COLOR_TEXT, fontsize=8)
    
    # Global Title
    fig.suptitle("T-1000 // WATER (H2O) 2D POTENTIAL ENERGY SURFACE & LINEARIZATION DYNAMICS", color=COLOR_TEXT, fontsize=14, fontweight="heavy", y=0.96)
    
    plt.savefig(OUTPUT_PATH, dpi=180, facecolor=COLOR_BG_ROOT)
    plt.close()
    print(f"Water PES Manifold chart successfully saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    plot_water_pes()

