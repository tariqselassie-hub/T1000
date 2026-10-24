# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Charting: Non-Linear Optical (NLO) & Ferroelectric Pareto Frontier.
Visualizes the 119 screened lead molecules from the Pitt Quantum Repository (106k molecules)
mapping permanent dipole moment, optical transparency band gap, polarizability, and mass.
Strictly adheres to Deen OS visual specification:
  - Deep Obsidian (#05080d), Void Charcoal (#0f172a)
  - Laser Emerald Green (#00e676), Radiant Gold (#ffb300), Alert Crimson (#ef4444)
  - Polar Slate (#f8fafc), Muted Slate Grid (#1e293b)
  - Zero emojis. Zero cyan/purple.
"""

from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from deen_theme import (
    apply_deen_theme,
    style_card,
    save_deen_figure,
    BG_OBSIDIAN,
    CARD_CHARCOAL,
    GREEN_QUANTUM,
    GREEN_EMERALD,
    GOLD_HARDWARE,
    GOLD_BRIGHT,
    RED_ALERT,
    RED_CRIMSON,
    TEXT_POLAR,
    TEXT_SLATE,
    TEXT_MUTED,
    GRID_SLATE,
)


def main() -> None:
    apply_deen_theme()

    json_path = Path("results/experiments/pitt_nlo_ferroelectric_pareto.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = data["all_candidates"]
    top_leads = data["top_leads"]

    mws = [c["molwt"] for c in candidates]
    dips = [c["dipole"] for c in candidates]
    gaps = [c["gap"] for c in candidates]
    alphas = [c["polarizability"] for c in candidates]
    foms = [c["fom_nlo"] for c in candidates]
    dxs = [abs(c["dipole_x"]) for c in candidates]
    dys = [abs(c["dipole_y"]) for c in candidates]
    dzs = [abs(c["dipole_z"]) for c in candidates]

    fig = plt.figure(figsize=(19, 13), facecolor=BG_OBSIDIAN)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)

    # -------------------------------------------------------------
    # SUBPLOT 1: Dipole Moment vs Optical Band Gap (Pareto Frontier)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    style_card(
        ax1,
        title="NLO PARETO FRONTIER: DIPOLE MOMENT VS OPTICAL BAND GAP",
        subtitle="Optically Transparent Ferroelectric Sweet Spot (Dipole >= 20 D, Gap >= 3.0 eV)",
    )

    from matplotlib.colors import LinearSegmentedColormap
    deen_cmap = LinearSegmentedColormap.from_list("deen_gold_emerald", [GOLD_HARDWARE, GREEN_QUANTUM])

    sc1 = ax1.scatter(
        gaps,
        dips,
        c=foms,
        cmap=deen_cmap,
        s=75,
        alpha=0.88,
        edgecolors=GOLD_BRIGHT,
        linewidths=0.7,
        zorder=4,
    )
    cbar1 = fig.colorbar(sc1, ax=ax1, pad=0.02)
    cbar1.set_label("NLO FOM (Dipole * Polarizability)", color=TEXT_SLATE, fontsize=8)
    cbar1.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelcolor=TEXT_MUTED)

    # Threshold guidelines
    ax1.axvline(3.0, color=RED_ALERT, linestyle="--", lw=1.2, alpha=0.7, zorder=2)
    ax1.axhline(20.0, color=RED_ALERT, linestyle="--", lw=1.2, alpha=0.7, zorder=2)
    ax1.text(3.1, 21.0, "SELECTION THRESHOLDS", color=RED_ALERT, fontsize=7.5, fontweight="bold")

    # Highlight Top #1
    lead1 = top_leads[0]
    ax1.scatter([lead1["gap"]], [lead1["dipole"]], color=RED_ALERT, s=180, edgecolors=GOLD_BRIGHT, lw=1.8, zorder=6)
    ax1.annotate(
        f"#1 NLO LEAD ({lead1['formula']})\nFOM: {lead1['fom_nlo']:.1f}\nDipole: {lead1['dipole']:.1f} D | Gap: {lead1['gap']:.2f} eV\nAlpha: {lead1['polarizability']:.1f}",
        xy=(lead1["gap"], lead1["dipole"]),
        xytext=(lead1["gap"] - 3.2, lead1["dipole"] + 4.5),
        arrowprops=dict(facecolor=GOLD_BRIGHT, edgecolor=GRID_SLATE, width=1.0, headwidth=5),
        color=TEXT_POLAR,
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.5", fc=CARD_CHARCOAL, ec=GREEN_QUANTUM, lw=1.2),
    )

    ax1.set_xlabel("Optical Transparency Band Gap (eV) [LUMO - HOMO]", fontsize=9, color=TEXT_SLATE)
    ax1.set_ylabel("Permanent Dipole Moment |D| (Debye)", fontsize=9, color=TEXT_SLATE)
    ax1.set_xlim(2.5, 9.5)
    ax1.set_ylim(18.0, 54.0)

    # -------------------------------------------------------------
    # SUBPLOT 2: 3D Dipole Vector Decomposition (|Dx|, |Dy|, |Dz|)
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    style_card(
        ax2,
        title="3D DIPOLE VECTOR ANISOTROPY (TOP 12 LEADS)",
        subtitle="Uniaxial vs Triaxial Polarization Vectors across Molecular Axes",
    )

    top12 = top_leads[:12]
    indices = np.arange(len(top12))
    width = 0.25

    t_dx = [abs(c["dipole_x"]) for c in top12]
    t_dy = [abs(c["dipole_y"]) for c in top12]
    t_dz = [abs(c["dipole_z"]) for c in top12]

    b1 = ax2.bar(indices - width, t_dx, width, label="|D_x| Axis", color=GOLD_HARDWARE, alpha=0.9, edgecolor=GRID_SLATE)
    b2 = ax2.bar(indices, t_dy, width, label="|D_y| Axis", color=GREEN_EMERALD, alpha=0.9, edgecolor=GRID_SLATE)
    b3 = ax2.bar(indices + width, t_dz, width, label="|D_z| Axis", color=RED_ALERT, alpha=0.9, edgecolor=GRID_SLATE)

    ax2.set_xticks(indices)
    ax2.set_xticklabels([f"#{c['rank']}\n{c['formula'][:7]}" for c in top12], fontsize=7.5, color=TEXT_MUTED)
    ax2.set_xlabel("Ranked Lead Candidate", fontsize=9, color=TEXT_SLATE)
    ax2.set_ylabel("Cartesian Dipole Magnitude (Debye)", fontsize=9, color=TEXT_SLATE)
    ax2.set_ylim(0, 48)
    ax2.legend(loc="upper right", framealpha=0.6, facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=8)

    # -------------------------------------------------------------
    # SUBPLOT 3: NLO Figure of Merit vs Molecular Weight
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    style_card(
        ax3,
        title="SPECIFIC NLO DENSITY: FOM VS MOLECULAR MASS",
        subtitle="High-Performance Low-Mass Materials for Vacuum Sublimation & Thin Films",
    )

    sc3 = ax3.scatter(
        mws,
        foms,
        c=gaps,
        cmap="summer",  # Yellow to Green
        s=80,
        alpha=0.9,
        edgecolors=GOLD_BRIGHT,
        linewidths=0.8,
        zorder=4,
    )
    cbar3 = fig.colorbar(sc3, ax=ax3, pad=0.02)
    cbar3.set_label("Band Gap (eV)", color=TEXT_SLATE, fontsize=8)
    cbar3.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelcolor=TEXT_MUTED)

    # Callout for low MW high FOM leads
    ax3.annotate(
        "Low-MW Thin-Film Sweet Spot\n(MW < 350 Da, FOM > 3,000)",
        xy=(lead1["molwt"], lead1["fom_nlo"]),
        xytext=(lead1["molwt"] + 25, lead1["fom_nlo"] - 400),
        arrowprops=dict(facecolor=GOLD_BRIGHT, edgecolor=GRID_SLATE, width=1.0, headwidth=5),
        color=TEXT_POLAR,
        fontsize=8,
        bbox=dict(boxstyle="square,pad=0.4", fc=CARD_CHARCOAL, ec=GOLD_HARDWARE, lw=1.0),
    )

    ax3.set_xlabel("Molecular Weight (Da) [Processability <= 500 Da]", fontsize=9, color=TEXT_SLATE)
    ax3.set_ylabel("NLO Figure of Merit (FOM = |D| * Alpha)", fontsize=9, color=TEXT_SLATE)
    ax3.set_xlim(220, 520)
    ax3.set_ylim(500, 4200)

    # -------------------------------------------------------------
    # SUBPLOT 4: Lead Candidate Pharmacophore / Electronic Matrix
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    style_card(
        ax4,
        title="SOVEREIGN LEAD TELEMETRY & PHYSICAL PROPERTIES",
        subtitle="Quantum Invariants for Top 6 Non-Linear Optical Candidates",
    )
    ax4.axis("off")

    headers = ["Rank", "Formula", "MW (Da)", "Dipole(D)", "Gap(eV)", "Alpha", "FOM(NLO)"]
    rows = []
    for c in top_leads[:6]:
        rows.append([
            f"#{c['rank']}",
            c["formula"],
            f"{c['molwt']:.1f}",
            f"{c['dipole']:.2f}",
            f"{c['gap']:.2f}",
            f"{c['polarizability']:.1f}",
            f"{c['fom_nlo']:.1f}",
        ])

    table = ax4.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.08, 0.16, 0.13, 0.14, 0.13, 0.13, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    # Style table cells strictly in Deen theme
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_SLATE)
        if row_idx == 0:
            cell.set_facecolor(CARD_CHARCOAL)
            cell.set_text_props(color=GOLD_BRIGHT, weight="bold")
        else:
            cell.set_facecolor(BG_OBSIDIAN if row_idx % 2 == 0 else CARD_CHARCOAL)
            if col_idx == 0:
                cell.set_text_props(color=GREEN_QUANTUM, weight="bold")
            elif col_idx in [3, 6]:
                cell.set_text_props(color=GOLD_HARDWARE, weight="bold")
            else:
                cell.set_text_props(color=TEXT_POLAR)

    # Title header
    fig.suptitle(
        "T-1000 PITT QUANTUM REPOSITORY: NON-LINEAR OPTICAL & FERROELECTRIC PARETO FRONTIER\n"
        "Screened 105,926 Molecules | 119 Processable Wide-Gap Leads | Deen OS Sovereign Specification",
        fontsize=12,
        fontweight="bold",
        color=TEXT_POLAR,
        y=0.98,
    )

    out_file = Path("results/charts/pitt_nlo_ferroelectrics.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_deen_figure(fig, out_file)
    print(f"[+] Sovereign NLO Chart Successfully Generated: {out_file}")


if __name__ == "__main__":
    main()
