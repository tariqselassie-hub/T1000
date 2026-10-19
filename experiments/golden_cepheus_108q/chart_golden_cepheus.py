# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Rigetti Cepheus-1-108Q Golden Benchmark Visualizer.
Renders 4-panel sovereign publication diagram in strict Deen OS Visual Specification:
- Deep Obsidian (#05080d)
- Laser Emerald Green (#00e676)
- Radiant Gold (#ffb300)
- Alert Red (#ef4444)
- Polar Slate/White (#f8fafc)
Strictly ZERO cyan, ZERO purple, ZERO emojis.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

BG_OBSIDIAN = "#05080d"
CARD_DARK = "#0d131f"
CARD_CHARCOAL = "#111827"
GRID_SLATE = "#1e293b"
GREEN_QUANTUM = "#00e676"
GREEN_EMERALD = "#00ff88"
GOLD_HARDWARE = "#ffb300"
GOLD_BRIGHT = "#ffd700"
RED_ALERT = "#ef4444"
RED_CRIMSON = "#dc2626"
TEXT_POLAR = "#f8fafc"
TEXT_SLATE = "#94a3b8"
TEXT_MUTED = "#64748b"


def chart_golden_cepheus():
    cur_dir = Path(__file__).resolve().parent
    json_path = cur_dir / "cepheus_108q_telemetry.json"
    out_png = cur_dir / "cepheus_108q_golden_showdown.png"

    if not json_path.exists():
        print(f"[!] Telemetry file missing: {json_path}")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plaquettes = data["plaquettes"]
    cols, rows = 11, 8

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 GOLDEN EXPERIMENT: RIGETTI CEPHEUS-1-108Q MODULAR SEAM MITIGATION",
        color=TEXT_POLAR,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # -------------------------------------------------------------------------
    # Panel 1: 12x9 Modular Chiplet Lattice & 88 Plaquette Seam Topology
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor(CARD_DARK)
    ax1.set_title(
        "PANEL 1: Cepheus-108Q 12-Chiplet Lattice & 88-Plaquette Seam Map",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    for p in plaquettes:
        c, r = p["col"], p["row"]
        fid = p["fid_teleport_pauli"]
        zone = p["zone_type"]

        if zone == "Intra-Chiplet":
            bg_c = "#0a2e1d"
            edge_c = GREEN_QUANTUM
            lw = 1.0
        elif zone == "Seam-Straddling":
            bg_c = "#2d2305"
            edge_c = GOLD_HARDWARE
            lw = 1.4
        else: # 4-Chiplet Corner
            bg_c = "#3b1111"
            edge_c = RED_ALERT
            lw = 2.0

        rect = patches.Rectangle((c, r), 0.90, 0.90, facecolor=bg_c, edgecolor=edge_c, linewidth=lw, zorder=2)
        ax1.add_patch(rect)

        txt_col = RED_ALERT if zone == "4-Chiplet Corner" else TEXT_POLAR
        ax1.text(
            c + 0.45, r + 0.45,
            f"P{p['plaquette_id']}\n{fid:.4f}",
            color=txt_col,
            fontsize=6.5,
            fontweight="bold",
            ha="center", va="center",
            zorder=3,
        )

    # Draw Inter-Chiplet Seams (every 3 columns and every 3 rows)
    for sc in [2.95, 5.95, 8.95]:
        ax1.axvline(sc, color=GOLD_BRIGHT, linestyle="--", linewidth=2.2, alpha=0.9, zorder=4)

    for sr in [2.95, 5.95]:
        ax1.axhline(sr, color=GOLD_BRIGHT, linestyle="--", linewidth=2.2, alpha=0.9, zorder=4)

    # Chiplet labels
    ax1.text(1.5, 7.65, "CHIPLET 0-3", color=TEXT_MUTED, fontsize=8, ha="center")
    ax1.text(4.5, 7.65, "CHIPLET 4-7", color=TEXT_MUTED, fontsize=8, ha="center")
    ax1.text(7.5, 7.65, "CHIPLET 8-11", color=TEXT_MUTED, fontsize=8, ha="center")

    ax1.set_xlim(-0.3, cols + 0.3)
    ax1.set_ylim(-0.3, rows + 0.3)
    ax1.set_xlabel("Plaquette Column Index (0 .. 10)", color=TEXT_SLATE, fontsize=9)
    ax1.set_ylabel("Plaquette Row Index (0 .. 7)", color=TEXT_SLATE, fontsize=9)
    ax1.set_xticks(range(cols))
    ax1.set_yticks(range(rows))
    ax1.tick_params(colors=TEXT_SLATE)
    ax1.grid(False)

    legend_patches = [
        patches.Patch(facecolor="#0a2e1d", edgecolor=GREEN_QUANTUM, label="Intra-Chiplet (p=0.8%, F=0.9683)"),
        patches.Patch(facecolor="#2d2305", edgecolor=GOLD_HARDWARE, label="Seam-Straddling (p=2.2%, F=0.9414)"),
        patches.Patch(facecolor="#3b1111", edgecolor=RED_ALERT, label="4-Chiplet Corner Junction (F=0.9146)"),
    ]
    ax1.legend(handles=legend_patches, loc="lower right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 2: Golden Showdown: SWAP vs 4-Edge Teleportation Across Zones
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor(CARD_DARK)
    ax2.set_title(
        "PANEL 2: Seam Degradation & Teleport Mitigation Across Architectural Zones",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    zones = [
        "Intra-Chiplet\n(48 Plaquettes)",
        "Seam-Straddling\n(34 Plaquettes)",
        "4-Chiplet Corner\n(6 Plaquettes)",
        "Overall Cepheus\n(88 Plaquettes)"
    ]
    f_swap = [
        data["mean_fid_swap_intra"],
        data["mean_fid_swap_seam"],
        data["mean_fid_swap_corner"],
        data["overall_mean_swap"]
    ]
    f_teleport = [
        data["mean_fid_teleport_intra"],
        data["mean_fid_teleport_seam"],
        data["mean_fid_teleport_corner"],
        data["overall_mean_teleport"]
    ]

    x = np.arange(len(zones))
    w = 0.35

    b1 = ax2.bar(x - w/2, f_swap, w, label="Protocol 1: Sequential SWAP (3 CNOTs)", color=RED_ALERT, edgecolor=RED_CRIMSON, zorder=3)
    b2 = ax2.bar(x + w/2, f_teleport, w, label="Protocol 6: 4-Edge Teleport + Pauli Frame", color=GREEN_QUANTUM, edgecolor=GREEN_EMERALD, zorder=3)

    for bar, val in zip(b1, f_swap):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.4f}", ha="center", va="bottom", color=TEXT_POLAR, fontsize=8.5, fontweight="bold")

    for bar, val in zip(b2, f_teleport):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.4f}", ha="center", va="bottom", color=GREEN_QUANTUM, fontsize=8.5, fontweight="bold")

    ax2.set_ylabel("Diagonal Bell State Fidelity F", color=TEXT_SLATE, fontsize=10)
    ax2.set_ylim(0.82, 0.99)
    ax2.set_xticks(x)
    ax2.set_xticklabels(zones, fontsize=9, color=TEXT_POLAR)
    ax2.tick_params(colors=TEXT_SLATE)
    ax2.legend(loc="lower left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=9)
    ax2.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.5, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 3: 4-Chiplet Corner Critical Focus (P24, P27, P30, P57, P60, P63)
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor(CARD_DARK)
    ax3.set_title(
        "PANEL 3: Critical 4-Chiplet Corner Intersections (All 4 Edges Bump-Bond Seams)",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    corner_pids = [24, 27, 30, 57, 60, 63]
    corner_labels = [f"P{pid}" for pid in corner_pids]
    corner_swap = []
    corner_teleport = []
    corner_boost = []

    for pid in corner_pids:
        match = next(p for p in plaquettes if p["plaquette_id"] == pid)
        s_val = match["fid_swap"]
        t_val = match["fid_teleport_pauli"]
        corner_swap.append(s_val)
        corner_teleport.append(t_val)
        corner_boost.append((t_val - s_val) * 100.0)

    xc = np.arange(len(corner_pids))
    wc = 0.35

    ax3.bar(xc - wc/2, corner_swap, wc, label="SWAP on Seams", color=GOLD_HARDWARE, edgecolor=GOLD_BRIGHT, zorder=3)
    ax3.bar(xc + wc/2, corner_teleport, wc, label="4-Edge Teleport + Pauli Frame", color=GREEN_QUANTUM, edgecolor=GREEN_EMERALD, zorder=3)

    for i in range(len(corner_pids)):
        ax3.text(xc[i] - wc/2, corner_swap[i] + 0.002, f"{corner_swap[i]:.4f}", ha="center", va="bottom", color=TEXT_POLAR, fontsize=8, fontweight="bold")
        ax3.text(xc[i] + wc/2, corner_teleport[i] + 0.002, f"{corner_teleport[i]:.4f}\n(+{corner_boost[i]:.2f}%)", ha="center", va="bottom", color=GREEN_QUANTUM, fontsize=7.5, fontweight="bold")

    ax3.set_ylabel("Corner Plaquette Fidelity F", color=TEXT_SLATE, fontsize=10)
    ax3.set_ylim(0.83, 0.95)
    ax3.set_xticks(xc)
    ax3.set_xticklabels(corner_labels, fontsize=10, color=TEXT_POLAR)
    ax3.tick_params(colors=TEXT_SLATE)
    ax3.legend(loc="lower left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=9)
    ax3.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.5, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 4: Golden Experiment Hardware Coupler & Audit Matrix
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor(CARD_DARK)
    ax4.set_title(
        "PANEL 4: Rigetti Cepheus-1-108Q Machine Audit & Golden Parameters",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )
    ax4.axis("off")

    metrics_table = [
        ["Target QPU Architecture", "Rigetti Cepheus-1-108Q Modular Superconducting Array", GOLD_BRIGHT],
        ["Lattice Dimensions", f"{data['total_qubits']} Physical Qubits (12x9 Planar Grid)", TEXT_POLAR],
        ["Modular Die Partitioning", "Twelve 3x3 Chiplets (4 Columns x 3 Rows)", GREEN_QUANTUM],
        ["Total Physical Couplers", f"{data['total_couplers']} Couplers (99 Horizontal + 96 Vertical)", TEXT_POLAR],
        ["Intra-Chiplet Couplers", f"{data['intra_die_couplers']} Couplers (p_gate = 0.80%, F_SWAP = 0.9368)", GREEN_QUANTUM],
        ["Inter-Chiplet Seam Bump Bonds", f"{data['seam_couplers']} Seams (p_seam = 2.20%, F_SWAP = 0.8977)", RED_ALERT],
        ["Total 2x2 Plaquette Tiles", f"{data['total_plaquettes']} Plaquettes (11 Columns x 8 Rows)", TEXT_POLAR],
        ["Intra-Die Plaquettes", f"{data['intra_die_plaquettes']} Plaquettes (Mean Teleport F = {data['mean_fid_teleport_intra']:.4f})", GREEN_QUANTUM],
        ["Seam-Straddling Plaquettes", f"{data['seam_straddling_plaquettes']} Plaquettes (Mean Teleport F = {data['mean_fid_teleport_seam']:.4f})", GOLD_BRIGHT],
        ["4-Chiplet Corner Plaquettes", f"{data['corner_4chiplet_plaquettes']} Plaquettes (Critical Seam F = {data['mean_fid_teleport_corner']:.4f})", RED_ALERT],
        ["Absolute Seam Purification Boost", "+5.58% Absolute Fidelity Gain (0.8588 -> 0.9146)", GREEN_QUANTUM],
        ["Pauli Z-Frame Recovery Acceptance", f"{data['overall_mean_keep']*100:.1f}% Usable Subspace (Zero Gate Delay)", GREEN_QUANTUM],
        ["T-1000 Pure-Rust Simulation Latency", f"{data['execution_time_us']:.2f} microseconds (Full 88-Plaquette Sweep)", GOLD_BRIGHT],
        ["Quil Submission Suite", "7 Native Quil 2.0 Circuits Ready for Cepheus QCS", GREEN_QUANTUM],
    ]

    y_start = 0.94
    dy = 0.065

    for idx, (param, val, color) in enumerate(metrics_table):
        y = y_start - idx * dy
        bg_c = CARD_CHARCOAL if idx % 2 == 0 else CARD_DARK
        rect = patches.Rectangle((0.02, y - 0.022), 0.96, 0.052, facecolor=bg_c, edgecolor=GRID_SLATE, linewidth=0.5, zorder=1)
        ax4.add_patch(rect)
        ax4.text(0.04, y, param, color=TEXT_SLATE, fontsize=8.2, va="center", zorder=2)
        ax4.text(0.96, y, val, color=color, fontsize=8.2, fontweight="bold", ha="right", va="center", zorder=2)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_png, dpi=300, facecolor=BG_OBSIDIAN, edgecolor="none")
    plt.close()

    print(f" [+] Rendered Golden Cepheus Visual: {out_png} ({out_png.stat().st_size / 1024:.1f} KB)")
    return out_png


if __name__ == "__main__":
    chart_golden_cepheus()
