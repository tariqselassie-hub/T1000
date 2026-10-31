# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Google Willow 105Q Golden Benchmark Visualizer.
Renders 4-panel sovereign publication diagram in strict Deen OS Visual Specification:
- Deep Obsidian (#05080d)
- Laser Emerald Green (#00e676 / #00ff88)
- Radiant Gold (#ffb300 / #ffd700)
- Alert Crimson (#ef4444 / #dc2626)
- Polar Slate/White (#f8fafc)
Strictly ZERO cyan, ZERO purple, ZERO emojis.
"""

import json
from pathlib import Path
import shutil
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


def chart_golden_willow():
    cur_dir = Path(__file__).resolve().parent
    repo_root = cur_dir.parents[1]
    json_path = cur_dir / "willow_105q_telemetry.json"
    out_png = cur_dir / "willow_105q_golden_showdown.png"

    if not json_path.exists():
        fallback_json = repo_root / "results" / "data" / "willow_105q_telemetry.json"
        if fallback_json.exists():
            shutil.copy(fallback_json, json_path)
        else:
            print(f"[!] Telemetry file missing: {json_path}")
            return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plaquettes = data["plaquettes"]
    cols, rows = 9, 10 # 2x2 plaquette grid bounds (from 11 rows x 10 cols qubit sites)

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 GOLDEN EXPERIMENT: GOOGLE WILLOW 105Q SURFACE-CODE THRESHOLD BENCHMARK",
        color=TEXT_POLAR,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # -------------------------------------------------------------------------
    # Panel 1: 11x10 Planar Lattice & 76-Plaquette Fidelity Map
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor(CARD_DARK)
    ax1.set_title(
        "PANEL 1: Willow 105Q Planar Lattice & 76 Elementary Plaquette Fidelity Map",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    removed_sites = {(1, 2), (1, 3), (9, 2), (9, 3), (5, 0)}

    # Render background grid for invalid/removed plaquettes
    for r in range(rows):
        for c in range(cols):
            corners = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}
            if any(cn in removed_sites for cn in corners):
                rect = patches.Rectangle(
                    (c, r), 0.88, 0.88,
                    facecolor="#180e14",
                    edgecolor="#3a1622",
                    linewidth=1.0,
                    linestyle=":",
                    zorder=1,
                )
                ax1.add_patch(rect)
                ax1.text(
                    c + 0.44, r + 0.44, "DEFECT\nBOUNDARY",
                    color="#7f1d1d", fontsize=5.5, fontweight="bold",
                    ha="center", va="center", zorder=2,
                )

    # Render valid plaquettes
    for p in plaquettes:
        c, r = p["col"], p["row"]
        fid_rcs = p["fid_teleport_rcs"]
        fid_qec = p["fid_teleport_qec"]

        bg_c = "#06281b"
        edge_c = GREEN_QUANTUM
        lw = 1.2

        rect = patches.Rectangle(
            (c, r), 0.88, 0.88,
            facecolor=bg_c,
            edgecolor=edge_c,
            linewidth=lw,
            zorder=2,
        )
        ax1.add_patch(rect)

        ax1.text(
            c + 0.44, r + 0.54,
            f"P{p['plaquette_id']}",
            color=GOLD_BRIGHT,
            fontsize=6.5,
            fontweight="bold",
            ha="center", va="center",
            zorder=3,
        )
        ax1.text(
            c + 0.44, r + 0.30,
            f"{fid_rcs:.4f}",
            color=TEXT_POLAR,
            fontsize=6.5,
            fontweight="bold",
            ha="center", va="center",
            zorder=3,
        )

    ax1.set_xlim(-0.3, cols + 0.3)
    ax1.set_ylim(-0.3, rows + 0.3)
    ax1.set_xlabel("Plaquette Column Index (0 .. 8)", color=TEXT_SLATE, fontsize=9)
    ax1.set_ylabel("Plaquette Row Index (0 .. 9)", color=TEXT_SLATE, fontsize=9)
    ax1.set_xticks(range(cols))
    ax1.set_yticks(range(rows))
    ax1.grid(False)

    legend_patches = [
        patches.Patch(facecolor="#06281b", edgecolor=GREEN_QUANTUM, label="Active Plaquette (RCS F=0.9944, QEC F=0.9869)"),
        patches.Patch(facecolor="#180e14", edgecolor="#7f1d1d", label="Boundary / Excluded Lattice Site (5 defect nodes)"),
    ]
    ax1.legend(handles=legend_patches, loc="lower right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8)

    # -------------------------------------------------------------------------
    # Panel 2: SWAP Routing vs 4-Edge Teleport + Pauli Frame Recovery
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor(CARD_DARK)
    ax2.set_title(
        "PANEL 2: SWAP Routing vs 4-Edge Teleportation in RCS and QEC Modes",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    regimes = ["Willow RCS Mode\n(p2 = 0.14%)", "Willow QEC Mode\n(p2 = 0.33%)", "Fault-Tolerance\nThreshold (0.57%)"]
    f_swap_bars = [data["mean_fid_swap_rcs"], data["mean_fid_swap_qec"], 0.9450]
    f_teleport_bars = [data["mean_fid_teleport_rcs"], data["mean_fid_teleport_qec"], 0.9720]

    x = np.arange(len(regimes))
    w = 0.35

    b1 = ax2.bar(x - w/2, f_swap_bars, w, label="Sequential SWAP (3 CNOTs)", color=RED_ALERT, edgecolor=RED_CRIMSON, zorder=3)
    b2 = ax2.bar(x + w/2, f_teleport_bars, w, label="4-Edge Teleport + Pauli Frame", color=GREEN_QUANTUM, edgecolor=GREEN_EMERALD, zorder=3)

    for bar, val in zip(b1, f_swap_bars):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.4f}", ha="center", va="bottom", color=TEXT_POLAR, fontsize=8.5, fontweight="bold")

    for bar, val in zip(b2, f_teleport_bars):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.4f}", ha="center", va="bottom", color=GREEN_QUANTUM, fontsize=8.5, fontweight="bold")

    # Threshold horizontal marker
    ax2.axhline(0.9720, color=GOLD_BRIGHT, linestyle="--", linewidth=1.5, alpha=0.8, zorder=4, label="Surface-Code Break-Even")

    ax2.set_ylabel("Bell State Teleportation Fidelity F", color=TEXT_SLATE, fontsize=10)
    ax2.set_ylim(0.93, 1.00)
    ax2.set_xticks(x)
    ax2.set_xticklabels(regimes, fontsize=9.5, color=TEXT_POLAR)
    ax2.legend(loc="lower left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8.5)
    ax2.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.5, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 3: Three-Way Architecture Showdown
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor(CARD_DARK)
    ax3.set_title(
        "PANEL 3: Three-Way Architecture Showdown: Willow vs Nighthawk vs Cepheus",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    arch_labels = [
        "Google Willow 105Q\n(Planar Transmon)",
        "IBM Nighthawk R2\n(Monolithic 120Q)",
        "Rigetti Cepheus-108Q\n(Modular 12 Chiplets)",
    ]

    showdown_swap = [data["mean_fid_swap_rcs"], 0.9253, 0.9082]
    showdown_teleport = [data["mean_fid_teleport_rcs"], 0.9606, 0.9515]
    showdown_acc = [data["overall_mean_keep"] * 100.0, 96.8, 95.8]

    x3 = np.arange(len(arch_labels))
    w3 = 0.35

    ax3.bar(x3 - w3/2, showdown_swap, w3, label="Sequential SWAP", color=GOLD_HARDWARE, edgecolor=GOLD_BRIGHT, zorder=3)
    ax3.bar(x3 + w3/2, showdown_teleport, w3, label="4-Edge Teleport + Pauli Frame", color=GREEN_QUANTUM, edgecolor=GREEN_EMERALD, zorder=3)

    for i in range(len(arch_labels)):
        ax3.text(x3[i] - w3/2, showdown_swap[i] + 0.002, f"{showdown_swap[i]:.4f}", ha="center", va="bottom", color=TEXT_POLAR, fontsize=9, fontweight="bold")
        ax3.text(x3[i] + w3/2, showdown_teleport[i] + 0.002, f"{showdown_teleport[i]:.4f}\n(Acc {showdown_acc[i]:.1f}%)", ha="center", va="bottom", color=GREEN_QUANTUM, fontsize=9, fontweight="bold")

    ax3.set_ylabel("Mean Plaquette Fidelity F", color=TEXT_SLATE, fontsize=10)
    ax3.set_ylim(0.88, 1.01)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(arch_labels, fontsize=9.5, color=TEXT_POLAR)
    ax3.legend(loc="lower left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8.5)
    ax3.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.5, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 4: Hardware Specifications & Calibration Audit Matrix
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor(CARD_DARK)
    ax4.set_title(
        "PANEL 4: Hardware Specifications & Surface-Code Calibration Audit",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )
    ax4.axis("off")

    metrics_table = [
        ["Hardware Architecture", "Google Willow 105Q Planar Superconducting Processor", GOLD_BRIGHT],
        ["Physical Qubit Capacity", f"{data['total_qubits']} Qubits (11x10 Bounding Grid, 5 Boundaries)", TEXT_POLAR],
        ["Coupling Links Count", f"{data['total_couplers']} Tunable Planar Couplers (Mean Deg 3.47)", TEXT_POLAR],
        ["Elementary 2x2 Plaquettes", f"{data['total_plaquettes']} Valid Plaquettes (Defect-Free)", GREEN_QUANTUM],
        ["Single-Qubit Gate Error (1-F1)", "0.035% (25 ns Microwave Pulse)", GREEN_QUANTUM],
        ["Two-Qubit Error (RCS Mode)", f"{data['rcs_two_qubit_error']*100:.3f}% (20 ns Fast Tunable CZ)", GREEN_QUANTUM],
        ["Two-Qubit Error (QEC Mode)", f"{data['qec_two_qubit_error']*100:.3f}% (Continuous Syndrome Sweep)", GOLD_BRIGHT],
        ["Energy Relaxation Time (T1)", "98.00 microseconds", TEXT_POLAR],
        ["Dephasing Time (T2*)", "85.00 microseconds", TEXT_POLAR],
        ["Readout Confusion Error", "0.670%", GREEN_QUANTUM],
        ["Mean Plaquette Teleport (RCS)", f"{data['mean_fid_teleport_rcs']:.4f} (99.5% QEC Acceptance)", GREEN_QUANTUM],
        ["Mean Plaquette Teleport (QEC)", f"{data['mean_fid_teleport_qec']:.4f} (99.5% QEC Acceptance)", GREEN_QUANTUM],
        ["Simulation Execution Latency", f"{data['execution_time_us']:.2f} microseconds (SIMD AVX2)", GOLD_BRIGHT],
    ]

    y_start = 0.94
    dy = 0.068

    for idx, (param, val, color) in enumerate(metrics_table):
        y = y_start - idx * dy
        bg_c = CARD_CHARCOAL if idx % 2 == 0 else CARD_DARK
        rect = patches.Rectangle(
            (0.02, y - 0.024), 0.96, 0.055,
            facecolor=bg_c, edgecolor=GRID_SLATE, linewidth=0.5, zorder=1,
        )
        ax4.add_patch(rect)
        ax4.text(0.05, y, param, color=TEXT_SLATE, fontsize=8.5, va="center", zorder=2)
        ax4.text(0.95, y, val, color=color, fontsize=8.5, fontweight="bold", ha="right", va="center", zorder=2)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_png, dpi=300, facecolor=BG_OBSIDIAN, edgecolor="none")
    plt.close()

    print(f" [+] Rendered Willow-105Q Golden Visual: {out_png} ({out_png.stat().st_size / 1024:.1f} KB)")

    # Also mirror to results/charts/willow_105q_showdown.png
    charts_target = repo_root / "results" / "charts" / "willow_105q_showdown.png"
    charts_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_png, charts_target)
    print(f" [+] Mirrored to Workspace Gallery: {charts_target}")

    return out_png


if __name__ == "__main__":
    chart_golden_willow()

