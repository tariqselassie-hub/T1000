# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Quantinuum System Model H2-1 Golden Benchmark Visualizer.
Renders 4-panel sovereign publication diagram adhering to Deen OS Visual Specification:
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


def chart_golden_quantinuum_h2():
    cur_dir = Path(__file__).resolve().parent
    repo_root = cur_dir.parents[1]
    json_path = cur_dir / "quantinuum_h2_telemetry.json"
    out_png = cur_dir / "quantinuum_h2_golden_showdown.png"

    if not json_path.exists():
        fallback_json = repo_root / "results" / "data" / "quantinuum_h2_telemetry.json"
        if fallback_json.exists():
            shutil.copy(fallback_json, json_path)
        else:
            print(f"[!] Telemetry file missing: {json_path}")
            return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sweep = data["distance_sweep"]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 GOLDEN EXPERIMENT: QUANTINUUM H2-1 ALL-TO-ALL TRAPPED-ION BENCHMARK",
        color=TEXT_POLAR,
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # -------------------------------------------------------------------------
    # Panel 1: QCCD Linear Racetrack & 4 Parallel Gate Zones
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor(CARD_DARK)
    ax1.set_title(
        "PANEL 1: 56-Qubit QCCD Linear Racetrack Trap & 4 Interaction Zones",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    # Draw Racetrack Surface Electrodes
    track_outer = patches.FancyBboxPatch(
        (0.8, 1.2), 8.4, 2.6,
        boxstyle="round,pad=0.2,rounding_size=1.0",
        facecolor="#08101d",
        edgecolor=GRID_SLATE,
        linewidth=2.5,
        zorder=1,
    )
    track_inner = patches.FancyBboxPatch(
        (2.0, 1.9), 6.0, 1.2,
        boxstyle="round,pad=0.1,rounding_size=0.5",
        facecolor=CARD_DARK,
        edgecolor=GRID_SLATE,
        linewidth=2.0,
        zorder=2,
    )
    ax1.add_patch(track_outer)
    ax1.add_patch(track_inner)

    # 4 Gate Zones
    zone_coords = [
        (2.5, 3.4, "GATE ZONE 1"),
        (6.0, 3.4, "GATE ZONE 2"),
        (2.5, 1.1, "GATE ZONE 3"),
        (6.0, 1.1, "GATE ZONE 4"),
    ]
    for zx, zy, zlabel in zone_coords:
        zbox = patches.Rectangle(
            (zx - 0.7, zy - 0.35), 1.4, 0.7,
            facecolor="#2d2305",
            edgecolor=GOLD_BRIGHT,
            linewidth=1.8,
            linestyle="--",
            zorder=3,
        )
        ax1.add_patch(zbox)
        ax1.text(
            zx, zy, zlabel,
            color=GOLD_BRIGHT, fontsize=7.5, fontweight="bold",
            ha="center", va="center", zorder=4,
        )

    # Draw 56 Ytterbium-171 Ions along the track
    top_x = np.linspace(1.5, 8.5, 24)
    bot_x = np.linspace(8.5, 1.5, 24)
    curve_left_angles = np.linspace(np.pi / 2, 3 * np.pi / 2, 4)
    curve_right_angles = np.linspace(-np.pi / 2, np.pi / 2, 4)

    ion_x = []
    ion_y = []
    for x in top_x:
        ion_x.append(x)
        ion_y.append(3.4)
    for ang in curve_right_angles:
        ion_x.append(8.5 + 0.7 * np.cos(ang))
        ion_y.append(2.5 + 0.9 * np.sin(ang))
    for x in bot_x:
        ion_x.append(x)
        ion_y.append(1.6)
    for ang in curve_left_angles:
        ion_x.append(1.5 + 0.7 * np.cos(ang))
        ion_y.append(2.5 + 0.9 * np.sin(ang))

    ax1.scatter(
        ion_x[:56], ion_y[:56],
        s=75, color=GREEN_QUANTUM, edgecolors=GREEN_EMERALD,
        linewidths=1.2, zorder=5, label="171Yb+ Data Qubit",
    )

    # Shuttling connectivity lines
    ax1.plot([2.5, 6.0], [3.4, 1.6], color=GOLD_HARDWARE, linestyle=":", linewidth=1.2, alpha=0.7, zorder=3)
    ax1.plot([1.5, 7.5], [3.4, 3.4], color=GREEN_QUANTUM, linestyle=":", linewidth=1.0, alpha=0.5, zorder=3)
    ax1.text(5.0, 2.5, "ALL-TO-ALL DYNAMIC SHUTTLING\n(K56 Complete Connectivity)", color=TEXT_POLAR, fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=4)

    ax1.set_xlim(0.0, 10.0)
    ax1.set_ylim(0.0, 5.0)
    ax1.axis("off")
    ax1.legend(loc="lower center", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8.5)

    # -------------------------------------------------------------------------
    # Panel 2: Entanglement Fidelity vs Separation Distance (Hops)
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor(CARD_DARK)
    ax2.set_title(
        "PANEL 2: Distance-Dependent Bell Fidelity: Trapped-Ion vs Planar Grids",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    d_hops = [pt["distance_hops"] for pt in sweep]
    f_h2 = [pt["quantinuum_h2_fidelity"] for pt in sweep]
    f_willow = [pt["willow_105q_fidelity"] for pt in sweep]
    f_nighthawk = [pt["nighthawk_r2_fidelity"] for pt in sweep]
    f_cepheus = [pt["cepheus_108q_fidelity"] for pt in sweep]

    ax2.plot(d_hops, f_h2, color=GREEN_QUANTUM, linewidth=3.0, marker="o", markersize=6, label="Quantinuum H2-1 (All-to-All, Zero SWAPs)", zorder=5)
    ax2.plot(d_hops, f_willow, color=GOLD_BRIGHT, linewidth=2.0, linestyle="--", marker="s", markersize=5, label="Google Willow 105Q (Planar Grid, p2=0.14%)", zorder=4)
    ax2.plot(d_hops, f_nighthawk, color=TEXT_SLATE, linewidth=2.0, linestyle="-.", marker="^", markersize=5, label="IBM Nighthawk R2 (Heavy-Hex, p2=0.62%)", zorder=3)
    ax2.plot(d_hops, f_cepheus, color=RED_ALERT, linewidth=2.0, linestyle=":", marker="d", markersize=5, label="Rigetti Cepheus-108Q (Modular Seam, p2=0.8-2.2%)", zorder=2)

    ax2.set_xlabel("Graph Routing Distance Between Target Qubits (Hops)", color=TEXT_SLATE, fontsize=10)
    ax2.set_ylabel("Entanglement Bell Fidelity F", color=TEXT_SLATE, fontsize=10)
    ax2.set_ylim(0.55, 1.01)
    ax2.set_xticks(d_hops)
    ax2.legend(loc="lower left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8.5)
    ax2.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.5, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 3: Two-Qubit Operation Accumulation Overhead
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor(CARD_DARK)
    ax3.set_title(
        "PANEL 3: Two-Qubit Pulse Burden Across Distance (SWAP Network Scaling)",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )

    sample_distances = [1, 3, 5, 10, 15, 20]
    x3 = np.arange(len(sample_distances))
    w3 = 0.38

    h2_ops = [1 for _ in sample_distances] # Always 1 direct MS gate
    planar_ops = [1 + 3 * (d - 1) for d in sample_distances]

    b_h2 = ax3.bar(x3 - w3/2, h2_ops, w3, label="Quantinuum H2 (QCCD Shuttled)", color=GREEN_QUANTUM, edgecolor=GREEN_EMERALD, zorder=3)
    b_planar = ax3.bar(x3 + w3/2, planar_ops, w3, label="Planar Grid Transmon (SWAP Routed)", color=RED_ALERT, edgecolor=RED_CRIMSON, zorder=3)

    for bar, val in zip(b_h2, h2_ops):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1.0, f"{val}", ha="center", va="bottom", color=GREEN_QUANTUM, fontsize=8.5, fontweight="bold")

    for bar, val in zip(b_planar, planar_ops):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 1.0, f"{val}", ha="center", va="bottom", color=RED_ALERT, fontsize=8.5, fontweight="bold")

    ax3.set_xlabel("Routing Distance (Hops)", color=TEXT_SLATE, fontsize=10)
    ax3.set_ylabel("Total 2-Qubit Physical Operations Required", color=TEXT_SLATE, fontsize=10)
    ax3.set_ylim(0, 68)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([f"d={d}" for d in sample_distances], fontsize=9.5, color=TEXT_POLAR)
    ax3.legend(loc="upper left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8.5)
    ax3.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.5, zorder=0)

    # -------------------------------------------------------------------------
    # Panel 4: Hardware Specifications & Modality Audit Matrix
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor(CARD_DARK)
    ax4.set_title(
        "PANEL 4: Hardware Specifications & Trapped-Ion Physical Calibration",
        color=GOLD_BRIGHT,
        fontsize=11,
        pad=10,
    )
    ax4.axis("off")

    metrics_table = [
        ["Hardware Architecture", "Quantinuum System Model H2-1 Trapped-Ion QPU", GOLD_BRIGHT],
        ["Qubit Physical Species", "171Yb+ Hyperfine Clock States (138Ba+ Sympathetic)", TEXT_POLAR],
        ["Physical Qubit Capacity", f"{data['qubit_capacity']} Qubits (All-to-All QCCD Racetrack)", GREEN_QUANTUM],
        ["Dynamic Coupler Pairs", f"{data['total_couplers']} All-to-All Physical Couplers (K56)", GREEN_QUANTUM],
        ["Parallel Interaction Zones", f"{data['interaction_zones']} Concurrent Two-Qubit Laser Zones", GOLD_BRIGHT],
        ["Single-Qubit Gate Infidelity", f"{data['single_qubit_error']*100:.4f}% (10 us Laser Pulse)", GREEN_QUANTUM],
        ["Two-Qubit Mølmer-Sørensen Error", f"{data['two_qubit_ms_error']*100:.3f}% (150 us Entangling Gate)", GREEN_QUANTUM],
        ["SPAM Readout Confusion Error", f"{data['readout_error']*100:.3f}%", GREEN_QUANTUM],
        ["Relaxation Lifetime (T1)", f"{data['t1_relaxation_us']/1e6:.1f} Seconds (> 1.5 Minutes)", TEXT_POLAR],
        ["Dephasing Lifetime (T2*)", f"{data['t2_dephasing_us']/1e6:.1f} Seconds", TEXT_POLAR],
        ["Routing SWAPs at 20 Hops", "0 SWAPs (F = 0.9964 vs 0.9149 on Willow Transmon)", GREEN_QUANTUM],
        ["Effective Fidelity Retention", "100.0% Flat Across Arbitrary Graph Distances", GREEN_QUANTUM],
        ["Simulation Latency (AVX2)", f"{data['execution_time_us']:.2f} microseconds (SIMD Pure-Rust)", GOLD_BRIGHT],
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

    print(f" [+] Rendered Quantinuum H2-1 Golden Visual: {out_png} ({out_png.stat().st_size / 1024:.1f} KB)")

    # Mirror to results/charts/ and assets/
    charts_target = repo_root / "results" / "charts" / "quantinuum_h2_showdown.png"
    charts_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_png, charts_target)
    print(f" [+] Mirrored to Workspace Gallery: {charts_target}")

    assets_target = repo_root / "assets" / "quantinuum_h2_golden_showdown.png"
    if assets_target.parent.exists():
        shutil.copy(out_png, assets_target)
        print(f" [+] Mirrored to Assets: {assets_target}")

    return out_png


if __name__ == "__main__":
    chart_golden_quantinuum_h2()

