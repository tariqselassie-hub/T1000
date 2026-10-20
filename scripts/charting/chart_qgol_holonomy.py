# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Deen OS Sovereign Telemetry Visualizer: Quantum Conway's Game of Life (Q-GoL).
Renders 4-panel empirical suite:
  1. Multi-Scale AVX2 SIMD Throughput & Latency Scaling (16x16 -> 64x64).
  2. Mach-Zehnder Glider Soliton S-Matrix Interferometry & Entanglement.
  3. Quantum Vortex Dipole Vorticity & Probability Flux Dynamics.
  4. Floquet Kicked Chaos & OTOC Operator Scrambling.
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Inject current workspace scripts/charting to path
sys.path.insert(0, str(Path(__file__).parent))
from deen_theme import (
    apply_deen_theme,
    BG_OBSIDIAN,
    CARD_CHARCOAL,
    GREEN_QUANTUM,
    GREEN_EMERALD,
    GOLD_HARDWARE,
    GOLD_BRIGHT,
    RED_ALERT,
    TEXT_POLAR,
    TEXT_MUTED,
    GRID_SLATE,
)

TELEMETRY_JSON = Path("results/experiments/qgol_simulation_telemetry.json")
OUTPUT_PLOT = Path("results/plots/qgol_holonomy_and_solitons.png")


def main():
    if not TELEMETRY_JSON.exists():
        print(f"Error: {TELEMETRY_JSON} does not exist.")
        sys.exit(1)

    with open(TELEMETRY_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    apply_deen_theme()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=BG_OBSIDIAN)
    (ax1, ax2), (ax3, ax4) = axes

    # -------------------------------------------------------------------------
    # Panel 1: Multi-Scale AVX2 SIMD Scaling
    # -------------------------------------------------------------------------
    bench = data["multi_scale_benchmark"]
    dims = [b["grid_dimension"] for b in bench]
    tputs = [b["throughput_cells_per_sec"] / 1000.0 for b in bench]
    lats = [b["latency_us_per_step"] for b in bench]
    errs = [b["norm_conservation_error"] for b in bench]

    x = np.arange(len(dims))
    width = 0.38

    rects1 = ax1.bar(
        x - width / 2,
        tputs,
        width,
        label="Throughput (kCells/s)",
        color=GREEN_QUANTUM,
        edgecolor=BG_OBSIDIAN,
        alpha=0.9,
    )
    ax1.set_ylabel("Throughput (kCells / sec)", color=GREEN_QUANTUM, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(tputs) * 1.25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dims, color=TEXT_POLAR, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=GREEN_QUANTUM)
    ax1.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.set_yscale("log")
    rects2 = ax1_twin.plot(
        x + width / 2,
        lats,
        color=GOLD_HARDWARE,
        marker="s",
        linewidth=2.2,
        markersize=7,
        label="Latency (us/step)",
    )
    ax1_twin.set_ylabel("Latency per Step (us, log scale)", color=GOLD_HARDWARE, fontsize=11, fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor=GOLD_HARDWARE)

    # Value callouts on throughput
    for i, (tp, lat, err) in enumerate(zip(tputs, lats, errs)):
        ax1.text(
            i - width / 2,
            tp + 25.0,
            f"{tp:.1f}k\n({err:.1e})",
            ha="center",
            va="bottom",
            color=TEXT_POLAR,
            fontsize=9,
            fontweight="bold",
        )

    ax1.set_title(
        "[1] AVX2 SIMD Multi-Scale Throughput & Latency Scaling",
        color=TEXT_POLAR,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # -------------------------------------------------------------------------
    # Panel 2: S-Matrix Interferometry & Entanglement Entropy
    # -------------------------------------------------------------------------
    sweep = data["interferometry_sweep"]["sweep_results"]
    phases = [pt["phase_deg"] for pt in sweep]
    living_pops = [pt["living_population"] for pt in sweep]
    ffs_pops = [pt["ffs_population"] for pt in sweep]
    entang = [pt["peak_entanglement"] for pt in sweep]

    ax2.plot(
        phases,
        living_pops,
        color=GREEN_QUANTUM,
        linewidth=2.2,
        marker="o",
        label="Living Population (|1>)",
    )
    ax2.plot(
        phases,
        ffs_pops,
        color=GOLD_HARDWARE,
        linewidth=2.2,
        marker="^",
        linestyle="--",
        label="FFS Population (|2>)",
    )
    ax2.set_xlabel("Relative Collision Phase Delta theta (deg)", color=TEXT_POLAR, fontsize=11)
    ax2.set_ylabel("Sector Population", color=TEXT_POLAR, fontsize=11)
    ax2.tick_params(colors=TEXT_MUTED)
    ax2.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        phases,
        entang,
        color=RED_ALERT,
        linewidth=2.4,
        linestyle="-.",
        label="Bipartite Entropy S (nats)",
    )
    ax2_twin.axhline(
        np.log(2.0),
        color=TEXT_MUTED,
        linestyle=":",
        linewidth=1.2,
        label="1 ebit (ln 2 = 0.6931 nats)",
    )
    ax2_twin.set_ylabel("Entanglement Entropy S (nats)", color=RED_ALERT, fontsize=11, fontweight="bold")
    ax2_twin.tick_params(axis="y", labelcolor=RED_ALERT)

    lines_2, labels_2 = ax2.get_legend_handles_labels()
    lines_2t, labels_2t = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines_2 + lines_2t, labels_2 + labels_2t, loc="lower left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=9)

    ax2.set_title(
        "[2] Mach-Zehnder Glider Soliton S-Matrix Interferometry",
        color=TEXT_POLAR,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # -------------------------------------------------------------------------
    # Panel 3: Quantum Vortex Dipole & Circulation Flux Dynamics
    # -------------------------------------------------------------------------
    vortex_scen = next(s for s in data["scenarios"] if s["scenario"] == "Quantum Vortex Dipole")
    steps_v = [h["step"] for h in vortex_scen["history"]]
    vorticities = [h["max_vorticity"] for h in vortex_scen["history"]]
    vortex_counts = [h["vortex_count"] for h in vortex_scen["history"]]
    currents = [h["mean_current_magnitude"] for h in vortex_scen["history"]]

    ax3.plot(
        steps_v,
        vorticities,
        color=GREEN_EMERALD,
        linewidth=2.4,
        marker="o",
        label="Max Vorticity |omega|",
    )
    ax3.plot(
        steps_v,
        currents,
        color=GOLD_BRIGHT,
        linewidth=2.0,
        marker="s",
        label="Mean Probability Current |j|",
    )
    ax3.set_xlabel("Time Step (Tick)", color=TEXT_POLAR, fontsize=11)
    ax3.set_ylabel("Vorticity & Current Field Magnitude", color=TEXT_POLAR, fontsize=11)
    ax3.tick_params(colors=TEXT_MUTED)
    ax3.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)

    ax3_twin = ax3.twinx()
    ax3_twin.bar(
        steps_v,
        vortex_counts,
        width=0.45,
        color=RED_ALERT,
        alpha=0.35,
        edgecolor=RED_ALERT,
        label="Vortex Cores (|omega| > 0.15)",
    )
    ax3_twin.set_ylabel("Active Vortex Core Count", color=RED_ALERT, fontsize=11, fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor=RED_ALERT)

    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_3t, labels_3t = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_3t, labels_3 + labels_3t, loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=9)

    ax3.set_title(
        "[3] Gross-Pitaevskii Quantum Vortex Dipole Circulation",
        color=TEXT_POLAR,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # -------------------------------------------------------------------------
    # Panel 4: Floquet Kicked Quantum Chaos & OTOC Scrambling
    # -------------------------------------------------------------------------
    chaos_scen = next(s for s in data["scenarios"] if s["scenario"] == "Floquet Kicked Quantum Chaos")
    steps_c = [h["step"] for h in chaos_scen["history"]]
    otoc = [h["scrambling_otoc"] for h in chaos_scen["history"]]
    entropy_c = [h["entanglement_entropy"] for h in chaos_scen["history"]]
    mom_density = [h["mean_momentum_density"] for h in chaos_scen["history"]]

    ax4.plot(
        steps_c,
        otoc,
        color=RED_ALERT,
        linewidth=2.5,
        marker="d",
        label="OTOC Scrambling Proxy C(t)",
    )
    ax4.plot(
        steps_c,
        entropy_c,
        color=GREEN_QUANTUM,
        linewidth=2.0,
        marker="o",
        label="Bipartite Entropy S(t) (nats)",
    )
    ax4.plot(
        steps_c,
        mom_density,
        color=GOLD_HARDWARE,
        linewidth=1.8,
        linestyle="--",
        marker="v",
        label="Mean Momentum Density n(k)",
    )
    ax4.set_xlabel("Floquet Kick Step (Period tau)", color=TEXT_POLAR, fontsize=11)
    ax4.set_ylabel("Chaos & Scrambling Observables", color=TEXT_POLAR, fontsize=11)
    ax4.tick_params(colors=TEXT_MUTED)
    ax4.grid(True, linestyle=":", color=GRID_SLATE, alpha=0.6)
    ax4.legend(loc="upper left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, fontsize=9)

    ax4.set_title(
        "[4] Floquet Kicked Quantum Chaos & Scrambling OTOC (K = 2.40)",
        color=TEXT_POLAR,
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Figure Header
    fig.suptitle(
        "T-1000 Sovereign Quantum Conway Cellular Automaton Telemetry (Monograph M153)\n"
        "AVX2+FMA SIMD Splitting | Confinement-Induced Resonances | Fractional Fermi Seas (alpha = 2)",
        color=TEXT_POLAR,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT, dpi=200, facecolor=BG_OBSIDIAN, edgecolor="none")
    plt.close()
    print(f"[+] Sovereign chart successfully saved to: {OUTPUT_PLOT.resolve()}")


if __name__ == "__main__":
    main()
