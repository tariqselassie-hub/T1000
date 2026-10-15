# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes T-1000 10-to-26 Qubit Statevector Scaling & AVX2+FMA SIMD Benchmarks.
Adheres strictly to the Deen OS Sovereign Palette.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deen_theme import (
    apply_deen_theme, style_card, save_deen_figure,
    BG_OBSIDIAN, GREEN_QUANTUM, GREEN_EMERALD, GOLD_HARDWARE, GOLD_BRIGHT,
    RED_ALERT, TEXT_POLAR, TEXT_MUTED, GRID_SLATE
)

def chart_qnn_scaling(csv_path: Path, output_dir: Path) -> Path:
    apply_deen_theme()
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 SIMD ENGINE // 10-TO-26 QUBIT HILBERT SPACE SCALING TELEMETRY",
        color=TEXT_POLAR, fontsize=13, fontweight="bold", y=0.98
    )

    qubits = df["qubits"]

    # 1. Latency Scaling (Log Scale)
    ax1 = axes[0]
    ax1.semilogy(qubits, df["total_latency_ms"], color=GREEN_QUANTUM, lw=2.4, marker="s", markersize=6, label="Total Circuit Latency (ms)")
    ax1.semilogy(qubits, df["per_layer_ms"], color=GOLD_HARDWARE, lw=2.0, ls="--", marker="o", markersize=5, label="Per-Layer Butterfly (ms)")
    ax1.set_xlabel("Register Size (Qubits)", color=TEXT_MUTED)
    ax1.set_ylabel("Execution Latency (ms, log scale)", color=TEXT_MUTED)
    ax1.set_xticks(qubits)
    style_card(ax1, "PANEL 1: SIMD LATENCY SCALING", "AVX2+FMA Tiled Butterfly Execution")
    ax1.legend(loc="upper left")

    # 2. RAM Footprint & Cache Saturation
    ax2 = axes[1]
    # Convert RAM MB to logarithmic bar chart
    ram_mb = df["ram_mb"]
    bars = ax2.bar(qubits, ram_mb, color=GOLD_HARDWARE, width=1.1, alpha=0.85, edgecolor=GRID_SLATE, label="Statevector Buffer (MB)")
    ax2.set_yscale("log")
    ax2.set_xlabel("Register Size (Qubits)", color=TEXT_MUTED)
    ax2.set_ylabel("Memory Footprint (MB, log scale)", color=TEXT_MUTED)
    ax2.set_xticks(qubits)
    
    # Annotate hardware cache boundaries
    ax2.axhline(0.032, color=GREEN_QUANTUM, ls=":", lw=1.5, label="L1D Cache (32 KB / 11q)")
    ax2.axhline(36.0, color=GOLD_BRIGHT, ls="--", lw=1.5, label="L2 Tile Cache (36 MB)")
    ax2.axhline(1024.0, color=RED_ALERT, ls="-.", lw=1.5, label="26-Qubit Allocation (1 GB)")
    style_card(ax2, "PANEL 2: HARDWARE CACHE HIERARCHY", "Memory Footprint vs CPU Cache Architecture")
    ax2.legend(loc="upper left")

    # 3. Cross-Entropy Benchmarking (XEB) Chaos Invariance
    ax3 = axes[2]
    ax3.plot(qubits, df["f_xeb_with_pi"], color=GREEN_EMERALD, lw=2.2, marker="D", markersize=5, label="F_XEB (Rademacher-π)")
    ax3.plot(qubits, df["f_xeb_no_pi"], color=GOLD_HARDWARE, lw=1.8, ls="--", marker="^", markersize=5, label="F_XEB (Zero-Phase)")
    ax3.plot(qubits, df["f_xeb_cross_nisq"], color=RED_ALERT, lw=1.8, ls=":", marker="x", markersize=5, label="F_XEB (NISQ 5% Depol)")
    ax3.axhline(1.0, color=TEXT_POLAR, ls="-", lw=1.0, alpha=0.5, label="Ideal Chaos (F_XEB = 1.000)")
    ax3.set_xlabel("Register Size (Qubits)", color=TEXT_MUTED)
    ax3.set_ylabel("Linear XEB Fidelity", color=TEXT_MUTED)
    ax3.set_xticks(qubits)
    ax3.set_ylim([0.90, 1.05])
    style_card(ax3, "PANEL 3: PORTER-THOMAS CHAOS INVARIANCE", "Linear Cross-Entropy Benchmarking vs NISQ Depolarization")
    ax3.legend(loc="lower left")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

    png_path = output_dir / "qnn_scaling_26_qubits.png"
    save_deen_figure(fig, png_path)
    return png_path

if __name__ == "__main__":
    ws_root = Path(__file__).resolve().parents[2]
    csv_file = ws_root / "results" / "experiments" / "qnn_scaling_26_qubits.csv"
    out_dir = ws_root / "results" / "charts"
    if csv_file.exists():
        png = chart_qnn_scaling(csv_file, out_dir)
        print(f" [+] Generated QNN scaling chart: {png}")
    else:
        print(f" [-] Missing data file: {csv_file}")
