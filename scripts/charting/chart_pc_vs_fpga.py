# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Visualizes T-1000 PC (AVX2+FMA SIMD) vs FPGA (Artix-7 XC7A35T) 26-Qubit Scaling.
Analyzes execution latency, hardware acceleration speedup, and numerical fidelity (f64 vs Q1.15).
Adheres strictly to the Deen OS Sovereign Visual Specification.
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

def chart_pc_vs_fpga(csv_path: Path, output_dir: Path) -> Path:
    apply_deen_theme()
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), facecolor=BG_OBSIDIAN)
    fig.suptitle(
        "T-1000 BENCHMARK // PC (INTEL CORE ULTRA 7 AVX2) VS FPGA (ARTIX-7 XC7A35T) SCALING",
        color=TEXT_POLAR, fontsize=12, fontweight="bold", y=0.98
    )

    qubits = df["qubits"]

    # -------------------------------------------------------------------------
    # Panel 1: Latency Scaling (PC AVX2 vs FPGA Pure Silicon vs FPGA USB-UART)
    # -------------------------------------------------------------------------
    ax1 = axes[0]
    ax1.semilogy(
        qubits, df["pc_per_layer_ms"],
        color=GREEN_QUANTUM, lw=2.4, marker="s", markersize=6,
        label="PC AVX2+FMA CPU (ms/layer)"
    )
    ax1.semilogy(
        qubits, df["fpga_silicon_per_layer_ms"],
        color=GOLD_HARDWARE, lw=2.2, marker="o", markersize=6,
        label="FPGA Pure Silicon (100 MHz, 8 Lanes)"
    )
    ax1.semilogy(
        qubits, df["fpga_uart_per_layer_ms"],
        color=GOLD_BRIGHT, lw=1.8, ls="--", marker="^", markersize=5,
        label="FPGA USB-UART Bridge (COM6 @ 3.0 MBaud)"
    )

    # Highlight on-chip physical BRAM boundary
    ax1.axvline(14, color=RED_ALERT, ls=":", lw=1.5, label="Artix-7 On-Chip BRAM (14Q)")

    ax1.set_xlabel("Register Size (Qubits)", color=TEXT_MUTED)
    ax1.set_ylabel("Per-Layer Execution Latency (ms, log scale)", color=TEXT_MUTED)
    ax1.set_xticks(qubits)
    style_card(ax1, "PANEL 1: EXECUTION LATENCY SCALING", "Per-Layer Butterfly: PC AVX2 vs FPGA Silicon")
    ax1.legend(loc="upper left")

    # -------------------------------------------------------------------------
    # Panel 2: Silicon Acceleration Speedup Ratio (PC / FPGA Silicon)
    # -------------------------------------------------------------------------
    ax2 = axes[1]
    speedup = df["silicon_speedup_ratio"]
    bars = ax2.bar(
        qubits, speedup,
        color=GOLD_HARDWARE, width=1.1, alpha=0.88, edgecolor=GRID_SLATE,
        label="Pure Silicon Speedup (x)"
    )
    
    # Annotate speedup values on top of bars
    for q, s in zip(qubits, speedup):
        ax2.annotate(
            f"{s:.1f}x",
            xy=(q, s),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color=TEXT_POLAR,
            fontsize=8,
            fontweight="bold"
        )

    ax2.set_xlabel("Register Size (Qubits)", color=TEXT_MUTED)
    ax2.set_ylabel("Speedup Multiplier (T_PC / T_FPGA_Silicon)", color=TEXT_MUTED)
    ax2.set_xticks(qubits)
    ax2.set_ylim([0, max(speedup) * 1.15])
    style_card(ax2, "PANEL 2: SILICON COMPUTE ACCELERATION", "DSP48E1 Pipeline Concurrency vs Host CPU")
    ax2.legend(loc="upper right")

    # -------------------------------------------------------------------------
    # Panel 3: Numerical Fidelity & Chaos Invariance (f64 vs Q1.15 Fixed-Point)
    # -------------------------------------------------------------------------
    ax3 = axes[2]
    ax3.plot(
        qubits, df["pc_f_xeb"],
        color=GREEN_EMERALD, lw=2.2, marker="D", markersize=5,
        label="PC F_XEB (Double Precision f64)"
    )
    ax3.plot(
        qubits, df["fpga_f_xeb"],
        color=GOLD_HARDWARE, lw=2.0, ls="--", marker="o", markersize=5,
        label="FPGA F_XEB (Fixed-Point Q1.15)"
    )
    ax3.plot(
        qubits, df["quantum_state_overlap"],
        color=GREEN_QUANTUM, lw=1.8, ls="-.", marker="x", markersize=5,
        label="State Overlap |<PC|FPGA>|^2"
    )

    ax3.axhline(1.0, color=TEXT_POLAR, ls="-", lw=1.0, alpha=0.5, label="Haar Invariance (F_XEB = 1.000)")
    ax3.axvline(14, color=RED_ALERT, ls=":", lw=1.5, label="On-Chip 14Q Boundary")

    ax3.set_xlabel("Register Size (Qubits)", color=TEXT_MUTED)
    ax3.set_ylabel("Fidelity Metric", color=TEXT_MUTED)
    ax3.set_xticks(qubits)
    ax3.set_ylim([0.90, 1.02])
    style_card(ax3, "PANEL 3: NUMERICAL FIDELITY INVARIANCE", "IEEE 754 f64 vs Artix-7 DSP Fixed-Point Q1.15")
    ax3.legend(loc="lower left")

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])

    png_path = output_dir / "pc_vs_fpga_26q_scaling.png"
    save_deen_figure(fig, png_path)
    return png_path

if __name__ == "__main__":
    ws_root = Path(__file__).resolve().parents[2]
    csv_file = ws_root / "results" / "experiments" / "pc_vs_fpga_26q_scaling.csv"
    out_dir = ws_root / "results" / "charts"
    if csv_file.exists():
        png = chart_pc_vs_fpga(csv_file, out_dir)
        print(f" [+] Generated PC vs FPGA scaling chart: {png}")
    else:
        print(f" [-] Missing data file: {csv_file}")
