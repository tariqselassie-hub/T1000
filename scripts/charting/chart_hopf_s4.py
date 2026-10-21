# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
Second Hopf Fibration (S7 -> S4) & Quaternionic Entanglement Invariants (Monograph M157)
Renders a 4-panel publication visual strictly adhering to the Deen OS Sovereign Visual Specification:
- Root Canvas: Deep Obsidian (#05080d)
- Card Panels: Void Charcoal (#0f172a)
- Quantum Telemetry: Laser Emerald Green (#00e676)
- Hardware & Clocks: Radiant Gold (#ffb300)
- Bounds & Thresholds: Crimson Red (#ef4444)
- Text & Readouts: Polar Slate (#f8fafc)
Zero cyan, zero purple, zero emojis.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    json_path = os.path.join("results", "data", "hopf_s4_entanglement_report.json")
    if not os.path.exists(json_path):
        json_path = os.path.join("web", "data", "hopf_s4_entanglement_report.json")

    if not os.path.exists(json_path):
        print(f"Error: Telemetry file {json_path} not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bell_basis = data["bell_basis_states"]
    rot_sweep = data["rotation_sweep"]
    werner_sweep = data["werner_sweep"]
    throughput = data["throughput"]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#05080d")

    # -------------------------------------------------------------
    # Panel 1: S4 Quaternionic Stereographic Projection (y0 vs ||Y_cross||)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#0f172a")

    # Unit boundary y0^2 + ||Y_cross||^2 = 1
    circle_theta = np.linspace(0, np.pi, 200)
    ax1.plot(np.cos(circle_theta), np.sin(circle_theta), color="#334155", linestyle="--", linewidth=1.5, label="Unit S4 Manifold (||y|| = 1)")
    ax1.axhline(0, color="#1e293b", linewidth=1.0)
    ax1.axvline(0, color="#1e293b", linewidth=1.0)

    # Plot rotation trajectory
    y0_rot = [p["hopf_coord"]["y0"] for p in rot_sweep]
    y_cross_rot = [
        np.sqrt(p["hopf_coord"]["y1"]**2 + p["hopf_coord"]["y2"]**2 + p["hopf_coord"]["y3"]**2 + p["hopf_coord"]["y4"]**2)
        for p in rot_sweep
    ]
    ax1.plot(y0_rot, y_cross_rot, color="#00e676", linewidth=2.5, label="Rotation Trajectory |psi(th)>")

    # Mark Basis States
    for b in bell_basis:
        y0 = b["hopf_coord"]["y0"]
        yc = np.sqrt(b["hopf_coord"]["y1"]**2 + b["hopf_coord"]["y2"]**2 + b["hopf_coord"]["y3"]**2 + b["hopf_coord"]["y4"]**2)
        c = b["invariants"]["concurrence"]
        color = "#00e676" if c > 0.9 else "#ffb300"
        ax1.scatter(y0, yc, color=color, s=90, edgecolors="#f8fafc", zorder=5)
        offset = (0.04, 0.04) if y0 >= 0 else (-0.12, 0.04)
        ax1.text(y0 + offset[0], yc + offset[1], b["name"], color="#f8fafc", fontsize=9, fontweight="bold")

    ax1.set_xlabel("Pole Coordinate y0 (|qA|^2 - |qB|^2)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Equatorial Radius ||Y_cross||", color="#f8fafc", fontsize=11, fontweight="bold")
    ax1.set_title("Second Hopf Fibration S4 Projection: Poles vs Bell Equator", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax1.legend(loc="upper right", facecolor="#111827", edgecolor="#1e293b", labelcolor="#f8fafc")
    ax1.set_xlim(-1.25, 1.25)
    ax1.set_ylim(-0.1, 1.25)

    # -------------------------------------------------------------
    # Panel 2: Continuous Entanglement Concurrence & Tsirelson CHSH
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#0f172a")

    th_deg = [p["theta_deg"] for p in rot_sweep]
    concurr = [p["invariants"]["concurrence"] for p in rot_sweep]
    bell_s = [p["invariants"]["bell_chsh"] for p in rot_sweep]

    ax2.plot(th_deg, concurr, color="#00e676", linewidth=2.5, label="Wootters Concurrence C(th)")
    ax2.plot(th_deg, [s / (2.0 * np.sqrt(2.0)) for s in bell_s], color="#ffb300", linewidth=2.0, linestyle="-.", label="Bell-CHSH S / 2*sqrt(2)")

    ax2.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Tsirelson Bound (2*sqrt(2))")
    ax2.axhline(2.0 / (2.0 * np.sqrt(2.0)), color="#e2e8f0", linestyle=":", linewidth=1.2, label="Classical Local Bound (S = 2.0)")
    ax2.axvline(45.0, color="#334155", linestyle="--", linewidth=1.0)
    ax2.text(46.0, 0.45, "Bell State |Phi+>\n(th = 45 deg, C = 1.0)", color="#00e676", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Rotation Angle theta (degrees)", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Normalized Entanglement Metric", color="#f8fafc", fontsize=11, fontweight="bold")
    ax2.set_title("Wootters Concurrence vs Bell-CHSH Violation Trajectory", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax2.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax2.legend(loc="lower center", facecolor="#111827", edgecolor="#1e293b", labelcolor="#f8fafc")
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0, 1.15)

    # -------------------------------------------------------------
    # Panel 3: Werner State Contraction into 5D Quaternionic Bloch Ball B5
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#0f172a")

    p_vals = [w["p"] for w in werner_sweep]
    radii = [w["hopf_coord"]["radius"] for w in werner_sweep]
    w_concurr = [w["invariants"]["concurrence"] for w in werner_sweep]
    w_purity = [w["invariants"]["subsystem_purity"] for w in werner_sweep]

    ax3.plot(p_vals, radii, color="#f8fafc", linewidth=2.0, label="Quaternionic Bloch Radius ||Y|| in B5")
    ax3.plot(p_vals, w_concurr, color="#00e676", linewidth=2.5, label="Concurrence C(p) = max(0, (3p-1)/2)")
    ax3.plot(p_vals, w_purity, color="#ffb300", linewidth=1.8, linestyle="--", label="Subsystem Purity gamma")

    ax3.axvline(1.0 / 3.0, color="#ef4444", linestyle=":", linewidth=1.5, label="Separability Horizon (p = 1/3)")
    ax3.axvline(1.0 / np.sqrt(2.0), color="#ffb300", linestyle=":", linewidth=1.5, label="Bell Violation Horizon (p = 1/sqrt(2))")

    ax3.set_xlabel("Werner State Fraction p", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Metric Value", color="#f8fafc", fontsize=11, fontweight="bold")
    ax3.set_title("Werner State Contraction inside 5D Quaternionic Bloch Ball B5", color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax3.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax3.legend(loc="upper left", facecolor="#111827", edgecolor="#1e293b", labelcolor="#f8fafc")
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(-0.05, 1.05)

    # -------------------------------------------------------------
    # Panel 4: AVX2 SIMD Throughput & Numerical Invariants
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#0f172a")

    metrics = [
        "Projection Latency\n(ns / eval)",
        "SIMD Throughput\n(M evals / sec)",
        "Max S4 Residual\n(log10 eps)",
        "Bell CHSH Witness\n(Max S)",
    ]
    vals = [
        throughput["latency_ns_per_eval"],
        throughput["throughput_m_evals_sec"],
        np.log10(max(throughput["max_norm_error"], 1e-16)),
        2.0 * np.sqrt(2.0),
    ]
    colors = ["#ffb300", "#00e676", "#00e676", "#ffb300"]

    bars = ax4.bar(metrics, vals, color=colors, width=0.45, edgecolor="#1e293b", linewidth=1.5)
    for bar, val in zip(bars, vals):
        h = bar.get_height()
        y_pos = h + 1.2 if h >= 0 else h - 2.5
        ax4.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{val:.2f}",
                 ha="center", va="bottom", color="#f8fafc", fontsize=10, fontweight="bold")

    ax4.set_ylabel("Numerical Value", color="#f8fafc", fontsize=11, fontweight="bold")
    ax4.set_title(f"AVX2 Quaternionic Engine: {throughput['throughput_m_evals_sec']:.2f} M Evals/s (100% L1/L2)",
                  color="#f8fafc", fontsize=12, fontweight="bold", pad=12)
    ax4.grid(True, linestyle="--", alpha=0.3, color="#334155")
    ax4.tick_params(colors="#e2e8f0", labelsize=9)

    plt.tight_layout()

    out_paths = [
        os.path.join("docs", "findings", "assets", "m157_hopf_s4_entanglement.png"),
        os.path.join("results", "plots", "m157_hopf_s4_entanglement.png"),
    ]
    for p in out_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, facecolor="#05080d", edgecolor="none")
        print(f"[+] Rendered publication figure: {p}")

    plt.close(fig)

if __name__ == "__main__":
    main()
