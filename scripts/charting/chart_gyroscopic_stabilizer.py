# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
8-Node Gyroscopic Throwing-Jack & Wobbling Pulsar Visualizer
Generates a 4-panel 300 DPI high-resolution telemetry chart adhering strictly
to the Deen OS Sovereign Visual Specification.
"""

import json
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Sovereign Visual Palette
BG_OBSIDIAN = "#05080d"
CARD_CHARCOAL = "#0f172a"
QUANTUM_EMERALD = "#00e676"
HARDWARE_AMBER = "#ffb300"
ALERT_CRIMSON = "#ef4444"
TEXT_POLAR = "#f8fafc"
TEXT_MUTED = "#94a3b8"
GRID_SLATE = "#1e293b"

def plot_gyroscopic_stabilizer():
    json_path = os.path.join("results", "hosts", "unknown-cpu", "experiments", "gyroscopic_stabilizer.json")
    if not os.path.exists(json_path):
        json_path = os.path.join("results", "experiments", "gyroscopic_stabilizer.json")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mechanics = data["mechanics_steps"]
    quantum = data["quantum_steps"]

    time = np.array([m["time"] for m in mechanics])
    tau_a_norm = np.array([math.sqrt(m["torque_jet_a"]["x"]**2 + m["torque_jet_a"]["y"]**2 + m["torque_jet_a"]["z"]**2) for m in mechanics])
    tau_b_norm = np.array([math.sqrt(m["torque_jet_b"]["x"]**2 + m["torque_jet_b"]["y"]**2 + m["torque_jet_b"]["z"]**2) for m in mechanics])
    tau_net_norm = np.array([math.sqrt(m["torque_net"]["x"]**2 + m["torque_net"]["y"]**2 + m["torque_net"]["z"]**2) for m in mechanics])
    com_drift = np.array([m["center_of_mass_drift"] for m in mechanics])

    phi_a = np.array([q["phase_noise_a_rad"] for q in quantum])
    phi_b = np.array([q["phase_noise_b_rad"] for q in quantum])
    fidelity = np.array([q["core_fidelity"] for q in quantum])
    parity = np.array([q["parity_correlator_zab"] for q in quantum])

    fig = plt.figure(figsize=(16, 11), facecolor=BG_OBSIDIAN)
    plt.subplots_adjust(left=0.07, right=0.95, top=0.91, bottom=0.07, wspace=0.25, hspace=0.30)

    # Sovereign Title Banner
    fig.text(0.5, 0.965, "T-1000 GYROSCOPIC THROWING-JACK & PULSAR ZERO-POINT STABILIZER", 
             color=TEXT_POLAR, fontsize=15, fontweight="bold", ha="center", fontfamily="monospace")
    fig.text(0.5, 0.940, "8-Node Dual-Cone Hourglass Topology  |  Torque Cancellation tau_net = 0  |  DFS Phase-Lock F_(4,5) = 1.000",
             color=QUANTUM_EMERALD, fontsize=11, ha="center", fontfamily="monospace")

    # -------------------------------------------------------------
    # PANEL 1 (Top Left): 3D Throwing-Jack & Precession Geometry
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1, projection="3d", facecolor=CARD_CHARCOAL)
    ax1.set_facecolor(CARD_CHARCOAL)
    
    # Draw central fulcrum (4, 5)
    ax1.scatter([0], [0], [0], color=QUANTUM_EMERALD, s=250, edgecolor=TEXT_POLAR, linewidth=2, label="Fulcrum Node (4,5)")
    
    # Canonical Jack arm coordinates at t=0
    nodes = data["initial_topology"]["nodes"]
    for n in nodes:
        pos = n["initial_pos"]
        if n["id"] in [1, 2, 3]:
            col = HARDWARE_AMBER
            ax1.plot([0, pos["x"]], [0, pos["y"]], [0, pos["z"]], color=HARDWARE_AMBER, linewidth=2.5, alpha=0.8)
            ax1.scatter([pos["x"]], [pos["y"]], [pos["z"]], color=HARDWARE_AMBER, s=120)
            ax1.text(pos["x"]*1.15, pos["y"]*1.15, pos["z"]*1.05, f"Node {n['id']}", color=TEXT_POLAR, fontsize=8, fontfamily="monospace")
        elif n["id"] in [6, 7, 8]:
            col = ALERT_CRIMSON
            ax1.plot([0, pos["x"]], [0, pos["y"]], [0, pos["z"]], color=ALERT_CRIMSON, linewidth=2.5, alpha=0.8)
            ax1.scatter([pos["x"]], [pos["y"]], [pos["z"]], color=ALERT_CRIMSON, s=120)
            ax1.text(pos["x"]*1.15, pos["y"]*1.15, pos["z"]*1.05, f"Node {n['id']}", color=TEXT_POLAR, fontsize=8, fontfamily="monospace")
            
    # Draw precessional guide rings
    theta_ring = np.linspace(0, 2*np.pi, 60)
    r_arm = 1.5
    zh = r_arm * 0.8165
    ax1.plot(r_arm * np.cos(theta_ring), r_arm * np.sin(theta_ring), zh, color=HARDWARE_AMBER, linestyle="--", alpha=0.4)
    ax1.plot(r_arm * np.cos(theta_ring), r_arm * np.sin(theta_ring), -zh, color=ALERT_CRIMSON, linestyle="--", alpha=0.4)
    
    ax1.set_title("3D Throwing-Jack & Dual-Cone Topology", color=TEXT_POLAR, fontsize=12, pad=12, fontfamily="monospace")
    ax1.grid(True, color=GRID_SLATE, alpha=0.5)
    ax1.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax1.legend(loc="upper left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8)

    # -------------------------------------------------------------
    # PANEL 2 (Top Right): Torque Cancellation & Balancing
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 2, facecolor=CARD_CHARCOAL)
    ax2.plot(time, tau_a_norm, color=HARDWARE_AMBER, linewidth=2.0, label="||tau_A(t)|| (Jet A)")
    ax2.plot(time, tau_b_norm, color=ALERT_CRIMSON, linestyle="--", linewidth=2.0, label="||tau_B(t)|| (Jet B)")
    ax2.plot(time, tau_net_norm * 1e14, color=QUANTUM_EMERALD, linewidth=2.5, label="||tau_net(t)|| (x10^-14 N*m)")

    ax2.set_title("Dynamic Precessional Torque Cancellation", color=TEXT_POLAR, fontsize=12, fontfamily="monospace")
    ax2.set_xlabel("Time t (seconds)", color=TEXT_MUTED, fontsize=10, fontfamily="monospace")
    ax2.set_ylabel("Torque Magnitude (N*m)", color=TEXT_MUTED, fontsize=10, fontfamily="monospace")
    ax2.grid(True, color=GRID_SLATE, linestyle=":", alpha=0.6)
    ax2.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax2.legend(loc="upper right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8)

    # -------------------------------------------------------------
    # PANEL 3 (Bottom Left): Polar Phase Flutter vs Core Lock
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 3, facecolor=CARD_CHARCOAL)
    ax3.plot(time, phi_a, color=HARDWARE_AMBER, linewidth=1.8, label="Jet A Phase Noise delta(t)")
    ax3.plot(time, phi_b, color=ALERT_CRIMSON, linestyle="--", linewidth=1.8, label="Jet B Counter-Phase -delta(t)")
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(time, fidelity, color=QUANTUM_EMERALD, linewidth=2.5, label="Core Fidelity F_(4,5)(t)")
    ax3_twin.set_ylim(0.99999990, 1.00000005)
    ax3_twin.set_ylabel("Core State Fidelity", color=QUANTUM_EMERALD, fontsize=10, fontfamily="monospace")
    ax3_twin.tick_params(colors=QUANTUM_EMERALD, labelsize=9)

    ax3.set_title("Kinetic Phase Flutter vs Core DFS Invariance", color=TEXT_POLAR, fontsize=12, fontfamily="monospace")
    ax3.set_xlabel("Time t (seconds)", color=TEXT_MUTED, fontsize=10, fontfamily="monospace")
    ax3.set_ylabel("Phase Perturbation (rad)", color=TEXT_MUTED, fontsize=10, fontfamily="monospace")
    ax3.grid(True, color=GRID_SLATE, linestyle=":", alpha=0.6)
    ax3.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax3.legend(loc="lower left", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8)

    # -------------------------------------------------------------
    # PANEL 4 (Bottom Right): Center of Mass & Parity Correlator
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4, facecolor=CARD_CHARCOAL)
    ax4.plot(time, parity, color=QUANTUM_EMERALD, linewidth=2.0, label="Parity Correlator <Z_A Z_B> = -1.0")
    
    ax4_twin = ax4.twinx()
    ax4_twin.plot(time, com_drift * 1e16, color=HARDWARE_AMBER, linestyle=":", linewidth=2.0, label="Center of Mass Drift (x10^-16 m)")
    ax4_twin.set_ylabel("COM Drift (10^-16 m)", color=HARDWARE_AMBER, fontsize=10, fontfamily="monospace")
    ax4_twin.tick_params(colors=HARDWARE_AMBER, labelsize=9)

    ax4.set_title("Parity Conservation & Invariant Barycenter", color=TEXT_POLAR, fontsize=12, fontfamily="monospace")
    ax4.set_xlabel("Time t (seconds)", color=TEXT_MUTED, fontsize=10, fontfamily="monospace")
    ax4.set_ylabel("Parity Expectation Value", color=TEXT_MUTED, fontsize=10, fontfamily="monospace")
    ax4.set_ylim(-1.05, -0.95)
    ax4.grid(True, color=GRID_SLATE, linestyle=":", alpha=0.6)
    ax4.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax4.legend(loc="lower right", facecolor=CARD_CHARCOAL, edgecolor=GRID_SLATE, labelcolor=TEXT_POLAR, fontsize=8)

    out_dir = os.path.join("results", "charts")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "gyroscopic_stabilizer_manifold.png")
    plt.savefig(out_png, dpi=300, facecolor=BG_OBSIDIAN, edgecolor="none")
    plt.close()
    print(f"Sovereign Visual Chart generated at {out_png}")

if __name__ == "__main__":
    plot_gyroscopic_stabilizer()
