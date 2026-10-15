# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

"""
T-1000 Sovereign Charting: Molecule #109 (NCc1cccc(CO)c1) Pharmacophore Attractor.
Visualizes the convergence of 3 inactive precursors (#297, #304, #93) into #109 with 3.7x-9.3x potency jumps.
Strictly adheres to Deen OS visual specification (Deep Obsidian, Emerald Green, Hardware Gold, Alert Red).
Zero emojis. Zero cyan/purple.
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

    report_path = Path("results/experiments/lead_109_convergence_analysis.json")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lead = data["lead_molecule"]
    precursors = data["precursors"]
    profiles = data["convergence_profiles"]
    attestation = data["fpga_attestation"]

    fig = plt.figure(figsize=(18, 12), facecolor=BG_OBSIDIAN)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

    # -------------------------------------------------------------
    # SUBPLOT 1: The 2D Molecular Manifold Attractor Funnel
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    style_card(
        ax1,
        title="2D MANIFOLD CONVERGENCE FUNNEL (LOCAL CLUSTER)",
        subtitle="Three Distinct Inactive Precursors Converging onto Molecule #109",
    )

    # Plot Lead #109
    ax1.scatter(
        [lead["manifold_x"]],
        [lead["manifold_y"]],
        color=GREEN_EMERALD,
        s=260,
        edgecolors=GOLD_BRIGHT,
        linewidths=2.0,
        zorder=6,
        label=f"#109 Active Lead (Act: {lead['bioactivity']:.2f})",
    )

    # Plot Precursors & Vectors
    colors = [RED_ALERT, GOLD_HARDWARE, RED_CRIMSON]
    labels = ["#297 (-NH2, Act: 0.92)", "#304 (-Et, Act: 1.94)", "#93 (-F, Act: 2.31)"]

    for p, prof, col, lbl in zip(precursors, profiles, colors, labels):
        ax1.scatter([p["manifold_x"]], [p["manifold_y"]], color=col, s=140, edgecolors=GRID_SLATE, linewidths=1.2, zorder=5, label=lbl)

        # Quiver vector pointing to #109
        dx = lead["manifold_x"] - p["manifold_x"]
        dy = lead["manifold_y"] - p["manifold_y"]
        ax1.annotate(
            "",
            xy=(lead["manifold_x"], lead["manifold_y"]),
            xytext=(p["manifold_x"], p["manifold_y"]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col,
                lw=2.2,
                mutation_scale=16,
            ),
            zorder=4,
        )

    # Annotate Lead #109 Callout
    ax1.annotate(
        f"MASTER ATTRACTOR #109\nSMILES: {lead['smiles']}\nBioactivity: {lead['bioactivity']:.2f}\nLogP: {lead['logp']:.2f} | TPSA: {lead['tpsa']:.1f}",
        xy=(lead["manifold_x"], lead["manifold_y"]),
        xytext=(lead["manifold_x"] - 0.09, lead["manifold_y"] - 0.09),
        arrowprops=dict(facecolor=GOLD_BRIGHT, edgecolor=GRID_SLATE, width=1.0, headwidth=5),
        color=TEXT_POLAR,
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="square,pad=0.5", fc=CARD_CHARCOAL, ec=GREEN_QUANTUM, lw=1.2),
    )

    ax1.set_xlabel("MIMAC Manifold Coordinate X", fontsize=9, color=TEXT_SLATE)
    ax1.set_ylabel("MIMAC Manifold Coordinate Y", fontsize=9, color=TEXT_SLATE)
    ax1.set_xlim(7.68, 7.89)
    ax1.set_ylim(6.12, 6.38)
    ax1.legend(loc="upper left")

    # -------------------------------------------------------------
    # SUBPLOT 2: Structural Architecture & Orbital Mechanism
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    style_card(
        ax2,
        title="MOLECULAR ARCHITECTURE & ORBITAL MECHANISM",
        subtitle="Meta-Substituted Benzylamine Scaffold Functional Breakdown",
    )
    ax2.axis("off")

    mech_text = [
        ("CORE SCAFFOLD", "meta-Substituted Benzylamine:  [ R ] ── ( Arene Ring ) ── CH2-NH2", GOLD_BRIGHT),
        ("", "", TEXT_POLAR),
        ("ACTIVE LEAD (#109)", "R = -CH2OH (Hydroxymethyl)  -->  Bioactivity: 8.56 (Top 1% Lead)", GREEN_EMERALD),
        ("Mechanism", "• sp3 methylene (-CH2-) spacer decouples lone pairs from arene resonance.", TEXT_SLATE),
        ("Conformation", "• 360-degree rotational freedom reaches deep catalytic hydrogen bonding pocket.", TEXT_SLATE),
        ("", "", TEXT_POLAR),
        ("PRECURSOR #297", "R = -NH2 (Aromatic Amino)      -->  Bioactivity: 0.92 (9.3x Dud)", RED_ALERT),
        ("Conjugation Trap", "• Nitrogen lone pair conjugates into arene pi-system (planar sp2 geometry).", TEXT_SLATE),
        ("Geometric Failure", "• 1.54 A shorter, rigid coplanar trap misses catalytic target entirely.", TEXT_SLATE),
        ("", "", TEXT_POLAR),
        ("PRECURSOR #304", "R = -CH2CH3 (Ethyl Group)     -->  Bioactivity: 1.94 (4.4x Dud)", GOLD_HARDWARE),
        ("Steric Mismatch", "• Completely lipophilic alkyl chain lacks hydrogen bond donor/acceptor capacity.", TEXT_SLATE),
        ("", "", TEXT_POLAR),
        ("PRECURSOR #93", "R = -F (Fluoro Halogen)        -->  Bioactivity: 2.31 (3.7x Dud)", RED_CRIMSON),
        ("Polarization Deficit", "• Electronegative halogen bound directly to ring with zero donor reach.", TEXT_SLATE),
    ]

    y_cur = 0.96
    for title_part, desc_part, col in mech_text:
        if not title_part and not desc_part:
            y_cur -= 0.02
            continue
        if title_part in ["CORE SCAFFOLD", "ACTIVE LEAD (#109)", "PRECURSOR #297", "PRECURSOR #304", "PRECURSOR #93"]:
            ax2.text(0.02, y_cur, f"{title_part}: {desc_part}", transform=ax2.transAxes, fontsize=9.5, fontweight="bold", color=col)
            y_cur -= 0.055
        else:
            ax2.text(0.05, y_cur, f"{desc_part}", transform=ax2.transAxes, fontsize=8.5, color=col)
            y_cur -= 0.048

    # -------------------------------------------------------------
    # SUBPLOT 3: 10-Qubit Hilbert Space Expectation Spectrum
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    style_card(
        ax3,
        title="10-QUBIT HILBERT OBSERVABLE EXPECTATIONS <Z_q>",
        subtitle="Invariant Global Wavefunction Channels vs Active Electronic Shift Channels",
    )

    qubits = np.arange(10)
    width = 0.20

    ax3.bar(qubits - 1.5 * width, lead["z_expectations"], width=width, color=GREEN_QUANTUM, label="#109 Active (-CH2OH)", alpha=0.9)
    ax3.bar(qubits - 0.5 * width, precursors[0]["z_expectations"], width=width, color=RED_ALERT, label="#297 Inactive (-NH2)", alpha=0.8)
    ax3.bar(qubits + 0.5 * width, precursors[1]["z_expectations"], width=width, color=GOLD_HARDWARE, label="#304 Inactive (-Et)", alpha=0.8)
    ax3.bar(qubits + 1.5 * width, precursors[2]["z_expectations"], width=width, color=RED_CRIMSON, label="#93 Inactive (-F)", alpha=0.8)

    ax3.set_xlabel("Hilbert Space Qubit Index (q = 0 to 9)", fontsize=9, color=TEXT_SLATE)
    ax3.set_ylabel("Pauli Observable Expectation <Z_q>", fontsize=9, color=TEXT_SLATE)
    ax3.set_xticks(qubits)
    ax3.set_xticklabels([f"q{q}" for q in qubits])
    ax3.legend(loc="lower right")

    # -------------------------------------------------------------
    # SUBPLOT 4: Hardware Silicon Attestation & Telemetry Matrix
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    style_card(
        ax4,
        title="PHYSICAL SILICON ATTESTATION & CONVERGENCE MATRIX",
        subtitle=f"Artix-7 FPGA (COM6 @ 3.0 MBaud) Execution Attestation",
    )
    ax4.axis("off")

    # Hardware Banner
    status_color = GREEN_EMERALD if attestation["status"] == "verified_on_silicon" else GOLD_HARDWARE
    hw_text = (
        f"PHYSICAL FPGA ATTESTATION: {attestation['status'].upper()}\n"
        f"Silicon Compute: {attestation['silicon_exec_us']:.2f} us (12 Layers @ 100 MHz) | ACK Latency: {attestation['ack_ms']:.3f} ms\n"
        f"Hardware Hash Seed: {attestation['hash_seed'][:32]}..."
    )
    rect_hw = plt.Rectangle((0.0, 0.78), 1.0, 0.20, facecolor=CARD_CHARCOAL, edgecolor=status_color, lw=1.2, transform=ax4.transAxes)
    ax4.add_patch(rect_hw)
    ax4.text(0.03, 0.88, hw_text, transform=ax4.transAxes, fontsize=8, color=TEXT_POLAR, verticalalignment="center", family="monospace")

    # Convergence Matrix Table
    table_rows = [
        ["Molecule ID", "Substituent", "Bioactivity", "Fold Gain", "Quantum Fid", "Tanimoto", "LogP", "TPSA"],
        ["#109 (Lead)", "-CH2OH", f"{lead['bioactivity']:.2f}", "1.0x (Ref)", "1.000", "1.000", f"{lead['logp']:.2f}", f"{lead['tpsa']:.1f}"],
        ["#297 (Dud)", "-NH2", f"{precursors[0]['bioactivity']:.2f}", f"{profiles[0]['potency_fold']:.1f}x", f"{profiles[0]['quantum_fidelity']:.3f}", f"{profiles[0]['tanimoto_similarity']:.3f}", f"{precursors[0]['logp']:.2f}", f"{precursors[0]['tpsa']:.1f}"],
        ["#304 (Dud)", "-CH2CH3", f"{precursors[1]['bioactivity']:.2f}", f"{profiles[1]['potency_fold']:.1f}x", f"{profiles[1]['quantum_fidelity']:.3f}", f"{profiles[1]['tanimoto_similarity']:.3f}", f"{precursors[1]['logp']:.2f}", f"{precursors[1]['tpsa']:.1f}"],
        ["#93 (Dud)", "-F", f"{precursors[2]['bioactivity']:.2f}", f"{profiles[2]['potency_fold']:.1f}x", f"{profiles[2]['quantum_fidelity']:.3f}", f"{profiles[2]['tanimoto_similarity']:.3f}", f"{precursors[2]['logp']:.2f}", f"{precursors[2]['tpsa']:.1f}"],
    ]

    col_widths = [0.15, 0.14, 0.13, 0.13, 0.13, 0.11, 0.10, 0.11]
    y_pos = 0.68
    for row_idx, row in enumerate(table_rows):
        x_pos = 0.02
        is_hdr = row_idx == 0
        is_lead = row_idx == 1
        bg_col = CARD_CHARCOAL if is_hdr else (BG_OBSIDIAN if row_idx % 2 == 0 else "#0c1322")
        txt_col = GOLD_BRIGHT if is_hdr else (GREEN_EMERALD if is_lead else TEXT_POLAR)

        rect = plt.Rectangle((0.0, y_pos - 0.08), 1.0, 0.11, facecolor=bg_col, edgecolor=GRID_SLATE, lw=0.8, transform=ax4.transAxes)
        ax4.add_patch(rect)

        for cell, w in zip(row, col_widths):
            fontweight = "bold" if (is_hdr or is_lead) else "normal"
            ax4.text(x_pos, y_pos - 0.03, cell, transform=ax4.transAxes, fontsize=8, color=txt_col, fontweight=fontweight)
            x_pos += w
        y_pos -= 0.12

    # Super Title
    fig.suptitle(
        "T-1000 MOLECULAR ATTRACTOR INVESTIGATION: NCc1cccc(CO)c1 (LEAD #109)\n"
        f"Multi-Path Pharmacophore Convergence: 3 Distinct SAR Cliffs Unlocking 3.7x to 9.3x Potency Multipliers",
        fontsize=13,
        fontweight="bold",
        color=TEXT_POLAR,
        y=0.98,
    )

    out_file = Path("results/charts/lead_109_attractor_analysis.png")
    saved = save_deen_figure(fig, out_file, dpi=300)
    print(f"  -> Generated Molecule #109 Attractor Chart: {saved}")

if __name__ == "__main__":
    main()
