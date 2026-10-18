// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Deen OS Sovereign Telemetry & ANSI Visualizer for Quantum Conway's Game of Life.
//! Formats quantum grid states with Obsidian (#05080d), Emerald (#00e676), Amber (#ffb300),
//! and Slate (#334155) without emojis, using strict Unicode box-drawing characters.

use crate::lattice::Lattice2D;
use crate::simulator::StepTelemetry;
use crate::state::QuantumCellStateSoA;

/// ANSI color constants conforming to Deen OS Sovereign Visual Specification.
pub mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const EMERALD: &str = "\x1b[38;2;0;230;118m"; // #00e676 (Living cells)
    pub const AMBER: &str = "\x1b[38;2;255;179;0m"; // #ffb300 (FFS l=2 cells)
    pub const SLATE_GRID: &str = "\x1b[38;2;51;65;85m"; // #334155 (Vacuum / Grid)
    pub const POLAR_WHITE: &str = "\x1b[38;2;248;250;252m"; // #f8fafc (Monospace text)
    pub const ALERT_RED: &str = "\x1b[38;2;239;68;68m"; // #ef4444 (Alerts / Norm err)
}

/// Formatter for visual and structured telemetry readouts.
#[derive(Debug, Clone, Default)]
pub struct TelemetryVisualizer;

impl TelemetryVisualizer {
    /// Render the 2D quantum lattice into an ANSI color-coded terminal view.
    pub fn render_grid(lattice: &Lattice2D, state: &QuantumCellStateSoA) -> String {
        let mut out = String::new();
        let w = lattice.width;
        let h = lattice.height;

        // Header border
        out.push_str(colors::SLATE_GRID);
        out.push_str("  ┌");
        for _ in 0..w {
            out.push_str("──");
        }
        out.push_str("─┐\n");

        for y in 0..h {
            out.push_str("  │ ");
            for x in 0..w {
                let idx = lattice.coord_to_idx(x, y);
                let p1 = state.prob_living(idx);
                let p2 = state.prob_ffs(idx);

                if p2 > 0.3 {
                    // Fractional Fermi Sea excitation (Amber)
                    out.push_str(colors::AMBER);
                    out.push_str("▲ ");
                } else if p1 > 0.6 {
                    // Strong living occupation (Emerald)
                    out.push_str(colors::EMERALD);
                    out.push_str("● ");
                } else if p1 > 0.15 {
                    // Quantum superposition branch (Dim Emerald)
                    out.push_str("\x1b[38;2;16;185;129m");
                    out.push_str("○ ");
                } else {
                    // Vacuum (Muted Slate)
                    out.push_str(colors::SLATE_GRID);
                    out.push_str("· ");
                }
            }
            out.push_str(colors::SLATE_GRID);
            out.push_str("│\n");
        }

        // Footer border
        out.push_str("  └");
        for _ in 0..w {
            out.push_str("──");
        }
        out.push_str("─┘\n");
        out.push_str(colors::RESET);

        out
    }

    /// Render a single-line or multi-line telemetry box for a simulation step.
    pub fn format_banner(telemetry: &StepTelemetry) -> String {
        let mut s = String::new();
        s.push_str(colors::POLAR_WHITE);
        s.push_str("╔══════════════════════════════════════════════════════════════════════════════════════════╗\n");
        s.push_str(&format!(
            "║ STEP: {:04} │ N_TOT: {:6.2} │ LIVING: {:5.2} │ FFS(l=2): {:5.2} │ NORM_ERR: {:9.2e} ║\n",
            telemetry.step,
            telemetry.total_population,
            telemetry.living_population,
            telemetry.ffs_population,
            telemetry.max_norm_error
        ));
        s.push_str(&format!(
            "║ ENTANGLEMENT S: {:6.4} nats │ MEAN MOMENTUM DENSITY n(k): {:6.4} (FFS PLATEAU: 0.5000) ║\n",
            telemetry.entanglement_entropy,
            telemetry.mean_momentum_density
        ));
        s.push_str("╚══════════════════════════════════════════════════════════════════════════════════════════╝\n");
        s.push_str(colors::RESET);
        s
    }
}
