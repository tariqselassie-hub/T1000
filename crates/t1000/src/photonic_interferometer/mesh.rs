// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Optical Waveguide Mesh & Beamsplitter Network
//!
//! Synthesizes the 16-mode unitary transition matrix $U_{16} = S_{16} C_{16}$ using
//! exclusively balanced 50:50 directional couplers ($H_2, H_4$).

use serde::{Deserialize, Serialize};

/// Directed optical arc mode representation in the interferometer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhotonicMode {
    pub mode_idx: usize,
    pub source_node: usize,
    pub target_node: usize,
    pub transition_prob: &'static str,
}

/// The 16 physical optical spatial/temporal modes corresponding to the non-zero arcs of $P$.
pub const PHOTONIC_MODES: [PhotonicMode; 16] = [
    // Hub v1 outgoing arcs (4 modes: 1/4 prob each)
    PhotonicMode {
        mode_idx: 0,
        source_node: 0,
        target_node: 1,
        transition_prob: "1/4",
    },
    PhotonicMode {
        mode_idx: 1,
        source_node: 0,
        target_node: 2,
        transition_prob: "1/4",
    },
    PhotonicMode {
        mode_idx: 2,
        source_node: 0,
        target_node: 5,
        transition_prob: "1/4",
    },
    PhotonicMode {
        mode_idx: 3,
        source_node: 0,
        target_node: 6,
        transition_prob: "1/4",
    },
    // Node v2 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode {
        mode_idx: 4,
        source_node: 1,
        target_node: 3,
        transition_prob: "1/2",
    },
    PhotonicMode {
        mode_idx: 5,
        source_node: 1,
        target_node: 6,
        transition_prob: "1/2",
    },
    // Node v3 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode {
        mode_idx: 6,
        source_node: 2,
        target_node: 0,
        transition_prob: "1/2",
    },
    PhotonicMode {
        mode_idx: 7,
        source_node: 2,
        target_node: 5,
        transition_prob: "1/2",
    },
    // Node v4 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode {
        mode_idx: 8,
        source_node: 3,
        target_node: 1,
        transition_prob: "1/2",
    },
    PhotonicMode {
        mode_idx: 9,
        source_node: 3,
        target_node: 4,
        transition_prob: "1/2",
    },
    // Node v5 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode {
        mode_idx: 10,
        source_node: 4,
        target_node: 3,
        transition_prob: "1/2",
    },
    PhotonicMode {
        mode_idx: 11,
        source_node: 4,
        target_node: 6,
        transition_prob: "1/2",
    },
    // Node v6 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode {
        mode_idx: 12,
        source_node: 5,
        target_node: 0,
        transition_prob: "1/2",
    },
    PhotonicMode {
        mode_idx: 13,
        source_node: 5,
        target_node: 2,
        transition_prob: "1/2",
    },
    // Node v7 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode {
        mode_idx: 14,
        source_node: 6,
        target_node: 0,
        transition_prob: "1/2",
    },
    PhotonicMode {
        mode_idx: 15,
        source_node: 6,
        target_node: 4,
        transition_prob: "1/2",
    },
];

/// Physical 50:50 directional coupler (beamsplitter) in the integrated mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Beamsplitter2x2 {
    pub coupler_id: usize,
    pub input_port_a: usize,
    pub input_port_b: usize,
    pub output_port_a: usize,
    pub output_port_b: usize,
}

/// Computes the 16x16 unitary optical scattering matrix $U_{\text{opt}} = S_{16} C_{16}$.
pub fn compile_mesh_unitary() -> ([[f64; 16]; 16], Vec<Beamsplitter2x2>) {
    let mut c_16 = [[0.0f64; 16]; 16];
    let inv_sqrt2 = 1.0 / 2.0f64.sqrt();

    // 4-port Hub mixer (4 beamsplitters)
    let h4_base = [
        [0.5, 0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
    ];
    for r in 0..4 {
        for c in 0..4 {
            c_16[r][c] = h4_base[r][c];
        }
    }

    // Six 2-port mixers (1 beamsplitter each)
    for k in 0..6 {
        let offset = 4 + 2 * k;
        c_16[offset][offset] = inv_sqrt2;
        c_16[offset][offset + 1] = inv_sqrt2;
        c_16[offset + 1][offset] = inv_sqrt2;
        c_16[offset + 1][offset + 1] = -inv_sqrt2;
    }

    // Routing permutation matrix S_16:
    let perm: [usize; 16] = [
        6, 12, 14, 3, // Target Node 0 inputs mapped to Hub mixer ports 0..3
        0, 8,         // Target Node 1 inputs
        1, 13,        // Target Node 2 inputs
        4, 10,        // Target Node 3 inputs
        9, 15,        // Target Node 4 inputs
        2, 7,         // Target Node 5 inputs
        5, 11,        // Target Node 6 inputs
    ];

    let mut s_16 = [[0.0f64; 16]; 16];
    for i in 0..16 {
        s_16[i][perm[i]] = 1.0;
    }

    // Compile full unitary U_16 = S_16 * C_16
    let mut u_16 = [[0.0f64; 16]; 16];
    for r in 0..16 {
        for k in 0..16 {
            let srk = s_16[r][k];
            if srk.abs() > 1e-15 {
                for c in 0..16 {
                    u_16[r][c] += srk * c_16[k][c];
                }
            }
        }
    }

    let couplers = vec![
        Beamsplitter2x2 { coupler_id: 0, input_port_a: 0, input_port_b: 1, output_port_a: 0, output_port_b: 1 },
        Beamsplitter2x2 { coupler_id: 1, input_port_a: 2, input_port_b: 3, output_port_a: 2, output_port_b: 3 },
        Beamsplitter2x2 { coupler_id: 2, input_port_a: 0, input_port_b: 2, output_port_a: 0, output_port_b: 2 },
        Beamsplitter2x2 { coupler_id: 3, input_port_a: 1, input_port_b: 3, output_port_a: 1, output_port_b: 3 },
        Beamsplitter2x2 { coupler_id: 4, input_port_a: 4, input_port_b: 5, output_port_a: 4, output_port_b: 5 },
        Beamsplitter2x2 { coupler_id: 5, input_port_a: 6, input_port_b: 7, output_port_a: 6, output_port_b: 7 },
        Beamsplitter2x2 { coupler_id: 6, input_port_a: 8, input_port_b: 9, output_port_a: 8, output_port_b: 9 },
        Beamsplitter2x2 { coupler_id: 7, input_port_a: 10, input_port_b: 11, output_port_a: 10, output_port_b: 11 },
        Beamsplitter2x2 { coupler_id: 8, input_port_a: 12, input_port_b: 13, output_port_a: 12, output_port_b: 13 },
        Beamsplitter2x2 { coupler_id: 9, input_port_a: 14, input_port_b: 15, output_port_a: 14, output_port_b: 15 },
    ];

    (u_16, couplers)
}

