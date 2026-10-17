// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Euler Dual-Graph Table & Bitmask Spoke Canceler for IBM Nighthawk R2
//!
//! Maps every elementary square plaquette ($P_0 \dots P_{98}$) and arbitrary sub-lattice
//! masks directly into static, cache-line-aligned bitmasks.
//!
//! Because shared internal couplers satisfy $Z_e^2 = I$, contracting any arbitrary
//! region $\Sigma$ of plaquettes reduces to a bitwise XOR (`^`) across the edge masks
//! of the selected faces.
//!
//! Fits all 218 couplers in a 256-bit (`[u64; 4]`) register and keeps the active
//! Plaquette-Zipper window inside exactly 384 bytes (six 64-byte L1 cache lines).

use serde::{Deserialize, Serialize};

pub const NUM_VERTICES: usize = 120;
pub const NUM_COUPLERS: usize = 218;
pub const NUM_PLAQUETTES: usize = 99;

/// 256-bit coupler bitmask (fits all 218 couplers in 32 bytes / half an L1 cache line).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(align(32))]
pub struct CouplerMask(pub [u64; 4]);

impl CouplerMask {
    #[inline(always)]
    pub const fn empty() -> Self {
        Self([0; 4])
    }

    #[inline(always)]
    pub const fn set_bit(&mut self, edge_id: usize) {
        self.0[edge_id >> 6] |= 1u64 << (edge_id & 63);
    }

    #[inline(always)]
    pub const fn test_bit(&self, edge_id: usize) -> bool {
        (self.0[edge_id >> 6] & (1u64 << (edge_id & 63))) != 0
    }

    /// Algebraic $Z_e^2 = I$ involution: XOR cancels any coupler shared an even number of times.
    #[inline(always)]
    pub const fn xor(&self, other: &Self) -> Self {
        Self([
            self.0[0] ^ other.0[0],
            self.0[1] ^ other.0[1],
            self.0[2] ^ other.0[2],
            self.0[3] ^ other.0[3],
        ])
    }

    #[inline(always)]
    pub const fn bitor(&self, other: &Self) -> Self {
        Self([
            self.0[0] | other.0[0],
            self.0[1] | other.0[1],
            self.0[2] | other.0[2],
            self.0[3] | other.0[3],
        ])
    }

    #[inline(always)]
    pub const fn bitand_not(&self, other: &Self) -> Self {
        Self([
            self.0[0] & !other.0[0],
            self.0[1] & !other.0[1],
            self.0[2] & !other.0[2],
            self.0[3] & !other.0[3],
        ])
    }

    #[inline(always)]
    pub const fn count_ones(&self) -> u32 {
        self.0[0].count_ones()
            + self.0[1].count_ones()
            + self.0[2].count_ones()
            + self.0[3].count_ones()
    }
}

/// Elementary $2 \times 2$ Plaquette descriptor (8 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct PlaquetteNode {
    pub edges: [u8; 4],    // [Top, Bottom, Left, Right] coupler IDs (0..217)
    pub vertices: [u8; 4], // [TL, TR, BL, BR] physical qubit IDs (0..119)
}

/// Static Euler Dual-Graph Table for IBM Nighthawk R2 ($12 \times 10$ Grid).
#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(align(64))]
pub struct NighthawkDualGraph {
    pub plaquettes: [PlaquetteNode; NUM_PLAQUETTES],
    pub face_masks: [CouplerMask; NUM_PLAQUETTES],
    pub coupler_endpoints: [(u8, u8); NUM_COUPLERS],
}

impl NighthawkDualGraph {
    /// Compile-time generation of the 99-plaquette / 218-coupler topology.
    pub const fn new() -> Self {
        let mut plaquettes = [PlaquetteNode {
            edges: [0; 4],
            vertices: [0; 4],
        }; NUM_PLAQUETTES];
        let mut face_masks = [CouplerMask::empty(); NUM_PLAQUETTES];
        let mut coupler_endpoints = [(0u8, 0u8); NUM_COUPLERS];

        // 1. Populate Horizontal Couplers (0..110)
        let mut r = 0;
        while r < 10 {
            let mut c = 0;
            while c < 11 {
                let e_id = 11 * r + c;
                let v0 = (12 * r + c) as u8;
                let v1 = (12 * r + c + 1) as u8;
                coupler_endpoints[e_id] = (v0, v1);
                c += 1;
            }
            r += 1;
        }

        // 2. Populate Vertical Couplers (110..218)
        let mut r = 0;
        while r < 9 {
            let mut c = 0;
            while c < 12 {
                let e_id = 110 + 12 * r + c;
                let v0 = (12 * r + c) as u8;
                let v1 = (12 * (r + 1) + c) as u8;
                coupler_endpoints[e_id] = (v0, v1);
                c += 1;
            }
            r += 1;
        }

        // 3. Populate 99 Elementary Plaquettes
        let mut pr = 0;
        while pr < 9 {
            let mut pc = 0;
            while pc < 11 {
                let p_idx = 11 * pr + pc;
                let e_top = (11 * pr + pc) as u8;
                let e_bot = (11 * (pr + 1) + pc) as u8;
                let e_left = (110 + 12 * pr + pc) as u8;
                let e_right = (110 + 12 * pr + pc + 1) as u8;

                let v_tl = (12 * pr + pc) as u8;
                let v_tr = (12 * pr + pc + 1) as u8;
                let v_bl = (12 * (pr + 1) + pc) as u8;
                let v_br = (12 * (pr + 1) + pc + 1) as u8;

                plaquettes[p_idx] = PlaquetteNode {
                    edges: [e_top, e_bot, e_left, e_right],
                    vertices: [v_tl, v_tr, v_bl, v_br],
                };

                let mut mask = CouplerMask::empty();
                mask.set_bit(e_top as usize);
                mask.set_bit(e_bot as usize);
                mask.set_bit(e_left as usize);
                mask.set_bit(e_right as usize);
                face_masks[p_idx] = mask;

                pc += 1;
            }
            pr += 1;
        }

        Self {
            plaquettes,
            face_masks,
            coupler_endpoints,
        }
    }

    /// Contracts any subset of plaquettes via discrete 2D Stokes' theorem.
    /// Returns `(boundary_mask, canceled_spokes_mask, boundary_count, spoke_count)`.
    #[inline]
    pub fn contract_sublattice(
        &self,
        active_plaquettes: &[usize],
    ) -> (CouplerMask, CouplerMask, u32, u32) {
        let mut boundary = CouplerMask::empty();
        let mut touched = CouplerMask::empty();

        for &p_idx in active_plaquettes {
            let f_mask = &self.face_masks[p_idx];
            // XOR automatically cancels every internal spoke shared by 2 plaquettes (Z_e^2 = I)
            boundary = boundary.xor(f_mask);
            touched = touched.bitor(f_mask);
        }

        // Internal spokes are edges touched by the sublattice that vanished from the boundary
        let canceled_spokes = touched.bitand_not(&boundary);
        let boundary_count = boundary.count_ones();
        let spoke_count = canceled_spokes.count_ones();

        (boundary, canceled_spokes, boundary_count, spoke_count)
    }
}

impl Default for NighthawkDualGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Active L1-Resident Plaquette-Zipper Execution State (384 Bytes total).
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub struct PlaquetteZipperWindow {
    /// 3-qubit sliding dense complex statevector ($2^3 \times 16\text{ B} = 128\text{ bytes}$, SoA layout).
    pub window_re: [f64; 8],
    pub window_im: [f64; 8],
    /// Boundary stabilizer link signs (+1 / -1 bit-packed across 218 couplers = 32 bytes).
    pub boundary_flux_mask: CouplerMask,
    /// Canceled internal spoke mask (32 bytes).
    pub canceled_spoke_mask: CouplerMask,
    /// Plaquette Wilson loop expectation accumulators or active sub-tree indices (192 bytes).
    pub plaquette_parity: [u64; 24],
}

impl Default for PlaquetteZipperWindow {
    fn default() -> Self {
        Self {
            window_re: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            window_im: [0.0; 8],
            boundary_flux_mask: CouplerMask::empty(),
            canceled_spoke_mask: CouplerMask::empty(),
            plaquette_parity: [0; 24],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub const STATIC_DUAL_GRAPH: NighthawkDualGraph = NighthawkDualGraph::new();

    #[test]
    fn test_memory_layout_and_l1_cache_size() {
        assert_eq!(
            std::mem::size_of::<PlaquetteZipperWindow>(),
            384,
            "PlaquetteZipperWindow must be exactly 384 bytes (six 64-byte L1 cache lines)"
        );
        assert_eq!(
            std::mem::align_of::<PlaquetteZipperWindow>(),
            64,
            "PlaquetteZipperWindow must be 64-byte cache-line aligned"
        );
        assert_eq!(
            std::mem::size_of::<CouplerMask>(),
            32,
            "CouplerMask must be exactly 32 bytes (256 bits)"
        );
    }

    #[test]
    fn test_full_nighthawk_contraction_stokes_invariant() {
        let graph = &STATIC_DUAL_GRAPH;
        let all_plaquettes: Vec<usize> = (0..NUM_PLAQUETTES).collect();

        let (boundary, canceled_spokes, boundary_count, spoke_count) =
            graph.contract_sublattice(&all_plaquettes);

        // All 99 square plaquettes contract to 40 perimeter boundary links
        assert_eq!(boundary_count, 40, "Perimeter boundary links must equal 40");
        // All 178 internal shared spokes cancel identically (Z_e^2 = I)
        assert_eq!(spoke_count, 178, "Canceled internal spokes must equal 178");
        assert_eq!(
            boundary_count + spoke_count,
            NUM_COUPLERS as u32,
            "Total couplers must equal 218"
        );

        // Verify boundary and canceled spokes are strictly disjoint
        assert_eq!(
            boundary.0[0] & canceled_spokes.0[0],
            0,
            "Boundary and internal spokes must not overlap"
        );
        assert_eq!(boundary.0[1] & canceled_spokes.0[1], 0);
        assert_eq!(boundary.0[2] & canceled_spokes.0[2], 0);
        assert_eq!(boundary.0[3] & canceled_spokes.0[3], 0);
    }

    #[test]
    fn test_single_and_adjacent_plaquette_contractions() {
        let graph = &STATIC_DUAL_GRAPH;

        // Single plaquette P0: 4 boundary edges, 0 internal spokes
        let (_, _, b_single, s_single) = graph.contract_sublattice(&[0]);
        assert_eq!(b_single, 4);
        assert_eq!(s_single, 0);

        // Two adjacent plaquettes P0 and P1 (horizontal neighbors):
        // Share 1 vertical spoke -> 6 boundary edges, 1 canceled internal spoke
        let (_, _, b_pair, s_pair) = graph.contract_sublattice(&[0, 1]);
        assert_eq!(b_pair, 6);
        assert_eq!(s_pair, 1);

        // Four plaquettes in 2x2 square: [0, 1, 11, 12]:
        // Share 4 internal spokes meeting at central hub -> 8 boundary edges, 4 internal spokes
        let (_, _, b_2x2, s_2x2) = graph.contract_sublattice(&[0, 1, 11, 12]);
        assert_eq!(b_2x2, 8);
        assert_eq!(s_2x2, 4);
    }
}
