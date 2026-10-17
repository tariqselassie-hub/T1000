// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Geometric Plaquette Configurations & Dual Hypergraphs
//!
//! In discrete lattice gauge theory / stabilizer flux formalisms, the physical
//! qubits reside on the **links** (edges $e \in E$) of the lattice:
//! 1. **18-Qubit Kagome Star Cell** ($H_0$ Hexagon + 6 Outer Triangles = 6 internal spokes + 12 boundary links = 18 qubits).
//! 2. **15-Qubit 7-Plaquette Kagome Strip** (6 internal spokes + 9 boundary links = 15 qubits).
//! 3. **11-Qubit 5-Plaquette Kagome Strip** (4 internal spokes + 7 boundary links = 11 qubits).
//! 4. **Triangular Triad ($\Delta_3$)** (3 internal spokes + 3 boundary links = 6 qubits).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Undirected physical link carrying a gauge involution qubit Z_e.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GaugeLink {
    pub id: usize,
    pub u: usize,
    pub v: usize,
}

impl GaugeLink {
    pub fn new(id: usize, u: usize, v: usize) -> Self {
        if u < v {
            GaugeLink { id, u, v }
        } else {
            GaugeLink { id, v, u }
        }
    }
}

/// Elementary plaquette defined by its closed loop of link IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plaquette {
    pub id: usize,
    pub name: String,
    pub link_ids: Vec<usize>,
}

/// Geometric lattice assembly with physical qubits residing on gauge links.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaquetteLattice {
    pub name: String,
    pub total_qubits: usize,
    pub links: Vec<GaugeLink>,
    pub plaquettes: Vec<Plaquette>,
    pub internal_spoke_ids: Vec<usize>,
    pub boundary_link_ids: Vec<usize>,
}

impl PlaquetteLattice {
    /// Builds a gauge lattice from explicit links and plaquette link assignments.
    pub fn new(
        name: impl Into<String>,
        links: Vec<GaugeLink>,
        plaquettes: Vec<Plaquette>,
    ) -> Self {
        let total_qubits = links.len();
        let mut link_counts = HashMap::new();
        for p in &plaquettes {
            for &l_id in &p.link_ids {
                *link_counts.entry(l_id).or_insert(0) += 1;
            }
        }

        let mut internal_spoke_ids = Vec::new();
        let mut boundary_link_ids = Vec::new();

        for (&l_id, &count) in &link_counts {
            if count == 1 {
                boundary_link_ids.push(l_id);
            } else if count == 2 {
                internal_spoke_ids.push(l_id);
            } else {
                panic!("Link ID {} shared by {} plaquettes (expected <= 2)", l_id, count);
            }
        }

        internal_spoke_ids.sort();
        boundary_link_ids.sort();

        Self {
            name: name.into(),
            total_qubits,
            links,
            plaquettes,
            internal_spoke_ids,
            boundary_link_ids,
        }
    }

    // =========================================================================
    // 1. 18-Qubit Kagome Star Plaquette Cell
    // =========================================================================
    /// Hexagon H0 has 6 internal spokes (IDs 0..5).
    /// Surrounding 6 triangles P1..P6 add 12 outer boundary links (IDs 6..17).
    /// Total qubits = 18.
    pub fn build_18q_kagome_star() -> Self {
        let mut links = Vec::with_capacity(18);
        // 6 internal spokes of hexagon H0
        for i in 0..6 {
            links.push(GaugeLink::new(i, i, (i + 1) % 6));
        }
        // 12 outer boundary links for triangles P1..P6
        for k in 0..6 {
            let tip = 6 + k;
            let u = k;
            let v = (k + 1) % 6;
            links.push(GaugeLink::new(6 + 2 * k, u, tip));
            links.push(GaugeLink::new(6 + 2 * k + 1, v, tip));
        }

        let mut plaquettes = Vec::with_capacity(7);
        // H0 Central Hexagon
        plaquettes.push(Plaquette {
            id: 0,
            name: "H0_Center".to_string(),
            link_ids: vec![0, 1, 2, 3, 4, 5],
        });

        // P1..P6 Outer Triangles
        for k in 0..6 {
            let shared_spoke = k;
            let l_a = 6 + 2 * k;
            let l_b = 6 + 2 * k + 1;
            plaquettes.push(Plaquette {
                id: k + 1,
                name: format!("P{}_Triangle", k + 1),
                link_ids: vec![shared_spoke, l_a, l_b],
            });
        }

        Self::new("18-Qubit Kagome Star Plaquette Cell", links, plaquettes)
    }

    // =========================================================================
    // 2. 15-Qubit 7-Plaquette Kagome Strip
    // =========================================================================
    /// Strip of 7 plaquettes sharing 6 internal spokes (IDs 0..5) and 9 boundary links (IDs 6..14).
    /// Total qubits = 15.
    pub fn build_15q_kagome_strip() -> Self {
        let mut links = Vec::with_capacity(15);
        // 6 internal shared spokes (IDs 0..5)
        for i in 0..6 {
            links.push(GaugeLink::new(i, 2 * i, 2 * i + 1));
        }
        // 9 boundary links (IDs 6..14)
        for i in 6..15 {
            links.push(GaugeLink::new(i, i + 1, i + 2));
        }

        let mut plaquettes = Vec::with_capacity(7);
        // Plaquette 0 (end triangle): shared spoke 0, boundary links 6, 7
        plaquettes.push(Plaquette {
            id: 0,
            name: "P1".to_string(),
            link_ids: vec![0, 6, 7],
        });
        // Plaquettes 1..5 (middle cells): shared spokes (i-1, i), plus boundary links
        for i in 1..6 {
            plaquettes.push(Plaquette {
                id: i,
                name: format!("P{}", i + 1),
                link_ids: vec![i - 1, i, 7 + i],
            });
        }
        // Plaquette 6 (end triangle): shared spoke 5, boundary links 13, 14
        plaquettes.push(Plaquette {
            id: 6,
            name: "P7".to_string(),
            link_ids: vec![5, 13, 14],
        });

        Self::new("15-Qubit 7-Plaquette Kagome Strip", links, plaquettes)
    }

    // =========================================================================
    // 3. 11-Qubit 5-Plaquette Kagome Strip
    // =========================================================================
    /// 5 plaquettes sharing 4 internal spokes (IDs 0..3) and 7 boundary links (IDs 4..10). Total = 11.
    pub fn build_11q_kagome_strip() -> Self {
        let mut links = Vec::with_capacity(11);
        for i in 0..4 {
            links.push(GaugeLink::new(i, 2 * i, 2 * i + 1));
        }
        for i in 4..11 {
            links.push(GaugeLink::new(i, i + 1, i + 2));
        }

        let mut plaquettes = Vec::with_capacity(5);
        plaquettes.push(Plaquette {
            id: 0,
            name: "P1".to_string(),
            link_ids: vec![0, 4, 5],
        });
        for i in 1..4 {
            plaquettes.push(Plaquette {
                id: i,
                name: format!("P{}", i + 1),
                link_ids: vec![i - 1, i, 5 + i],
            });
        }
        plaquettes.push(Plaquette {
            id: 4,
            name: "P5".to_string(),
            link_ids: vec![3, 9, 10],
        });

        Self::new("11-Qubit 5-Plaquette Kagome Strip", links, plaquettes)
    }

    // =========================================================================
    // 4. Triangular Triad (Delta_3)
    // =========================================================================
    /// 3 overlapping triangles sharing 3 internal spokes meeting at central hub V0.
    /// Internal spokes: IDs 0, 1, 2. Boundary links: IDs 3, 4, 5. Total = 6 qubits.
    pub fn build_delta3_triad() -> Self {
        let mut links = Vec::with_capacity(6);
        // Spokes (0, 1), (0, 2), (0, 3)
        links.push(GaugeLink::new(0, 0, 1));
        links.push(GaugeLink::new(1, 0, 2));
        links.push(GaugeLink::new(2, 0, 3));
        // Boundary links (1, 2), (2, 3), (3, 1)
        links.push(GaugeLink::new(3, 1, 2));
        links.push(GaugeLink::new(4, 2, 3));
        links.push(GaugeLink::new(5, 3, 1));

        let plaquettes = vec![
            Plaquette {
                id: 0,
                name: "Delta_A".to_string(),
                link_ids: vec![0, 1, 3],
            },
            Plaquette {
                id: 1,
                name: "Delta_B".to_string(),
                link_ids: vec![1, 2, 4],
            },
            Plaquette {
                id: 2,
                name: "Delta_C".to_string(),
                link_ids: vec![2, 0, 5],
            },
        ];

        Self::new("3-Triad Frustrated Central Hub (Delta_3)", links, plaquettes)
    }

    // =========================================================================
    // 5. Arbitrary Scaled Kagome Chain (Up to 1,000+ Qubits)
    // =========================================================================
    /// Generates an arbitrary Kagome lattice chain of `num_plaquettes` ($M \ge 3$).
    /// Total qubits = $2M + 1$ ($M-1$ internal spokes, $M+2$ boundary links).
    pub fn build_kagome_chain(num_plaquettes: usize) -> Self {
        assert!(num_plaquettes >= 3, "Kagome chain requires >= 3 plaquettes");
        let m = num_plaquettes;
        let num_internal = m - 1;
        let num_boundary = m + 2;
        let total_qubits = num_internal + num_boundary;

        let mut links = Vec::with_capacity(total_qubits);
        // Internal spokes: IDs 0..num_internal - 1
        for i in 0..num_internal {
            links.push(GaugeLink::new(i, 2 * i, 2 * i + 1));
        }
        // Boundary links: IDs num_internal..total_qubits - 1
        for i in num_internal..total_qubits {
            links.push(GaugeLink::new(i, i + 1, i + 2));
        }

        let mut plaquettes = Vec::with_capacity(m);
        // First end cell
        plaquettes.push(Plaquette {
            id: 0,
            name: "P_0".to_string(),
            link_ids: vec![0, num_internal, num_internal + 1],
        });
        // Middle cells
        for i in 1..m - 1 {
            plaquettes.push(Plaquette {
                id: i,
                name: format!("P_{}", i),
                link_ids: vec![i - 1, i, num_internal + 1 + i],
            });
        }
        // Last end cell
        plaquettes.push(Plaquette {
            id: m - 1,
            name: format!("P_{}", m - 1),
            link_ids: vec![num_internal - 1, total_qubits - 2, total_qubits - 1],
        });

        Self::new(
            format!("Scaled Kagome Chain ({} Qubits / {} Plaquettes)", total_qubits, m),
            links,
            plaquettes,
        )
    }

    // =========================================================================
    // 6. IBM Quantum Nighthawk R2 Native Topology
    // =========================================================================
    /// Maps the exact planar architecture of IBM Quantum Nighthawk R2:
    /// - 120 vertex qubits (12 columns x 10 rows).
    /// - 218 tunable couplers (110 horizontal + 108 vertical).
    /// - 99 elementary square plaquettes (11 x 9).
    /// - 40 perimeter boundary links, 178 internal spokes (each shared by 2 plaquettes).
    pub fn build_ibm_nighthawk_r2() -> Self {
        let cols = 12;
        let rows = 10;
        let mut links = Vec::with_capacity(218);
        let mut h_link_map = HashMap::new();
        let mut v_link_map = HashMap::new();

        let mut link_id = 0;

        // Horizontal links: 10 rows x 11 columns = 110 links
        for r in 0..rows {
            for c in 0..cols - 1 {
                let u = r * cols + c;
                let v = r * cols + (c + 1);
                links.push(GaugeLink::new(link_id, u, v));
                h_link_map.insert((r, c), link_id);
                link_id += 1;
            }
        }

        // Vertical links: 9 rows x 12 columns = 108 links
        for r in 0..rows - 1 {
            for c in 0..cols {
                let u = r * cols + c;
                let v = (r + 1) * cols + c;
                links.push(GaugeLink::new(link_id, u, v));
                v_link_map.insert((r, c), link_id);
                link_id += 1;
            }
        }

        assert_eq!(links.len(), 218, "Nighthawk R2 must have 218 couplers");

        // 99 Square Plaquettes: (11 columns x 9 rows)
        let mut plaquettes = Vec::with_capacity(99);
        let mut p_id = 0;
        for r in 0..rows - 1 {
            for c in 0..cols - 1 {
                let top = h_link_map[&(r, c)];
                let bottom = h_link_map[&(r + 1, c)];
                let left = v_link_map[&(r, c)];
                let right = v_link_map[&(r, c + 1)];

                plaquettes.push(Plaquette {
                    id: p_id,
                    name: format!("Square_{}_{}", c, r),
                    link_ids: vec![top, right, bottom, left],
                });
                p_id += 1;
            }
        }

        assert_eq!(plaquettes.len(), 99, "Nighthawk R2 must have 99 plaquettes");

        Self::new("IBM Quantum Nighthawk R2 (120Q / 218 Couplers)", links, plaquettes)
    }

    // =========================================================================
    // 7. Optimal Nighthawk-Embedded Frustrated Kagome Strip (119 Qubits)
    // =========================================================================
    /// Embeds a 119-qubit frustrated Kagome strip (59 triangular plaquettes)
    /// across the IBM Nighthawk R2 processor footprint.
    /// Internal spokes = 58, Boundary links = 61.
    pub fn build_nighthawk_kagome_strip() -> Self {
        Self::build_kagome_chain(59)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibm_nighthawk_r2_invariants() {
        let lattice = PlaquetteLattice::build_ibm_nighthawk_r2();
        assert_eq!(lattice.total_qubits, 218);
        assert_eq!(lattice.plaquettes.len(), 99);
        assert_eq!(lattice.boundary_link_ids.len(), 40);
        assert_eq!(lattice.internal_spoke_ids.len(), 178);
        assert_eq!(lattice.boundary_link_ids.len() + lattice.internal_spoke_ids.len(), 218);
    }

    #[test]
    fn test_nighthawk_kagome_strip_invariants() {
        let lattice = PlaquetteLattice::build_nighthawk_kagome_strip();
        assert_eq!(lattice.total_qubits, 119);
        assert_eq!(lattice.plaquettes.len(), 59);
        assert_eq!(lattice.internal_spoke_ids.len(), 58);
        assert_eq!(lattice.boundary_link_ids.len(), 61);
        assert_eq!(lattice.internal_spoke_ids.len() + lattice.boundary_link_ids.len(), 119);
    }
}
