// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use num_complex::Complex64;
use std::collections::HashMap;

/// 2D Planar Lattice Topologies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LatticeTopology {
    /// 2D Cartesian Grid of dimension width x height (coordination number z <= 4).
    SquareGrid { width: usize, height: usize },
    /// Heavy-Hexagonal Lattice (coordination number z <= 3).
    /// Modeled as an Lx x Ly honeycomb with bridge qubits on horizontal/vertical edges.
    HeavyHex { width: usize, height: usize },
}

impl LatticeTopology {
    pub fn num_qubits(&self) -> usize {
        match self {
            Self::SquareGrid { width, height } => width * height,
            Self::HeavyHex { width, height } => {
                // Number of core vertices + bridge vertices along edges
                let core_nodes = width * height;
                let horizontal_bridges = (width.saturating_sub(1)) * height;
                let vertical_bridges = width * (height.saturating_sub(1));
                core_nodes + horizontal_bridges + vertical_bridges
            }
        }
    }

    pub fn neighbors(&self, site: usize) -> Vec<usize> {
        match self {
            Self::SquareGrid { width, height } => {
                let w = *width;
                let h = *height;
                if site >= w * h {
                    return Vec::new();
                }
                let x = site % w;
                let y = site / w;
                let mut nbrs = Vec::with_capacity(4);

                // West
                if x > 0 {
                    nbrs.push(y * w + (x - 1));
                }
                // East
                if x + 1 < w {
                    nbrs.push(y * w + (x + 1));
                }
                // North
                if y > 0 {
                    nbrs.push((y - 1) * w + x);
                }
                // South
                if y + 1 < h {
                    nbrs.push((y + 1) * w + x);
                }
                nbrs.sort_unstable();
                nbrs
            }
            Self::HeavyHex { width, height } => {
                let w = *width;
                let h = *height;
                let core_count = w * h;
                let mut nbrs = Vec::new();

                if site < core_count {
                    let x = site % w;
                    let y = site / w;
                    // Connect to horizontal bridge if exists
                    if x + 1 < w {
                        let h_bridge = core_count + (y * (w - 1) + x);
                        nbrs.push(h_bridge);
                    }
                    if x > 0 {
                        let h_bridge = core_count + (y * (w - 1) + (x - 1));
                        nbrs.push(h_bridge);
                    }
                    // Connect to vertical bridge on alternating columns (z <= 3)
                    if (x + y).is_multiple_of(2) && y + 1 < h {
                        let v_base = core_count + (w - 1) * h;
                        let v_bridge = v_base + (y * w + x);
                        nbrs.push(v_bridge);
                    } else if !(x + y).is_multiple_of(2) && y > 0 {
                        let v_base = core_count + (w - 1) * h;
                        let v_bridge = v_base + ((y - 1) * w + x);
                        nbrs.push(v_bridge);
                    }
                } else {
                    // It is a bridge qubit: connected to exactly two core qubits
                    let h_bridge_count = (w.saturating_sub(1)) * h;
                    if site < core_count + h_bridge_count {
                        let idx = site - core_count;
                        let y = idx / (w - 1);
                        let x = idx % (w - 1);
                        nbrs.push(y * w + x);
                        nbrs.push(y * w + (x + 1));
                    } else {
                        let idx = site - core_count - h_bridge_count;
                        let y = idx / w;
                        let x = idx % w;
                        nbrs.push(y * w + x);
                        nbrs.push((y + 1) * w + x);
                    }
                }
                nbrs.sort_unstable();
                nbrs
            }
        }
    }

    /// Sorted canonical edge list with u < v.
    pub fn edges(&self) -> Vec<(usize, usize)> {
        let n = self.num_qubits();
        let mut edge_map = Vec::new();
        for u in 0..n {
            for v in self.neighbors(u) {
                if u < v {
                    edge_map.push((u, v));
                }
            }
        }
        edge_map.sort_unstable();
        edge_map
    }

    pub fn max_coordination(&self) -> usize {
        match self {
            Self::SquareGrid { .. } => 4,
            Self::HeavyHex { .. } => 3,
        }
    }
}

/// A Site Tensor on vertex v of a 2D planar tensor network state.
/// Dimensions: [phys_dim=2, bond_dim_0, bond_dim_1, ..., bond_dim_{d-1}].
#[derive(Debug, Clone, PartialEq)]
pub struct SiteTensor {
    pub site_idx: usize,
    pub physical_dim: usize,
    pub neighbors: Vec<usize>,
    pub bond_dims: Vec<usize>,
    pub strides: Vec<usize>,
    pub data: Vec<Complex64>,
}

impl SiteTensor {
    /// Create product state (|0> or |1>) with all virtual bond dimensions set to 1.
    pub fn new_product_state(site_idx: usize, neighbors: Vec<usize>, bit: u8) -> Self {
        let d = neighbors.len();
        let physical_dim = 2;
        let bond_dims = vec![1; d];
        let mut strides = vec![1; d + 1];

        // Row-major strides
        let mut cur_stride = 1;
        for i in (0..d).rev() {
            strides[i + 1] = cur_stride;
            cur_stride *= bond_dims[i];
        }
        strides[0] = cur_stride;

        let total_elems = physical_dim * cur_stride;
        let mut data = vec![Complex64::new(0.0, 0.0); total_elems];

        // State initialization: bit=0 -> amplitude at phys=0 is 1.0; bit=1 -> phys=1 is 1.0
        let target_phys = if bit == 0 { 0 } else { 1 };
        data[target_phys * strides[0]] = Complex64::new(1.0, 0.0);

        Self {
            site_idx,
            physical_dim,
            neighbors,
            bond_dims,
            strides,
            data,
        }
    }

    pub fn recompute_strides(&mut self) {
        let d = self.bond_dims.len();
        let mut strides = vec![1; d + 1];
        let mut cur_stride = 1;
        for i in (0..d).rev() {
            strides[i + 1] = cur_stride;
            cur_stride *= self.bond_dims[i];
        }
        strides[0] = cur_stride;
        self.strides = strides;
    }

    #[inline(always)]
    pub fn flat_index(&self, phys_idx: usize, bond_indices: &[usize]) -> usize {
        debug_assert_eq!(bond_indices.len(), self.bond_dims.len());
        let mut idx = phys_idx * self.strides[0];
        for (i, &b_idx) in bond_indices.iter().enumerate() {
            idx += b_idx * self.strides[i + 1];
        }
        idx
    }

    #[inline(always)]
    pub fn get_elem(&self, phys_idx: usize, bond_indices: &[usize]) -> Complex64 {
        self.data[self.flat_index(phys_idx, bond_indices)]
    }

    #[inline(always)]
    pub fn set_elem(&mut self, phys_idx: usize, bond_indices: &[usize], val: Complex64) {
        let idx = self.flat_index(phys_idx, bond_indices);
        self.data[idx] = val;
    }

    pub fn neighbor_bond_dim(&self, nbr: usize) -> Option<usize> {
        self.neighbors
            .iter()
            .position(|&x| x == nbr)
            .map(|pos| self.bond_dims[pos])
    }

    pub fn neighbor_bond_pos(&self, nbr: usize) -> Option<usize> {
        self.neighbors.iter().position(|&x| x == nbr)
    }

    pub fn total_elements(&self) -> usize {
        self.data.len()
    }

    pub fn memory_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<Complex64>()
            + self.neighbors.len() * std::mem::size_of::<usize>()
            + self.bond_dims.len() * std::mem::size_of::<usize>()
            + self.strides.len() * std::mem::size_of::<usize>()
    }
}

/// Diagonal Bond Matrix Lambda_e on edge e = (u, v).
/// Stores singular values / message root weights.
#[derive(Debug, Clone, PartialEq)]
pub struct BondTensor {
    pub edge: (usize, usize),
    pub weights: Vec<f64>,
}

impl BondTensor {
    pub fn new_trivial(edge: (usize, usize)) -> Self {
        assert!(edge.0 < edge.1, "canonical edge must have u < v");
        Self {
            edge,
            weights: vec![1.0],
        }
    }

    pub fn dim(&self) -> usize {
        self.weights.len()
    }

    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.weights.iter().map(|&w| w * w).sum();
        if norm_sq > 1e-28 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for w in &mut self.weights {
                *w *= inv_norm;
            }
        }
    }

    pub fn entanglement_entropy(&self) -> f64 {
        let norm_sq: f64 = self.weights.iter().map(|&w| w * w).sum();
        if norm_sq < 1e-28 {
            return 0.0;
        }
        let inv_norm_sq = 1.0 / norm_sq;
        let mut s = 0.0;
        for &w in &self.weights {
            let p = (w * w) * inv_norm_sq;
            if p > 1e-15 {
                s -= p * p.ln();
            }
        }
        s
    }

    pub fn memory_bytes(&self) -> usize {
        self.weights.len() * std::mem::size_of::<f64>()
    }
}

/// Planar 2D Tensor Network State matching physical hardware topology.
#[derive(Debug, Clone)]
pub struct PlanarTensorNetwork {
    pub topology: LatticeTopology,
    pub sites: Vec<SiteTensor>,
    pub bonds: HashMap<(usize, usize), BondTensor>,
    pub max_bond_dim: usize,
}

impl PlanarTensorNetwork {
    /// Initialize product state |0>^N with bond dimensions 1.
    pub fn new_product_state(topology: LatticeTopology, max_bond_dim: usize) -> Self {
        let n = topology.num_qubits();
        let mut sites = Vec::with_capacity(n);
        let mut bonds = HashMap::new();

        for site_idx in 0..n {
            let nbrs = topology.neighbors(site_idx);
            sites.push(SiteTensor::new_product_state(site_idx, nbrs, 0));
        }

        for (u, v) in topology.edges() {
            bonds.insert((u, v), BondTensor::new_trivial((u, v)));
        }

        Self {
            topology,
            sites,
            bonds,
            max_bond_dim,
        }
    }

    pub fn num_qubits(&self) -> usize {
        self.sites.len()
    }

    pub fn site(&self, site_idx: usize) -> &SiteTensor {
        &self.sites[site_idx]
    }

    pub fn site_mut(&mut self, site_idx: usize) -> &mut SiteTensor {
        &mut self.sites[site_idx]
    }

    pub fn bond(&self, u: usize, v: usize) -> Option<&BondTensor> {
        let key = if u < v { (u, v) } else { (v, u) };
        self.bonds.get(&key)
    }

    pub fn bond_mut(&mut self, u: usize, v: usize) -> Option<&mut BondTensor> {
        let key = if u < v { (u, v) } else { (v, u) };
        self.bonds.get_mut(&key)
    }

    /// Total memory footprint across all site and bond tensors.
    pub fn memory_bytes(&self) -> usize {
        let sites_mem: usize = self.sites.iter().map(|s| s.memory_bytes()).sum();
        let bonds_mem: usize = self.bonds.values().map(|b| b.memory_bytes()).sum();
        sites_mem + bonds_mem
    }

    /// Verifies structural and dimensional consistency along all edges.
    pub fn validate_integrity(&self) -> Result<(), String> {
        for (&(u, v), bond) in &self.bonds {
            let dim_u = self.sites[u]
                .neighbor_bond_dim(v)
                .ok_or_else(|| format!("Site {u} missing neighbor {v}"))?;
            let dim_v = self.sites[v]
                .neighbor_bond_dim(u)
                .ok_or_else(|| format!("Site {v} missing neighbor {u}"))?;

            if dim_u != dim_v {
                return Err(format!(
                    "Bond dimension mismatch on edge ({u}, {v}): site {u} has {dim_u}, site {v} has {dim_v}"
                ));
            }
            if dim_u != bond.dim() {
                return Err(format!(
                    "Bond tensor dimension mismatch on edge ({u}, {v}): site has {dim_u}, bond has {}",
                    bond.dim()
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_grid_topology_4x4() {
        let topo = LatticeTopology::SquareGrid {
            width: 4,
            height: 4,
        };
        assert_eq!(topo.num_qubits(), 16);
        assert_eq!(topo.neighbors(0), vec![1, 4]); // Corner (deg 2)
        assert_eq!(topo.neighbors(1), vec![0, 2, 5]); // Edge (deg 3)
        assert_eq!(topo.neighbors(5), vec![1, 4, 6, 9]); // Center (deg 4)
        assert_eq!(topo.edges().len(), 24); // 4*3 + 4*3 = 24 edges
    }

    #[test]
    fn test_heavy_hex_topology_bounds() {
        let topo = LatticeTopology::HeavyHex {
            width: 3,
            height: 3,
        };
        let n = topo.num_qubits();
        assert!(n > 9);
        for i in 0..n {
            let deg = topo.neighbors(i).len();
            assert!(
                deg <= 3,
                "heavy-hex site {i} exceeds max coordination 3 (got {deg})"
            );
        }
    }

    #[test]
    fn test_planar_tensor_network_initialization_36_qubits() {
        let topo = LatticeTopology::SquareGrid {
            width: 6,
            height: 6,
        };
        let tn = PlanarTensorNetwork::new_product_state(topo, 16);
        assert_eq!(tn.num_qubits(), 36);
        assert_eq!(tn.bonds.len(), 60); // 6*5 + 6*5 = 60 edges

        // Check memory consumption for initial state
        let mem = tn.memory_bytes();
        assert!(
            mem < 50_000,
            "Initial product state memory must be < 50 KB (was {mem} bytes)"
        );
        assert!(tn.validate_integrity().is_ok());

        // Verify state is |0>
        for site in &tn.sites {
            assert_eq!(
                site.get_elem(0, &vec![0; site.neighbors.len()]),
                Complex64::new(1.0, 0.0)
            );
            assert_eq!(
                site.get_elem(1, &vec![0; site.neighbors.len()]),
                Complex64::new(0.0, 0.0)
            );
        }
    }
}
