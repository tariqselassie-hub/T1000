// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Molecular Graph representation and OpenSMILES topology compiler for Geometric QGNN.
//!
//! Maps chemical molecules directly into graph topological representations suitable for
//! Lie-algebraic equivariant quantum graph convolutions.

use crate::smiles::{parse, BondKind, SmilesError};
use serde::{Deserialize, Serialize};

/// Number of numerical features encoded per graph node (atom).
pub const NODE_FEATURE_DIM: usize = 4;

/// Feature vector describing a single atom node.
/// [atomic_number_normalized, formal_charge, is_aromatic, connectivity_degree_normalized]
pub type NodeFeatures = [f64; NODE_FEATURE_DIM];

/// An undirected chemical bond connecting two atom nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphEdge {
    pub u: usize,
    pub v: usize,
    pub kind: BondKind,
}

impl GraphEdge {
    #[inline(always)]
    pub fn new(u: usize, v: usize, kind: BondKind) -> Self {
        Self {
            u: u.min(v),
            v: u.max(v),
            kind,
        }
    }

    /// Discrete bond order index for parameter sharing (0: Single, 1: Double, 2: Triple, 3: Aromatic).
    #[inline(always)]
    pub fn bond_class(&self) -> usize {
        match self.kind {
            BondKind::Single | BondKind::Up | BondKind::Down => 0,
            BondKind::Double => 1,
            BondKind::Triple => 2,
            BondKind::Aromatic => 3,
            BondKind::Quadruple => 1,
        }
    }
}

/// A molecular graph topology suitable for quantum Hilbert space mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MolecularGraph {
    /// Number of active atom vertices in the graph (corresponds to qubit count).
    pub num_nodes: usize,
    /// Per-node continuous feature vectors.
    pub node_features: Vec<NodeFeatures>,
    /// Unique chemical bonds in the molecule.
    pub edges: Vec<GraphEdge>,
    /// Element symbols for human-readable debugging.
    pub symbols: Vec<String>,
}

impl MolecularGraph {
    /// Constructs a molecular graph with specified node count and empty topology.
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            node_features: vec![[0.0; NODE_FEATURE_DIM]; num_nodes],
            edges: Vec::new(),
            symbols: vec!["C".to_string(); num_nodes],
        }
    }

    /// Compiles an OpenSMILES string into a MolecularGraph with at most `max_nodes` vertices.
    ///
    /// If the molecule has more than `max_nodes` heavy atoms, non-aromatic terminal leaves are
    /// contracted to preserve the maximum conjugated / core scaffold topology.
    pub fn from_smiles(smiles: &str, max_nodes: usize) -> Result<Self, SmilesError> {
        let parsed = parse(smiles)?;
        let total_atoms = parsed.atoms.len();
        if total_atoms == 0 {
            return Err(SmilesError::Empty);
        }

        let target_nodes = total_atoms.min(max_nodes);
        let mut degree = vec![0usize; total_atoms];
        for &(a, b, _) in &parsed.bonds {
            if a < total_atoms && b < total_atoms {
                degree[a] += 1;
                degree[b] += 1;
            }
        }

        // Rank atoms by structural priority: aromatic > high degree > high atomic number
        let mut ranked_indices: Vec<usize> = (0..total_atoms).collect();
        ranked_indices.sort_by(|&a, &b| {
            let score_a = (parsed.atoms[a].aromatic as usize) * 100
                + degree[a] * 10
                + atomic_number(&parsed.atoms[a].element);
            let score_b = (parsed.atoms[b].aromatic as usize) * 100
                + degree[b] * 10
                + atomic_number(&parsed.atoms[b].element);
            score_b.cmp(&score_a)
        });

        let selected: Vec<usize> = ranked_indices.into_iter().take(target_nodes).collect();
        let mut old_to_new = vec![None; total_atoms];
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            old_to_new[old_idx] = Some(new_idx);
        }

        let mut node_features = Vec::with_capacity(target_nodes);
        let mut symbols = Vec::with_capacity(target_nodes);
        for &old_idx in &selected {
            let atom = &parsed.atoms[old_idx];
            let z = atomic_number(&atom.element) as f64;
            let norm_z = z / 53.0; // Normalized against Iodine
            let charge = atom.charge as f64;
            let arom = if atom.aromatic { 1.0 } else { 0.0 };
            let deg = (degree[old_idx] as f64) / 4.0;
            node_features.push([norm_z, charge, arom, deg]);
            symbols.push(atom.element.clone());
        }

        let mut edges = Vec::new();
        for &(a, b, kind) in &parsed.bonds {
            if let (Some(new_a), Some(new_b)) = (old_to_new[a], old_to_new[b]) {
                if new_a != new_b {
                    let edge = GraphEdge::new(new_a, new_b, kind);
                    if !edges.contains(&edge) {
                        edges.push(edge);
                    }
                }
            }
        }

        Ok(Self {
            num_nodes: target_nodes,
            node_features,
            edges,
            symbols,
        })
    }

    /// Permutes the graph vertices by permutation `p` where `p[old] = new`.
    /// Used for testing permutation equivariance of the QGNN.
    pub fn permute(&self, p: &[usize]) -> Self {
        assert_eq!(p.len(), self.num_nodes);
        let mut new_features = vec![[0.0; NODE_FEATURE_DIM]; self.num_nodes];
        let mut new_symbols = vec![String::new(); self.num_nodes];
        for (old_i, &new_i) in p.iter().enumerate() {
            new_features[new_i] = self.node_features[old_i];
            new_symbols[new_i] = self.symbols[old_i].clone();
        }

        let mut new_edges = Vec::with_capacity(self.edges.len());
        for edge in &self.edges {
            new_edges.push(GraphEdge::new(p[edge.u], p[edge.v], edge.kind));
        }
        new_edges.sort_by(|a, b| (a.u, a.v).cmp(&(b.u, b.v)));

        Self {
            num_nodes: self.num_nodes,
            node_features: new_features,
            edges: new_edges,
            symbols: new_symbols,
        }
    }
}

/// Helper mapping element symbol to standard atomic number Z.
fn atomic_number(symbol: &str) -> usize {
    match symbol {
        "H" => 1,
        "B" => 5,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "Br" => 35,
        "I" => 53,
        _ => 6,
    }
}
