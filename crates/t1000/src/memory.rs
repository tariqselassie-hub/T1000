use crate::state::QuantumState;
use serde::{Deserialize, Serialize};
use t1000_core::{simulated_bifurcation, BifurcationConfig};

/// Memory entry stored in the Quantum Associative Memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub title: String,
    pub category: String,
    pub token_count: usize,
    pub quantum_state: QuantumState,
}

/// Quantum Associative Memory (QAM) for holographic indexing and ground-state recall.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumAssociativeMemory {
    pub dim: usize,
    pub entries: Vec<MemoryEntry>,
}

impl QuantumAssociativeMemory {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            entries: Vec::new(),
        }
    }

    /// Insert or update a memory item.
    pub fn insert(
        &mut self,
        id: String,
        title: String,
        category: String,
        token_count: usize,
        quantum_state: QuantumState,
    ) {
        assert_eq!(self.dim, quantum_state.dim);
        self.entries.push(MemoryEntry {
            id,
            title,
            category,
            token_count,
            quantum_state,
        });
    }

    /// Linear quantum fidelity scan ranking all entries by |<query | entry>|^2.
    pub fn recall_fidelity(&self, query: &QuantumState, top_k: usize) -> Vec<(&MemoryEntry, f64)> {
        assert_eq!(self.dim, query.dim);
        let mut scored: Vec<(&MemoryEntry, f64)> = self
            .entries
            .iter()
            .map(|entry| {
                let fid = query.fidelity(&entry.quantum_state);
                (entry, fid)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Adiabatic Simulated Bifurcation ground-state search for optimal associative recall.
    ///
    /// **Experimental.** This is a stochastic search over fidelities that are already fully
    /// computed, so it can at best match the exact argmax from [`Self::recall_fidelity`] and
    /// may return a worse entry. Prefer `recall_fidelity` when correctness matters.
    /// Maps energy landscape E_i = 1.0 - fidelity(query, entry_i) to locate the ground state.
    pub fn recall_bifurcation(
        &self,
        query: &QuantumState,
        seed: u64,
    ) -> Option<(&MemoryEntry, f64, f64)> {
        if self.entries.is_empty() {
            return None;
        }

        let n = self.entries.len();
        // Compute raw fidelities: |<query | entry>|^2
        let fidelities: Vec<f64> = self
            .entries
            .iter()
            .map(|e| query.fidelity(&e.quantum_state))
            .collect();

        let max_f = fidelities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_f = fidelities.iter().copied().fold(f64::INFINITY, f64::min);
        let spread = (max_f - min_f).max(1e-9);

        // Min-Max Contrast Normalization:
        // Highest fidelity maps to E = 0.0 (ground state minimum)
        // Lowest fidelity maps to E = 1.0 (highest excitation)
        let energies: Vec<f64> = fidelities
            .iter()
            .map(|&f| (1.0 - (f - min_f) / spread).clamp(0.0, 1.0))
            .collect();

        let cfg = BifurcationConfig {
            basis: n,
            seed,
            max_iters: 25,
            gamma: 2.8,
            dt: 0.25,
            collapse_seed: seed.wrapping_add(7),
        };

        let result = simulated_bifurcation(&energies, cfg);
        let winner = &self.entries[result.winner_index];
        let winner_fid = query.fidelity(&winner.quantum_state);

        Some((winner, winner_fid, result.winner_probability))
    }
}
