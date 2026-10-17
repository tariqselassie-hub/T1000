//! **Experimental.** Planar tensor-network backend (PEPS-style sites, SVD gate truncation,
//! belief-propagation gauging, boundary contraction).
//!
//! Known limitations; do not treat results as validated:
//! - `compute_amplitude` on a square grid keeps a dense boundary vector of size chi^height.
//!   `BoundaryMpsConfig::max_boundary_dim` only zeroes small entries (uncontrolled error,
//!   no memory saving); it is not a truncated boundary MPS, and `BoundaryMps` is unused.
//! - Heavy-hex amplitudes are only implemented for product states (all bond dims 1) and
//!   panic otherwise.
//! - The hand-written Jacobi SVD is slow; contraction cost grows quickly past ~9 qubits.
//! - No reference-oracle test coverage beyond 2-qubit Bell states.

pub mod bmps;
pub mod bp_gauge;
pub mod evolution;
pub mod linalg;
pub mod state;

pub use bmps::{
    compute_amplitude, compute_probability, sample_batch, sample_bitstring, BoundaryMps,
    BoundaryMpsConfig, SampleBatch,
};
pub use bp_gauge::{gauge_into_vidal, run_belief_propagation, BpConfig, BpMessageSet};
pub use evolution::{
    apply_single_qubit_gate, apply_two_qubit_gate, EvolutionTelemetry, SingleQubitGate,
    TwoQubitGate,
};
pub use linalg::{qr, svd, ComplexMatrix, SvdResult};
pub use state::{BondTensor, LatticeTopology, PlanarTensorNetwork, SiteTensor};
