// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

#![warn(unexpected_cfgs)]

pub mod channel;
pub mod interference;
pub mod projector;
pub mod register;
pub mod tensor;
pub mod porter_thomas;

pub use channel::{apply, ChannelKind, DecoherenceChannel};
pub use interference::{constructive_score, destructive_score, net_interference};
pub use projector::{collapse_with_seed, measure_all, Projector};
pub use register::{infer_radix_and_sites, QuditRadix, QuditRegister, QubitRegister};
pub use tensor::{EntanglementTensor, Side};

pub mod hamiltonian;
pub mod partial_trace;
pub mod permutation;
pub mod phase_gate;

pub use hamiltonian::{hamiltonian_time_evolve, Hamiltonian};
pub use partial_trace::PartialTrace;
pub use permutation::{permutation_apply, Permutation};
pub use phase_gate::{phase_gate_apply, PhaseGate};
pub mod povm;
pub use povm::{povm_measure_soft, POVM};

pub mod simulated_bifurcation;
pub use simulated_bifurcation::{bifurcation_interference, BifurcationConfig, BifurcationResult, simulated_bifurcation};

pub use porter_thomas::{
    effective_participation_ratio, exact_linear_xeb, expected_haar_entropy, expected_sum_p_sqr,
    haar_entropy_deficit, kolmogorov_smirnov_porter_thomas, naive_linear_xeb,
    sample_ideal_porter_thomas, shannon_entropy, verify_porter_thomas, KSResult,
    PorterThomasReport, ASYMPTOTIC_HAAR_ENTROPY_DEFICIT, EULER_MASCHERONI,
};
