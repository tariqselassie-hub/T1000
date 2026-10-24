// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Ab initio molecular VQE: Gaussian integrals, Hartree-Fock, the Jordan-Wigner qubit
//! Hamiltonian, exact diagonalisation, and a UCCSD variational solver.

pub mod basis;
pub mod boys;
pub mod circuit;
pub mod driver;
pub mod grouping;
pub mod fermion;
pub mod integrals;
pub mod lanczos;
pub mod linalg;
pub mod mo;
pub mod molecules;
pub mod noise;
pub mod noisy_vqe;
pub mod optimize;
pub mod pauli;
pub mod scf;
pub mod uccsd;
