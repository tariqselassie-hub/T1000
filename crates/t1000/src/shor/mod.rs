// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Shor's Order Finding & Prime Factorization Subsystem
//!
//! Provides the complete, reversible quantum implementation of Shor's algorithm
//! (Target 02 from *The Compendium of Super-Quadratic Computational Quantum Advantages*,
//! arXiv:2610.32595).
//!
//! Features:
//! - Reversible modular arithmetic & permutation multipliers $U_a |x\rangle = |a \cdot x \pmod N\rangle$
//! - Multi-qubit Quantum Phase Estimation (QPE) with exact IQFT
//! - Continued fractions convergent extraction $p_k / q_k \approx \phi$
//! - Deterministic prime factor recovery $p = \gcd(a^{r/2} - 1, N)$, $q = \gcd(a^{r/2} + 1, N)$

pub mod continued_fractions;
pub mod factorizer;
pub mod modular_arithmetic;
pub mod order_finding;

pub use continued_fractions::{ContinuedFractions, Convergent};
pub use factorizer::{ShorFactorizationReport, ShorFactorizer};
pub use modular_arithmetic::{
    classical_multiplicative_order, gcd, mod_pow, ModularMultiplier,
};
pub use order_finding::{QuantumOrderFinder, QuantumOrderResult};
