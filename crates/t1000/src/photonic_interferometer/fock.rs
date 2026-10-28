// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Multi-Photon Fock State Quantum Optics & Boson Sampling Engine
//!
//! Models indistinguishable multi-photon states in linear optical interferometers:
//! - Multi-mode occupation number representation $|n_0, n_1, \dots, n_{M-1}\rangle$.
//! - Exact matrix permanent calculation via Glynn's algorithm ($O(N 2^N)$).
//! - Transition amplitudes $\langle s | U | r \rangle$ between arbitrary input and output Fock states.
//! - Hong-Ou-Mandel (HOM) two-photon quantum interference and bunching visibility.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Multi-mode Fock state representing photon occupation numbers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FockState {
    pub modes: Vec<usize>,
}

impl FockState {
    /// Creates a Fock state from mode occupation numbers.
    pub fn new(modes: Vec<usize>) -> Self {
        Self { modes }
    }

    /// Creates a vacuum state with `n_modes` optical modes: $|0, 0, \dots, 0\rangle$.
    pub fn vacuum(n_modes: usize) -> Self {
        Self {
            modes: vec![0; n_modes],
        }
    }

    /// Single photon localized in `mode_idx` of an `n_modes` interferometer.
    pub fn single_photon(n_modes: usize, mode_idx: usize) -> Self {
        assert!(mode_idx < n_modes, "Mode index out of range");
        let mut modes = vec![0; n_modes];
        modes[mode_idx] = 1;
        Self { modes }
    }

    /// Two-photon input state with one photon in mode `a` and one in mode `b`.
    pub fn two_photon(n_modes: usize, mode_a: usize, mode_b: usize) -> Self {
        assert!(mode_a < n_modes && mode_b < n_modes, "Mode index out of bounds");
        let mut modes = vec![0; n_modes];
        modes[mode_a] += 1;
        modes[mode_b] += 1;
        Self { modes }
    }

    /// Total number of indistinguishable photons in the state: $N = \sum_i n_i$.
    pub fn total_photons(&self) -> usize {
        self.modes.iter().sum()
    }

    /// Number of optical modes $M$.
    pub fn n_modes(&self) -> usize {
        self.modes.len()
    }

    /// Computes the factorial product $\prod_{i} (n_i!)$.
    pub fn factorial_product(&self) -> f64 {
        let mut prod = 1.0f64;
        for &n in &self.modes {
            if n > 1 {
                let mut f = 1.0f64;
                for k in 2..=n {
                    f *= k as f64;
                }
                prod *= f;
            }
        }
        prod
    }

    /// Formats the Fock state in Dirac notation (e.g. `|1, 0, 1, 0>`).
    pub fn to_dirac_string(&self) -> String {
        let s = self
            .modes
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("|{s}>")
    }
}

/// Generates the complete basis of Fock states with $N$ photons distributed over $M$ modes.
///
/// The dimension of the Hilbert space is $\binom{M + N - 1}{N}$.
pub fn generate_fock_basis(n_modes: usize, n_photons: usize) -> Vec<FockState> {
    let mut basis = Vec::new();
    let mut current = vec![0; n_modes];

    fn backtrack(
        mode_idx: usize,
        remaining_photons: usize,
        n_modes: usize,
        current: &mut Vec<usize>,
        basis: &mut Vec<FockState>,
    ) {
        if mode_idx == n_modes - 1 {
            current[mode_idx] = remaining_photons;
            basis.push(FockState::new(current.clone()));
            return;
        }

        for n in 0..=remaining_photons {
            current[mode_idx] = n;
            backtrack(
                mode_idx + 1,
                remaining_photons - n,
                n_modes,
                current,
                basis,
            );
        }
    }

    backtrack(0, n_photons, n_modes, &mut current, &mut basis);
    basis
}

/// Computes the exact matrix permanent using Glynn's formula.
///
/// $\text{perm}(A) = \frac{1}{2^{N-1}} \sum_{\delta \in \{-1, +1\}^N, \delta_1 = 1} \left( \prod_k \delta_k \right) \prod_j \left( \sum_i \delta_i A_{i, j} \right)$
pub fn matrix_permanent(mat: &[Vec<Complex64>]) -> Complex64 {
    let n = mat.len();
    if n == 0 {
        return Complex64::new(1.0, 0.0);
    }
    if n == 1 {
        return mat[0][0];
    }
    if n == 2 {
        return mat[0][0] * mat[1][1] + mat[0][1] * mat[1][0];
    }

    let n_sub = 1usize << (n - 1);
    let mut sum_terms = Complex64::new(0.0, 0.0);
    let mut delta = vec![1.0f64; n];

    for mask in 0..n_sub {
        let mut sign_prod = 1.0f64;
        delta[0] = 1.0;
        for (i, d) in delta.iter_mut().enumerate().skip(1) {
            if (mask & (1 << (i - 1))) != 0 {
                *d = -1.0;
                sign_prod = -sign_prod;
            } else {
                *d = 1.0;
            }
        }

        let mut col_prod = Complex64::new(1.0, 0.0);
        for j in 0..n {
            let col_sum: Complex64 = mat.iter().enumerate().map(|(i, row)| row[j] * delta[i]).sum();
            col_prod *= col_sum;
        }

        sum_terms += col_prod * sign_prod;
    }

    let norm = (n_sub as f64).recip();
    sum_terms * norm
}

/// Computes the multi-photon quantum transition amplitude $\langle s | U | r \rangle$.
///
/// $\langle s | U | r \rangle = \frac{\text{perm}(U_{s, r})}{\sqrt{\prod_i s_i! \prod_j r_j!}}$
pub fn fock_transition_amplitude(
    unitary: &[[f64; 16]],
    in_state: &FockState,
    out_state: &FockState,
) -> Complex64 {
    let n_in = in_state.total_photons();
    let n_out = out_state.total_photons();
    if n_in != n_out {
        return Complex64::new(0.0, 0.0);
    }
    if n_in == 0 {
        return Complex64::new(1.0, 0.0);
    }

    // Build submatrix U_{s, r}
    // Rows correspond to output modes s (each mode i repeated s_i times)
    // Columns correspond to input modes r (each mode j repeated r_j times)
    let mut submatrix = Vec::with_capacity(n_in);
    for (i, &count_s) in out_state.modes.iter().enumerate() {
        for _ in 0..count_s {
            let mut row = Vec::with_capacity(n_in);
            for (j, &count_r) in in_state.modes.iter().enumerate() {
                for _ in 0..count_r {
                    row.push(Complex64::new(unitary[i][j], 0.0));
                }
            }
            submatrix.push(row);
        }
    }

    let perm = matrix_permanent(&submatrix);
    let denom = (in_state.factorial_product() * out_state.factorial_product()).sqrt();
    perm / denom
}

/// Computes the full output Fock state probability distribution for an input state.
pub fn simulate_boson_sampling(
    unitary: &[[f64; 16]],
    in_state: &FockState,
) -> Vec<(FockState, f64)> {
    let n_photons = in_state.total_photons();
    let n_modes = in_state.n_modes();
    let basis = generate_fock_basis(n_modes, n_photons);

    let mut dist = Vec::with_capacity(basis.len());
    for out_state in basis {
        let amp = fock_transition_amplitude(unitary, in_state, &out_state);
        let prob = amp.norm_sqr();
        dist.push((out_state, prob));
    }

    dist
}

/// Models the foundational Hong-Ou-Mandel (HOM) two-photon quantum interference effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HongOuMandelExperiment {
    /// Two-photon coincidence probability $P(|1, 1\rangle)$.
    pub coincidence_prob: f64,
    /// Two-photon bunching probability $P(|2, 0\rangle)$.
    pub bunching_prob_a: f64,
    /// Two-photon bunching probability $P(|0, 2\rangle)$.
    pub bunching_prob_b: f64,
    /// Quantum interference visibility $V \in [0, 1]$.
    pub visibility: f64,
}

impl HongOuMandelExperiment {
    /// Executes HOM interference simulation through a 2x2 directional coupler.
    ///
    /// For an ideal 50:50 coupler, destructive interference completely cancels coincidence:
    /// $P(|1, 1\rangle) = 0$, $P(|2, 0\rangle) = 0.5$, $P(|0, 2\rangle) = 0.5$, $V = 1.0$.
    pub fn simulate_2x2(coupler: [[f64; 2]; 2]) -> Self {
        // Input: |1, 1>
        // Submatrix for |1, 1> output:
        // row 0: coupler[0][0], coupler[0][1]
        // row 1: coupler[1][0], coupler[1][1]
        let sub_11 = vec![
            vec![Complex64::new(coupler[0][0], 0.0), Complex64::new(coupler[0][1], 0.0)],
            vec![Complex64::new(coupler[1][0], 0.0), Complex64::new(coupler[1][1], 0.0)],
        ];
        let perm_11 = matrix_permanent(&sub_11);
        let prob_11 = perm_11.norm_sqr();

        // Submatrix for |2, 0> output:
        // row 0 repeated twice: coupler[0]
        let sub_20 = vec![
            vec![Complex64::new(coupler[0][0], 0.0), Complex64::new(coupler[0][1], 0.0)],
            vec![Complex64::new(coupler[0][0], 0.0), Complex64::new(coupler[0][1], 0.0)],
        ];
        let perm_20 = matrix_permanent(&sub_20);
        let prob_20 = perm_20.norm_sqr() / 2.0; // divided by 2! = 2

        // Submatrix for |0, 2> output:
        // row 1 repeated twice: coupler[1]
        let sub_02 = vec![
            vec![Complex64::new(coupler[1][0], 0.0), Complex64::new(coupler[1][1], 0.0)],
            vec![Complex64::new(coupler[1][0], 0.0), Complex64::new(coupler[1][1], 0.0)],
        ];
        let perm_02 = matrix_permanent(&sub_02);
        let prob_02 = perm_02.norm_sqr() / 2.0;

        // Classical distinguishable particles would give P_distinguishable(1, 1) = 0.5
        let classical_coincidence = 0.5;
        let visibility = ((classical_coincidence - prob_11) / classical_coincidence).clamp(0.0, 1.0);

        Self {
            coincidence_prob: prob_11,
            bunching_prob_a: prob_20,
            bunching_prob_b: prob_02,
            visibility,
        }
    }
}

