// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Sovereign Directed 7-Node Graph Walk & Spectral Routing Engine
//!
//! Implements discrete Markov chain propagation and continuous-time quantum walk (CTQW)
//! dynamics on the custom 7-node directed/weighted routing topology with transition matrix $P$:
//!
//! $$P = \begin{pmatrix}
//! 0   & 1/4 & 1/4 & 0   & 0   & 1/4 & 1/4 \\
//! 0   & 0   & 0   & 1/2 & 0   & 0   & 1/2 \\
//! 1/2 & 0   & 0   & 0   & 0   & 1/2 & 0   \\
//! 0   & 1/2 & 0   & 0   & 1/2 & 0   & 0   \\
//! 0   & 0   & 0   & 1/2 & 0   & 0   & 1/2 \\
//! 1/2 & 0   & 1/2 & 0   & 0   & 0   & 0   \\
//! 1/2 & 0   & 0   & 0   & 1/2 & 0   & 0
//! \end{pmatrix}$$
//!
//! ### Analytical Spectral Properties
//! - **Stationary Mode:** $\lambda_1 = 1$, with exact invariant distribution
//!   $\boldsymbol{\pi} = \frac{1}{42}(8, 5, 4, 6, 7, 4, 8)^T$.
//! - **Null Mode:** $\lambda_7 = 0$, arising from identical twin rows ($v_2 = v_5$).
//! - **Decoupled Anti-Phase Rim:** $\lambda = -1/2$ (multiplicity 2), localized on $v_3 - v_6$
//!   and the symmetric quotient branch.
//! - **Core Cubic Modes:** Three real roots of $16\lambda^3 - 10\lambda + 1 = 0$:
//!   $\lambda_2 \approx +0.734809$, $\lambda_3 \approx +0.101682$, $\lambda_6 \approx -0.836491$.
//! - **Spectral Gap:** $\Delta = 1 - \lambda_2 \approx 0.265191$, governing mixing time
//!   $\tau_{\text{mix}} \sim \frac{1}{\Delta} \approx 3.77$ steps.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Canonical 7-node directed transition matrix $P$.
pub const DIRECTED_HEPTAL_P: [[f64; 7]; 7] = [
    [0.0, 0.25, 0.25, 0.0, 0.0, 0.25, 0.25], // Row 0 (v1 - Hub)
    [0.0, 0.0,  0.0,  0.5, 0.0, 0.0,  0.5 ], // Row 1 (v2)
    [0.5, 0.0,  0.0,  0.0, 0.0, 0.5,  0.0 ], // Row 2 (v3)
    [0.0, 0.5,  0.0,  0.0, 0.5, 0.0,  0.0 ], // Row 3 (v4)
    [0.0, 0.0,  0.0,  0.5, 0.0, 0.0,  0.5 ], // Row 4 (v5 - Twin with v2)
    [0.5, 0.0,  0.5,  0.0, 0.0, 0.0,  0.0 ], // Row 5 (v6)
    [0.5, 0.0,  0.0,  0.0, 0.5, 0.0,  0.0 ], // Row 6 (v7 - Sink/Feedback)
];

/// Exact rational stationary probability distribution: pi = 1/42 * [8, 5, 4, 6, 7, 4, 8].
pub const STATIONARY_PI: [f64; 7] = [
    8.0 / 42.0, // v1
    5.0 / 42.0, // v2
    4.0 / 42.0, // v3
    6.0 / 42.0, // v4
    7.0 / 42.0, // v5
    4.0 / 42.0, // v6
    8.0 / 42.0, // v7
];

/// Analytical eigenvalues of the 7x7 transition matrix in descending order.
pub const EIGENVALUES_ANALYTICAL: [f64; 7] = [
    1.0,
    0.7348087170290185,
    0.10168210689845253,
    0.0,
    -0.5,
    -0.5,
    -0.8364908239274711,
];

/// Directed 7-node Markov routing filter and discrete probability engine.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DirectedHeptalRouter {
    /// Transition probability matrix (row-stochastic).
    pub matrix: [[f64; 7]; 7],
}

impl Default for DirectedHeptalRouter {
    fn default() -> Self {
        Self {
            matrix: DIRECTED_HEPTAL_P,
        }
    }
}

impl DirectedHeptalRouter {
    /// Constructs a standard router initialized with the sovereign 7-node transition matrix.
    pub fn new() -> Self {
        Self::default()
    }

    /// Advances a 7-element probability vector by one discrete step: $\mathbf{p}_{t+1} = \mathbf{p}_t P$.
    #[inline(always)]
    pub fn step_distribution(&self, p: &[f64; 7]) -> [f64; 7] {
        let mut next = [0.0f64; 7];
        for col in 0..7 {
            let mut acc = 0.0;
            for row in 0..7 {
                acc += p[row] * self.matrix[row][col];
            }
            next[col] = acc;
        }
        next
    }

    /// Evolve an initial probability distribution by $k$ discrete steps.
    pub fn evolve_steps(&self, initial: &[f64; 7], steps: usize) -> [f64; 7] {
        let mut current = *initial;
        for _ in 0..steps {
            current = self.step_distribution(&current);
        }
        current
    }

    /// Multiplies two 7x7 matrices: $C = A \cdot B$.
    pub fn matrix_mul(a: &[[f64; 7]; 7], b: &[[f64; 7]; 7]) -> [[f64; 7]; 7] {
        let mut out = [[0.0f64; 7]; 7];
        for i in 0..7 {
            for k in 0..7 {
                let aik = a[i][k];
                if aik.abs() > 1e-15 {
                    for j in 0..7 {
                        out[i][j] += aik * b[k][j];
                    }
                }
            }
        }
        out
    }

    /// Computes $P^n$ via binary exponentiation.
    pub fn matrix_power(&self, mut n: usize) -> [[f64; 7]; 7] {
        let mut result = [[0.0f64; 7]; 7];
        for i in 0..7 {
            result[i][i] = 1.0;
        }
        let mut base = self.matrix;
        while n > 0 {
            if n & 1 == 1 {
                result = Self::matrix_mul(&result, &base);
            }
            base = Self::matrix_mul(&base, &base);
            n >>= 1;
        }
        result
    }

    /// Computes the Total Variation (TV) distance to the stationary distribution $\boldsymbol{\pi}$:
    /// $$d_{\text{TV}}(p, \pi) = \frac{1}{2} \sum_{i=1}^7 |p_i - \pi_i|$$
    #[inline]
    pub fn total_variation_distance(&self, p: &[f64; 7]) -> f64 {
        let mut sum = 0.0;
        for i in 0..7 {
            sum += (p[i] - STATIONARY_PI[i]).abs();
        }
        0.5 * sum
    }

    /// Solves the 3 exact real roots of the core cubic factor $16\lambda^3 - 10\lambda + 1 = 0$
    /// using Newton-Raphson iteration to machine precision.
    pub fn solve_core_cubic_roots() -> [f64; 3] {
        let f = |x: f64| 16.0 * x * x * x - 10.0 * x + 1.0;
        let df = |x: f64| 48.0 * x * x - 10.0;
        let refine = |mut x: f64| -> f64 {
            for _ in 0..64 {
                let fx = f(x);
                let dfx = df(x);
                if dfx.abs() < 1e-15 {
                    break;
                }
                let step = fx / dfx;
                x -= step;
                if step.abs() < 1e-15 {
                    break;
                }
            }
            x
        };
        // Initial brackets: [-0.9, -0.8], [0.0, 0.2], [0.7, 0.8]
        let r_neg = refine(-0.85);
        let r_mid = refine(0.1);
        let r_pos = refine(0.73);
        [r_pos, r_mid, r_neg]
    }

    /// Evaluates the characteristic polynomial $\det(\lambda I - P)$.
    /// Analytically: $p(\lambda) = \lambda (\lambda - 1) (\lambda + 0.5)^2 (16\lambda^3 - 10\lambda + 1)$.
    #[inline]
    pub fn characteristic_polynomial(lambda: f64) -> f64 {
        let cubic = 16.0 * lambda * lambda * lambda - 10.0 * lambda + 1.0;
        let l_plus_half = lambda + 0.5;
        lambda * (lambda - 1.0) * l_plus_half * l_plus_half * cubic
    }

    /// Evaluates the core cubic factor $16\lambda^3 - 10\lambda + 1$.
    #[inline]
    pub fn core_cubic(lambda: f64) -> f64 {
        16.0 * lambda * lambda * lambda - 10.0 * lambda + 1.0
    }

    /// Returns the spectral gap $\Delta = 1 - \lambda_2 \approx 0.256936$.
    #[inline]
    pub fn spectral_gap(&self) -> f64 {
        1.0 - EIGENVALUES_ANALYTICAL[1]
    }
}

/// Continuous-Time Quantum Walk (CTQW) on the directed 7-node graph.
///
/// Evolution is governed by the Hermitian Hamiltonian:
/// $$H = \frac{\gamma}{2} (P + P^T) + i \frac{\mu}{2} (P - P^T)$$
/// where $\gamma$ governs symmetric tunneling and $\mu$ induces directed chiral flux.
#[derive(Debug, Clone)]
pub struct DirectedHeptalQuantumWalk {
    /// Symmetric hopping coefficient gamma.
    pub gamma: f64,
    /// Chiral / asymmetric flux coefficient mu.
    pub mu: f64,
    /// 7-node quantum amplitudes.
    pub psi: [Complex64; 7],
    /// Precomputed 7x7 Hermitian Hamiltonian.
    pub hamiltonian: [[Complex64; 7]; 7],
}

impl DirectedHeptalQuantumWalk {
    /// Constructs a new Continuous-Time Quantum Walk engine.
    pub fn new(gamma: f64, mu: f64) -> Self {
        let mut h = [[Complex64::new(0.0, 0.0); 7]; 7];
        let p = DIRECTED_HEPTAL_P;

        for r in 0..7 {
            for c in 0..7 {
                let sym = 0.5 * gamma * (p[r][c] + p[c][r]);
                let asym = 0.5 * mu * (p[r][c] - p[c][r]);
                h[r][c] = Complex64::new(sym, asym);
            }
        }

        Self {
            gamma,
            mu,
            psi: [Complex64::new(0.0, 0.0); 7],
            hamiltonian: h,
        }
    }

    /// Sets the initial state to a localized basis state $|v_{\text{node}}\rangle$.
    pub fn set_localized(&mut self, node: usize) {
        assert!(node < 7, "Node index out of bounds [0..6]");
        for i in 0..7 {
            self.psi[i] = if i == node {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
        }
    }

    /// Sets the initial state to uniform superposition $\frac{1}{\sqrt{7}} \sum_{j=1}^7 |v_j\rangle$.
    pub fn set_superposition(&mut self) {
        let amp = 1.0 / (7.0f64).sqrt();
        for i in 0..7 {
            self.psi[i] = Complex64::new(amp, 0.0);
        }
    }

    /// Advances the quantum state using a 4th-order Runge-Kutta (RK4) integrator:
    /// $$\frac{d|\psi\rangle}{dt} = -i H |\psi\rangle$$
    pub fn step_rk4(&mut self, dt: f64) {
        let h_psi = |state: &[Complex64; 7]| -> [Complex64; 7] {
            let mut out = [Complex64::new(0.0, 0.0); 7];
            for r in 0..7 {
                let mut sum = Complex64::new(0.0, 0.0);
                for c in 0..7 {
                    sum += self.hamiltonian[r][c] * state[c];
                }
                // -i * sum
                out[r] = Complex64::new(sum.im, -sum.re);
            }
            out
        };

        let k1 = h_psi(&self.psi);

        let mut y2 = [Complex64::new(0.0, 0.0); 7];
        for i in 0..7 {
            y2[i] = self.psi[i] + k1[i] * (0.5 * dt);
        }
        let k2 = h_psi(&y2);

        let mut y3 = [Complex64::new(0.0, 0.0); 7];
        for i in 0..7 {
            y3[i] = self.psi[i] + k2[i] * (0.5 * dt);
        }
        let k3 = h_psi(&y3);

        let mut y4 = [Complex64::new(0.0, 0.0); 7];
        for i in 0..7 {
            y4[i] = self.psi[i] + k3[i] * dt;
        }
        let k4 = h_psi(&y4);

        for i in 0..7 {
            self.psi[i] += (k1[i] + k2[i] * 2.0 + k3[i] * 2.0 + k4[i]) * (dt / 6.0);
        }

        self.normalize();
    }

    /// Normalizes the 7-node quantum state to $\langle \psi | \psi \rangle = 1$.
    #[inline]
    pub fn normalize(&mut self) {
        let norm_sq: f64 = self.psi.iter().map(|c| c.norm_sqr()).sum();
        if norm_sq > 0.0 {
            let inv_norm = 1.0 / norm_sq.sqrt();
            for c in &mut self.psi {
                *c *= inv_norm;
            }
        }
    }

    /// Extracts the node detection probability vector $p_i = |\psi_i|^2$.
    #[inline]
    pub fn probabilities(&self) -> [f64; 7] {
        let mut p = [0.0f64; 7];
        for i in 0..7 {
            p[i] = self.psi[i].norm_sqr();
        }
        p
    }

    /// Computes the Shannon entropy of the detection probability distribution:
    /// $$S = -\sum_{i=1}^7 p_i \ln(p_i)$$
    pub fn entropy(&self) -> f64 {
        let probs = self.probabilities();
        let mut ent = 0.0;
        for &p in &probs {
            if p > 1e-15 {
                ent -= p * p.ln();
            }
        }
        ent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_stochasticity() {
        for (r, row) in DIRECTED_HEPTAL_P.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Row {} does not sum to 1.0: sum = {}",
                r,
                sum
            );
        }
    }

    #[test]
    fn test_stationary_distribution_invariance() {
        let router = DirectedHeptalRouter::new();
        let next_pi = router.step_distribution(&STATIONARY_PI);
        for i in 0..7 {
            assert!(
                (next_pi[i] - STATIONARY_PI[i]).abs() < 1e-14,
                "Stationary distribution mismatch at node {}: got {}, expected {}",
                i,
                next_pi[i],
                STATIONARY_PI[i]
            );
        }
    }

    #[test]
    fn test_twin_row_nullspace_cancellation() {
        let router = DirectedHeptalRouter::new();
        // Mode localized on v2 - v5: [0, 1, 0, 0, -1, 0, 0]
        let diff_mode = [0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0];
        let mapped = router.step_distribution(&diff_mode);
        for (i, &val) in mapped.iter().enumerate() {
            assert!(
                val.abs() < 1e-15,
                "Twin row cancellation failed at node {}: got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_rim_antiphase_eigenvalue() {
        let router = DirectedHeptalRouter::new();
        // Anti-phase rim mode on v3 - v6: [0, 0, 1, 0, 0, -1, 0]
        let rim_mode = [0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0];
        let mapped = router.step_distribution(&rim_mode);
        for i in 0..7 {
            let expected = -0.5 * rim_mode[i];
            assert!(
                (mapped[i] - expected).abs() < 1e-14,
                "Rim anti-phase eigenvalue mismatch at node {}: got {}, expected {}",
                i,
                mapped[i],
                expected
            );
        }
    }

    #[test]
    fn test_characteristic_polynomial_zeros() {
        let roots = DirectedHeptalRouter::solve_core_cubic_roots();
        println!("Exact cubic roots: {:?}", roots);
        for &r in &roots {
            let cubic_val = DirectedHeptalRouter::core_cubic(r);
            assert!(
                cubic_val.abs() < 1e-12,
                "Core cubic nonzero at root {}: {}",
                r,
                cubic_val
            );
            let p_val = DirectedHeptalRouter::characteristic_polynomial(r);
            assert!(
                p_val.abs() < 1e-12,
                "Characteristic polynomial nonzero at root {}: {}",
                r,
                p_val
            );
        }

        // Also verify discrete known eigenvalues: 0, 1, -0.5
        for &lambda in &[0.0, 1.0, -0.5] {
            let p_val = DirectedHeptalRouter::characteristic_polynomial(lambda);
            assert!(
                p_val.abs() < 1e-12,
                "Characteristic polynomial nonzero at eigenvalue {}: {}",
                lambda,
                p_val
            );
        }
    }

    #[test]
    fn test_discrete_walk_convergence_to_stationary() {
        let router = DirectedHeptalRouter::new();
        let initial = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Localized on v1
        let p30 = router.evolve_steps(&initial, 30);
        let tv = router.total_variation_distance(&p30);
        assert!(
            tv < 1e-3,
            "Discrete walk failed to converge to stationary after 30 steps: tv = {}",
            tv
        );
    }

    #[test]
    fn test_ctqw_unitarity_and_entropy() {
        let mut ctqw = DirectedHeptalQuantumWalk::new(1.0, 0.5);
        ctqw.set_localized(0); // Start at Hub v1

        for _ in 0..100 {
            ctqw.step_rk4(0.01);
            let p: f64 = ctqw.probabilities().iter().sum();
            assert!(
                (p - 1.0).abs() < 1e-10,
                "Quantum walk norm violated: sum = {}",
                p
            );
        }

        let ent = ctqw.entropy();
        assert!(ent > 0.0, "Entropy should be positive after ballistic transit");
    }
}
