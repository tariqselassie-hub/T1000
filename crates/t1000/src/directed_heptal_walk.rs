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
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],     // Row 1 (v2)
    [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],     // Row 2 (v3)
    [0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0],     // Row 3 (v4)
    [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5],     // Row 4 (v5 - Twin with v2)
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],     // Row 5 (v6)
    [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],     // Row 6 (v7 - Sink/Feedback)
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

/// 49-Dimensional Bipartite Szegedy Walk Operator ($W_{49} = \mathcal{R}_B \mathcal{R}_A$) on $\mathcal{H}_7 \otimes \mathcal{H}_7$.
///
/// Implements the exact unitary quantization of the non-reversible 7-node Markov chain:
/// - $\Pi_A = \sum_{i=1}^7 |\psi_i\rangle\langle\psi_i|$ with $|\psi_i\rangle = |i\rangle \otimes \sum_j \sqrt{P_{ij}} |j\rangle$
/// - $\Pi_B = \sum_{j=1}^7 |\phi_j\rangle\langle\phi_j|$ with $|\phi_j\rangle = \sum_i \sqrt{P^*_{ji}} |i\rangle \otimes |j\rangle$
/// - $\mathcal{R}_A = 2\Pi_A - I_{49}, \quad \mathcal{R}_B = 2\Pi_B - I_{49}$
/// - $W_{49} = \mathcal{R}_B \mathcal{R}_A$ (strictly unitary, $W_{49}^T W_{49} = I_{49}$).
#[derive(Debug, Clone)]
pub struct BipartiteSzegedyWalk {
    /// Time-reversed dual transition matrix $P^*_{ji} = \frac{\pi_i P_{ij}}{\pi_j}$.
    pub p_star: [[f64; 7]; 7],
    /// 49x49 Unitary Szegedy walk evolution matrix $W_{49} = \mathcal{R}_B \mathcal{R}_A$.
    pub w_49: [[f64; 49]; 49],
    /// 49-dimensional quantum state vector.
    pub state: [f64; 49],
}

impl Default for BipartiteSzegedyWalk {
    fn default() -> Self {
        Self::new()
    }
}

/// Molecular Stark dipole vector configuration on the 7-node directed graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MolecularDipoleGraph {
    /// 3D dipole vectors [Dx, Dy, Dz] in Debye for nodes v1..v7.
    pub dipoles: [[f64; 3]; 7],
    /// External field vector E_ext.
    pub e_field: [f64; 3],
    /// Temperature coupling parameter beta.
    pub beta: f64,
}

impl MolecularDipoleGraph {
    /// Constructs dipole graph populated from the top 7 Pitt NLO leads.
    pub fn from_pitt_leads() -> Self {
        Self {
            dipoles: [
                [38.20, 22.90, 0.50],   // v1 (Hub) - Lead #1 C27H39N2O3S+ (44.59 D)
                [-18.50, 29.40, -2.10], // v2 - Lead #2 C30H20O4 (34.81 D)
                [48.20, 2.30, 0.00],    // v3 - Lead #3 C25H32O7 (48.26 D)
                [-10.20, 47.90, 4.30],  // v4 - Lead #4 C21H29FN3O6S (49.08 D)
                [22.10, 0.00, 0.00],    // v5 - Lead #5 C32H22O2 (22.13 D)
                [-44.80, 0.00, 1.20],   // v6 - Lead #6 C20H23N7O6 (44.80 D)
                [38.50, 0.00, -1.00],   // v7 (Sink) - Lead #7 C30H56O3 (38.51 D)
            ],
            e_field: [0.05, 0.0, 0.0], // 0.05 a.u. field aligned along X-axis
            beta: 0.05,
        }
    }

    /// Computes the Stark potential V_i = -D_i . E_ext for each node.
    pub fn stark_potentials(&self) -> [f64; 7] {
        let mut v = [0.0; 7];
        for i in 0..7 {
            let dot = self.dipoles[i][0] * self.e_field[0]
                + self.dipoles[i][1] * self.e_field[1]
                + self.dipoles[i][2] * self.e_field[2];
            v[i] = -dot;
        }
        v
    }

    /// Computes the tilted transition matrix P_tilde.
    pub fn tilted_transition_matrix(&self) -> [[f64; 7]; 7] {
        let v = self.stark_potentials();
        let p = DIRECTED_HEPTAL_P;
        let mut p_tilt = [[0.0; 7]; 7];

        for i in 0..7 {
            let mut sum_row = 0.0;
            for j in 0..7 {
                if p[i][j] > 0.0 {
                    let weight = p[i][j] * (-self.beta * (v[j] - v[i])).exp();
                    p_tilt[i][j] = weight;
                    sum_row += weight;
                }
            }
            if sum_row > 1e-15 {
                for j in 0..7 {
                    p_tilt[i][j] /= sum_row;
                }
            }
        }
        p_tilt
    }

    /// Computes stationary distribution pi_tilde by numerical power iteration.
    pub fn stationary_distribution(&self) -> [f64; 7] {
        let p_tilt = self.tilted_transition_matrix();
        let mut pi = [1.0 / 7.0; 7];
        for _ in 0..200 {
            let mut next = [0.0; 7];
            for c in 0..7 {
                for r in 0..7 {
                    next[c] += pi[r] * p_tilt[r][c];
                }
            }
            pi = next;
        }
        let sum: f64 = pi.iter().sum();
        if sum > 0.0 {
            for x in &mut pi {
                *x /= sum;
            }
        }
        pi
    }
}

impl BipartiteSzegedyWalk {
    /// Constructs and initializes the 49x49 Bipartite Szegedy Walk operator from canonical P.
    pub fn new() -> Self {
        Self::from_transition_matrix(DIRECTED_HEPTAL_P, STATIONARY_PI)
    }

    /// Constructs unitary Szegedy walk operator from molecular dipole drift.
    pub fn with_molecular_dipole_drift(dipoles: &MolecularDipoleGraph) -> Self {
        let p_tilt = dipoles.tilted_transition_matrix();
        let pi_tilt = dipoles.stationary_distribution();
        Self::from_transition_matrix(p_tilt, pi_tilt)
    }

    /// Constructs the 49x49 Szegedy walk operator from an arbitrary row-stochastic P and stationary distribution pi.
    pub fn from_transition_matrix(p: [[f64; 7]; 7], pi: [f64; 7]) -> Self {
        // 1. Construct dual matrix P^*_{ji} = (pi_i * P_{ij}) / pi_j
        let mut p_star = [[0.0f64; 7]; 7];
        for i in 0..7 {
            for j in 0..7 {
                if p[i][j] > 0.0 && pi[j] > 1e-15 {
                    p_star[j][i] = (pi[i] * p[i][j]) / pi[j];
                }
            }
        }

        // 2. Build projectors Pi_A and Pi_B
        let mut pi_a = [[0.0f64; 49]; 49];
        let mut pi_b = [[0.0f64; 49]; 49];

        for i in 0..7 {
            let mut psi_i = [0.0f64; 49];
            for j in 0..7 {
                psi_i[7 * i + j] = p[i][j].sqrt();
            }
            for r in 0..49 {
                for c in 0..49 {
                    pi_a[r][c] += psi_i[r] * psi_i[c];
                }
            }
        }

        for j in 0..7 {
            let mut phi_j = [0.0f64; 49];
            for i in 0..7 {
                phi_j[7 * i + j] = p_star[j][i].sqrt();
            }
            for r in 0..49 {
                for c in 0..49 {
                    pi_b[r][c] += phi_j[r] * phi_j[c];
                }
            }
        }

        // 3. Build reflections R_A = 2*Pi_A - I, R_B = 2*Pi_B - I
        let mut r_a = [[0.0f64; 49]; 49];
        let mut r_b = [[0.0f64; 49]; 49];
        for r in 0..49 {
            for c in 0..49 {
                let delta = if r == c { 1.0 } else { 0.0 };
                r_a[r][c] = 2.0 * pi_a[r][c] - delta;
                r_b[r][c] = 2.0 * pi_b[r][c] - delta;
            }
        }

        // 4. Multiply W_49 = R_B * R_A
        let mut w_49 = [[0.0f64; 49]; 49];
        for r in 0..49 {
            for k in 0..49 {
                let rb_rk = r_b[r][k];
                if rb_rk.abs() > 1e-15 {
                    for c in 0..49 {
                        w_49[r][c] += rb_rk * r_a[k][c];
                    }
                }
            }
        }

        Self {
            p_star,
            w_49,
            state: [0.0f64; 49],
        }
    }

    /// Measures the deviation of $W_{49}$ from strict unitarity: $\|W^T W - I\|_{\max}$.
    pub fn unitarity_error(&self) -> f64 {
        let mut max_err = 0.0f64;
        for r in 0..49 {
            for c in 0..49 {
                let mut dot = 0.0;
                for k in 0..49 {
                    dot += self.w_49[k][r] * self.w_49[k][c];
                }
                let delta = if r == c { 1.0 } else { 0.0 };
                let err = (dot - delta).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        max_err
    }

    /// Sets the state to localized outgoing basis state $|\psi_{\text{node}}\rangle = |\text{node}\rangle \otimes \sum_j \sqrt{P_{\text{node}, j}} |j\rangle$.
    pub fn set_localized(&mut self, node: usize) {
        assert!(node < 7, "Node out of bounds");
        self.state = [0.0f64; 49];
        for j in 0..7 {
            self.state[7 * node + j] = DIRECTED_HEPTAL_P[node][j].sqrt();
        }
    }

    /// Sets the state to the exact stationary Szegedy eigenvector $|\psi_\pi\rangle = \sum_{i,j} \sqrt{\pi_i P_{ij}} |i, j\rangle$.
    pub fn set_stationary_state(&mut self) {
        let pi = STATIONARY_PI;
        let p = DIRECTED_HEPTAL_P;
        for i in 0..7 {
            for j in 0..7 {
                self.state[7 * i + j] = (pi[i] * p[i][j]).sqrt();
            }
        }
    }

    /// Advances the 49-amplitude statevector by one unitary Szegedy step: $|\psi_{t+1}\rangle = W_{49} |\psi_t\rangle$.
    #[inline(always)]
    pub fn step(&mut self) {
        let mut next = [0.0f64; 49];
        for r in 0..49 {
            let mut acc = 0.0;
            for c in 0..49 {
                acc += self.w_49[r][c] * self.state[c];
            }
            next[r] = acc;
        }
        self.state = next;
    }

    /// Extracts detection probabilities across the 7 graph nodes by tracing out the target register:
    /// $$p_i = \sum_{j=0}^6 |\psi(i, j)|^2$$
    #[inline]
    pub fn node_probabilities(&self) -> [f64; 7] {
        let mut p = [0.0f64; 7];
        for i in 0..7 {
            let mut sum = 0.0;
            for j in 0..7 {
                let amp = self.state[7 * i + j];
                sum += amp * amp;
            }
            p[i] = sum;
        }
        p
    }

    /// Extracts total probability residing in the hub $v_1$ (row 0).
    #[inline]
    pub fn hub_leakage(&self) -> f64 {
        let mut sum = 0.0;
        for j in 0..7 {
            let amp = self.state[j];
            sum += amp * amp;
        }
        sum
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
        assert!(
            ent > 0.0,
            "Entropy should be positive after ballistic transit"
        );
    }

    #[test]
    fn test_szegedy_unitarity_machine_precision() {
        let walk = BipartiteSzegedyWalk::new();
        let err = walk.unitarity_error();
        assert!(
            err < 1e-13,
            "Szegedy W_49 operator violated unitarity: error = {}",
            err
        );
    }

    #[test]
    fn test_szegedy_stationary_eigenvector_invariance() {
        let mut walk = BipartiteSzegedyWalk::new();
        walk.set_stationary_state();
        let initial_state = walk.state;
        walk.step();

        let mut diff_sq = 0.0;
        for i in 0..49 {
            let d = walk.state[i] - initial_state[i];
            diff_sq += d * d;
        }
        assert!(
            diff_sq.sqrt() < 1e-12,
            "Stationary state |psi_pi> not an invariant eigenvector of W_49: residual = {}",
            diff_sq.sqrt()
        );
    }

    #[test]
    fn test_molecular_dipole_szegedy_walk() {
        let dipoles = MolecularDipoleGraph::from_pitt_leads();
        let p_tilt = dipoles.tilted_transition_matrix();
        for row in &p_tilt {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "Row stochasticity violated: {sum}"
            );
        }
        let walk = BipartiteSzegedyWalk::with_molecular_dipole_drift(&dipoles);
        assert!(
            walk.unitarity_error() < 1e-13,
            "Dipole walk must be unitary"
        );
    }
}
