// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Ultrametric Tree Sorter & Quantum Multiplexor (QMUX) Engine
//!
//! Replaces flat classification projection with a block-diagonal, multi-register
//! quantum multiplexor ($W_{\text{tree}} = U_{\text{sink}} \cdot U_{\text{QMUX}} \cdot U_{\text{trunk}}$).
//!
//! Enforces an ultrametric geometry across three explicit Hilbert registers:
//! $$\mathcal{H} = \mathcal{H}_C \otimes \mathcal{H}_F \otimes \mathcal{H}_E$$
//!
//! - $\mathcal{H}_C$ ($n_C$ qubits, $d_C = 2^{n_C}$): Coarse class invariant router.
//! - $\mathcal{H}_F$ ($n_F$ qubits, $d_F = 2^{n_F}$): Fine subpopulation register conditioned on $y$.
//! - $\mathcal{H}_E$ ($n_E$ qubits, $d_E = 2^{n_E}$): Environment / spurious entropy sink.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Configuration parameters for the 3-Register Ultrametric Tree Sorter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct UltrametricTreeConfig {
    /// Number of qubits in Coarse Register H_C (primary taxonomy router).
    pub n_coarse: usize,
    /// Number of qubits in Fine Register H_F (subpopulation manifold).
    pub n_fine: usize,
    /// Number of qubits in Environment Register H_E (entropy sink).
    pub n_env: usize,
}

impl UltrametricTreeConfig {
    /// Constructs a validated 3-register configuration.
    pub const fn new(n_coarse: usize, n_fine: usize, n_env: usize) -> Self {
        assert!(n_coarse >= 1, "Must have at least 1 coarse qubit");
        assert!(n_fine >= 1, "Must have at least 1 fine qubit");
        Self {
            n_coarse,
            n_fine,
            n_env,
        }
    }

    /// Total number of qubits across all three registers.
    #[inline(always)]
    pub const fn total_qubits(&self) -> usize {
        self.n_coarse + self.n_fine + self.n_env
    }

    /// Full statevector dimension ($2^N$).
    #[inline(always)]
    pub const fn dim(&self) -> usize {
        1 << self.total_qubits()
    }

    /// Bitmask for the Coarse Register (LSB).
    #[inline(always)]
    pub const fn coarse_mask(&self) -> usize {
        (1 << self.n_coarse) - 1
    }

    /// Dimension of the Coarse Register ($2^{n_C}$).
    #[inline(always)]
    pub const fn d_coarse(&self) -> usize {
        1 << self.n_coarse
    }

    /// Bitmask for the Fine Register.
    #[inline(always)]
    pub const fn fine_mask(&self) -> usize {
        (1 << self.n_fine) - 1
    }

    /// Dimension of the Fine Register ($2^{n_F}$).
    #[inline(always)]
    pub const fn d_fine(&self) -> usize {
        1 << self.n_fine
    }

    /// Dimension of the Environment Register ($2^{n_E}$).
    #[inline(always)]
    pub const fn d_env(&self) -> usize {
        1 << self.n_env
    }
}

/// Standard 2x2 rotation gate around Y-axis: R_y(theta).
#[inline(always)]
pub fn gate_ry(theta: f64) -> [[Complex64; 2]; 2] {
    let half = theta * 0.5;
    let c = Complex64::new(half.cos(), 0.0);
    let s = Complex64::new(half.sin(), 0.0);
    [[c, -s], [s, c]]
}

/// Standard 2x2 rotation gate around Z-axis: R_z(phi).
#[inline(always)]
pub fn gate_rz(phi: f64) -> [[Complex64; 2]; 2] {
    let half = phi * 0.5;
    [
        [
            Complex64::new(half.cos(), -half.sin()),
            Complex64::default(),
        ],
        [Complex64::default(), Complex64::new(half.cos(), half.sin())],
    ]
}

/// Applies an arbitrary 1-qubit unitary gate U(2) to target `qubit`.
pub fn apply_single_qubit_gate(state: &mut [Complex64], qubit: usize, u: &[[Complex64; 2]; 2]) {
    let half_dim = state.len() >> 1;
    let low_mask = (1 << qubit) - 1;

    for i in 0..half_dim {
        let idx0 = ((i & !low_mask) << 1) | (i & low_mask);
        let idx1 = idx0 | (1 << qubit);

        let a0 = state[idx0];
        let a1 = state[idx1];

        state[idx0] = u[0][0] * a0 + u[0][1] * a1;
        state[idx1] = u[1][0] * a0 + u[1][1] * a1;
    }
}

/// Applies a Controlled-Z (CZ) entangling gate between two arbitrary wires.
pub fn apply_cz_gate(state: &mut [Complex64], q1: usize, q2: usize) {
    let m1 = 1 << q1;
    let m2 = 1 << q2;
    for (idx, amp) in state.iter_mut().enumerate() {
        if (idx & m1 != 0) && (idx & m2 != 0) {
            *amp = -*amp;
        }
    }
}

/// Applies a Controlled-NOT (CNOT) gate between control and target wires.
pub fn apply_cnot_gate(state: &mut [Complex64], control: usize, target: usize) {
    let half_dim = state.len() >> 1;
    let low_mask = (1 << target) - 1;
    let ctrl_mask = 1 << control;

    for i in 0..half_dim {
        let idx0 = ((i & !low_mask) << 1) | (i & low_mask);
        let idx1 = idx0 | (1 << target);

        if idx0 & ctrl_mask != 0 {
            state.swap(idx0, idx1);
        }
    }
}

/// Applies a 1-qubit unitary U(2) to target `fine_qubit` inside H_F,
/// conditioned on the coarse register state `y` in H_C (Quantum Multiplexor).
///
/// Layout: [ Env (MSB) | Fine | Coarse (LSB) ]
/// Each branch `y` in 0..(1 << n_coarse) has its own independent 2x2 unitary `branch_gates[y]`.
pub fn apply_qmux_single_qubit_gate(
    state: &mut [Complex64],
    cfg: &UltrametricTreeConfig,
    fine_qubit_offset: usize,
    branch_gates: &[[[Complex64; 2]; 2]],
) {
    assert_eq!(branch_gates.len(), 1 << cfg.n_coarse);
    assert!(fine_qubit_offset < cfg.n_fine);

    let target_bit = cfg.n_coarse + fine_qubit_offset;
    let half_dim = state.len() >> 1;
    let low_mask = (1 << target_bit) - 1;
    let coarse_mask = cfg.coarse_mask();

    for i in 0..half_dim {
        let idx0 = ((i & !low_mask) << 1) | (i & low_mask);
        let idx1 = idx0 | (1 << target_bit);

        let y = idx0 & coarse_mask;
        let u = &branch_gates[y];

        let a0 = state[idx0];
        let a1 = state[idx1];

        state[idx0] = u[0][0] * a0 + u[0][1] * a1;
        state[idx1] = u[1][0] * a0 + u[1][1] * a1;
    }
}

/// Applies an entropy-cooling partial SWAP bridge between Fine wire and Env wire:
/// R_swap(phi) = exp(-i * phi/2 * SWAP).
pub fn apply_cooling_bridge(
    state: &mut [Complex64],
    fine_wire: usize,
    env_wire: usize,
    coupling: f64,
) {
    if coupling.abs() < 1e-12 {
        return;
    }
    let c = Complex64::new((coupling * 0.5).cos(), 0.0);
    let s = Complex64::new(0.0, -(coupling * 0.5).sin());

    let m_f = 1 << fine_wire;
    let m_e = 1 << env_wire;

    for idx0 in 0..state.len() {
        if (idx0 & m_f == 0) && (idx0 & m_e != 0) {
            let idx1 = (idx0 & !m_e) | m_f; // Swap bits
            let a0 = state[idx0];
            let a1 = state[idx1];
            state[idx0] = c * a0 + s * a1;
            state[idx1] = s * a0 + c * a1;
        }
    }
}

/// Computes the joint Ultrametric Projection Probabilities P(y, s)
/// by tracing out the Environment Register H_E in a single linear scan.
pub fn compute_ultrametric_branch_distribution(
    state: &[Complex64],
    cfg: &UltrametricTreeConfig,
) -> Vec<Vec<f64>> {
    let d_c = cfg.d_coarse();
    let d_f = cfg.d_fine();
    let coarse_mask = cfg.coarse_mask();
    let fine_mask = cfg.fine_mask();

    let mut probs = vec![vec![0.0; d_f]; d_c];

    for (idx, amp) in state.iter().enumerate() {
        let p = amp.norm_sqr();
        let y = idx & coarse_mask;
        let s = (idx >> cfg.n_coarse) & fine_mask;
        probs[y][s] += p;
    }

    probs
}

/// Verifies the Ultrametric Gram Rigidity across Coarse and Fine registers.
/// Returns (Coarse Overlap Tr(rho_C_u * rho_C_v), Fine Overlap Tr(rho_F_u * rho_F_v)).
pub fn ultrametric_gram_pair(prob_u: &[Vec<f64>], prob_v: &[Vec<f64>]) -> (f64, f64) {
    let d_c = prob_u.len();
    let d_f = prob_u[0].len();

    let mut coarse_dot = 0.0;
    for y in 0..d_c {
        let p_u_y: f64 = prob_u[y].iter().sum();
        let p_v_y: f64 = prob_v[y].iter().sum();
        coarse_dot += p_u_y * p_v_y;
    }

    let mut fine_dot = 0.0;
    for s in 0..d_f {
        let p_u_s: f64 = (0..d_c).map(|y| prob_u[y][s]).sum();
        let p_v_s: f64 = (0..d_c).map(|y| prob_v[y][s]).sum();
        fine_dot += p_u_s * p_v_s;
    }

    (coarse_dot, fine_dot)
}

/// Evaluates single-qubit expectation values <Z_q> for all qubits in the state.
pub fn compute_z_expectations(state: &[Complex64], n_qubits: usize) -> Vec<f64> {
    let mut z_exp = vec![0.0; n_qubits];
    for (idx, amp) in state.iter().enumerate() {
        let p = amp.norm_sqr();
        for (q, exp) in z_exp.iter_mut().enumerate() {
            if idx & (1 << q) == 0 {
                *exp += p;
            } else {
                *exp -= p;
            }
        }
    }
    z_exp
}

/// Evaluates connected ZZ correlator <Z_a Z_b> - <Z_a><Z_b>.
pub fn compute_connected_correlator(state: &[Complex64], qa: usize, qb: usize) -> f64 {
    let ma = 1 << qa;
    let mb = 1 << qb;
    let mut exp_za = 0.0;
    let mut exp_zb = 0.0;
    let mut exp_zz = 0.0;

    for (idx, amp) in state.iter().enumerate() {
        let p = amp.norm_sqr();
        let sign_a = if idx & ma == 0 { 1.0 } else { -1.0 };
        let sign_b = if idx & mb == 0 { 1.0 } else { -1.0 };

        exp_za += sign_a * p;
        exp_zb += sign_b * p;
        exp_zz += sign_a * sign_b * p;
    }

    exp_zz - (exp_za * exp_zb)
}

/// Calculates the theoretical Ultrametric tree distance d_U between two leaves.
/// Satisfies the strong triangle inequality: d_U(u, v) <= max(d_U(u, w), d_U(w, v)).
pub fn ultrametric_tree_distance(
    yu: usize,
    su: usize,
    yv: usize,
    sv: usize,
    gamma1: f64,
    gamma2: f64,
) -> f64 {
    if yu != yv {
        gamma2
    } else if su != sv {
        gamma1
    } else {
        0.0
    }
}

/// Comprehensive loss components for the Ultrametric Tree Sorter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltrametricLossReport {
    /// Total combined loss L_total = L_coarse + lambda_f * L_fine + lambda_perp * L_decorr.
    pub loss_total: f64,
    /// Coarse class routing loss L_coarse.
    pub loss_coarse: f64,
    /// Fine subpopulation conditional loss L_fine.
    pub loss_fine: f64,
    /// Cross-register spurious decorrelation penalty L_decorr.
    pub loss_decorr: f64,
    /// Mean coarse class routing accuracy.
    pub coarse_accuracy: f64,
    /// Mean fine subpopulation identification accuracy.
    pub fine_accuracy: f64,
    /// Joint leaf prediction accuracy (y and s both correct).
    pub joint_accuracy: f64,
}

/// Full Parameterized Ultrametric Tree Sorter Circuit Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltrametricTreeSorter {
    /// Hilbert register configuration.
    pub cfg: UltrametricTreeConfig,
    /// Number of VQC trunk layers.
    pub n_trunk_layers: usize,
    /// Parameters for Stage 1: Global Trunk Feature Extraction.
    pub trunk_params: Vec<f64>,
    /// Parameters for Stage 2: QMUX Branch Unitaries `[d_C][n_fine_params]`.
    pub branch_params: Vec<Vec<f64>>,
    /// Couplings for Stage 3: Spurious Decoupling Bridges between H_F and H_E.
    pub sink_couplings: Vec<f64>,
}

impl UltrametricTreeSorter {
    /// Creates a new Ultrametric Tree Sorter with randomized or zeros initialization.
    pub fn new(cfg: UltrametricTreeConfig, n_trunk_layers: usize) -> Self {
        let total_q = cfg.total_qubits();
        let trunk_param_count = n_trunk_layers * total_q * 2;
        let trunk_params = vec![0.0; trunk_param_count];

        let d_c = cfg.d_coarse();
        let fine_params_per_branch = cfg.n_fine * 2;
        let branch_params = vec![vec![0.0; fine_params_per_branch]; d_c];

        let sink_count = cfg.n_fine.min(cfg.n_env);
        let sink_couplings = vec![0.0; sink_count];

        Self {
            cfg,
            n_trunk_layers,
            trunk_params,
            branch_params,
            sink_couplings,
        }
    }

    /// Initializes parameters with small pseudo-random Gaussian/uniform weights.
    pub fn init_seeded(&mut self, seed: u64) {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for p in self.trunk_params.iter_mut() {
            *p = rng.gen_range(-0.35..0.35);
        }
        for branch in self.branch_params.iter_mut() {
            for p in branch.iter_mut() {
                *p = rng.gen_range(-0.35..0.35);
            }
        }
        for s in self.sink_couplings.iter_mut() {
            *s = rng.gen_range(0.05..0.25);
        }
    }

    /// Total count of trainable variational parameters.
    pub fn total_parameters(&self) -> usize {
        let branch_total: usize = self.branch_params.iter().map(|b| b.len()).sum();
        self.trunk_params.len() + branch_total + self.sink_couplings.len()
    }

    /// Executes the 3-stage Ultrametric Tree circuit forward on an input statevector:
    /// 1. Stage 1: Global Trunk (U_trunk)
    /// 2. Stage 2: QMUX Branching (U_QMUX)
    /// 3. Stage 3: Spurious Cooling Bridges (U_sink)
    pub fn forward_state(&self, input_state: &[Complex64]) -> Vec<Complex64> {
        let total_q = self.cfg.total_qubits();
        assert_eq!(input_state.len(), self.cfg.dim());
        let mut state = input_state.to_vec();

        // Stage 1: Global Trunk Feature Extraction (entangled backbone)
        let mut p_idx = 0;
        for _ in 0..self.n_trunk_layers {
            for q in 0..total_q {
                let ry_p = self.trunk_params[p_idx];
                let rz_p = self.trunk_params[p_idx + 1];
                p_idx += 2;
                apply_single_qubit_gate(&mut state, q, &gate_ry(ry_p));
                apply_single_qubit_gate(&mut state, q, &gate_rz(rz_p));
            }
            // Entangling circular chain CZ across all registers
            for q in 0..total_q {
                let next_q = (q + 1) % total_q;
                apply_cz_gate(&mut state, q, next_q);
            }
        }

        // Stage 2: Quantum Multiplexor Branching (U_QMUX)
        // Each branch y in H_C controls independent rotations on H_F wires
        for f_offset in 0..self.cfg.n_fine {
            let mut branch_gates = Vec::with_capacity(self.cfg.d_coarse());
            for y in 0..self.cfg.d_coarse() {
                let p_ry = self.branch_params[y][f_offset * 2];
                let p_rz = self.branch_params[y][f_offset * 2 + 1];
                let u_ry = gate_ry(p_ry);
                let u_rz = gate_rz(p_rz);
                // Composite U = Rz * Ry
                let u = [
                    [
                        u_rz[0][0] * u_ry[0][0] + u_rz[0][1] * u_ry[1][0],
                        u_rz[0][0] * u_ry[0][1] + u_rz[0][1] * u_ry[1][1],
                    ],
                    [
                        u_rz[1][0] * u_ry[0][0] + u_rz[1][1] * u_ry[1][0],
                        u_rz[1][0] * u_ry[0][1] + u_rz[1][1] * u_ry[1][1],
                    ],
                ];
                branch_gates.push(u);
            }
            apply_qmux_single_qubit_gate(&mut state, &self.cfg, f_offset, &branch_gates);
        }

        // Stage 3: Spurious Decoupling & Trash Cooling (U_sink)
        // Bridges between Fine register and Env register to cool intra-class entropy
        let n_bridges = self.cfg.n_fine.min(self.cfg.n_env);
        for i in 0..n_bridges {
            let f_wire = self.cfg.n_coarse + i;
            let e_wire = self.cfg.n_coarse + self.cfg.n_fine + i;
            let coupling = self.sink_couplings[i];
            apply_cooling_bridge(&mut state, f_wire, e_wire, coupling);
        }

        state
    }

    /// Executes the 3-stage Ultrametric Tree circuit forward on input feature angles:
    /// 1. State initialization + angle injection
    /// 2. Stage 1: Global Trunk (U_trunk)
    /// 3. Stage 2: QMUX Branching (U_QMUX)
    /// 4. Stage 3: Spurious Cooling Bridges (U_sink)
    pub fn forward(&self, features: &[f64]) -> Vec<Complex64> {
        let total_q = self.cfg.total_qubits();
        let mut state = vec![Complex64::default(); self.cfg.dim()];
        state[0] = Complex64::new(1.0, 0.0); // |0...0>

        // Feature encoding: Ry(x_i) on each wire
        for q in 0..total_q {
            let angle = if q < features.len() { features[q] } else { 0.0 };
            apply_single_qubit_gate(&mut state, q, &gate_ry(angle));
        }

        self.forward_state(&state)
    }

    /// Predicts coarse class `y` and fine subpopulation `s` via leaf argmax.
    pub fn predict(&self, features: &[f64]) -> (usize, usize, f64) {
        let state = self.forward(features);
        let probs = compute_ultrametric_branch_distribution(&state, &self.cfg);

        let mut best_y = 0;
        let mut best_s = 0;
        let mut best_p = -1.0;

        for (y, s_row) in probs.iter().enumerate() {
            for (s, &p) in s_row.iter().enumerate() {
                if p > best_p {
                    best_p = p;
                    best_y = y;
                    best_s = s;
                }
            }
        }

        (best_y, best_s, best_p)
    }

    /// Evaluates the full ultrametric composite loss and accuracy across a batch.
    pub fn evaluate_batch(
        &self,
        samples: &[Vec<f64>],
        labels_y: &[usize],
        labels_s: &[usize],
        lambda_fine: f64,
        lambda_decorr: f64,
    ) -> UltrametricLossReport {
        assert_eq!(samples.len(), labels_y.len());
        assert_eq!(samples.len(), labels_s.len());
        let n = samples.len() as f64;

        let mut sum_loss_coarse = 0.0;
        let mut sum_loss_fine = 0.0;
        let mut sum_loss_decorr = 0.0;
        let mut correct_coarse = 0;
        let mut correct_fine = 0;
        let mut correct_joint = 0;

        for i in 0..samples.len() {
            let state = self.forward(&samples[i]);
            let probs = compute_ultrametric_branch_distribution(&state, &self.cfg);
            let target_y = labels_y[i];
            let target_s = labels_s[i];

            // Coarse routing fidelity
            let p_y: f64 = probs[target_y].iter().sum();
            sum_loss_coarse += 1.0 - p_y;

            // Branch-conditioned fine sorting
            let p_ys = probs[target_y][target_s];
            sum_loss_fine += 1.0 - p_ys;

            // Ultrametric independence decorrelation
            for qc in 0..self.cfg.n_coarse {
                for qf in 0..self.cfg.n_fine {
                    let fine_wire = self.cfg.n_coarse + qf;
                    let conn = compute_connected_correlator(&state, qc, fine_wire);
                    sum_loss_decorr += conn * conn;
                }
            }

            // Accuracy evaluation
            let mut pred_y = 0;
            let mut max_py = -1.0;
            for (y, row) in probs.iter().enumerate() {
                let py_sum: f64 = row.iter().sum();
                if py_sum > max_py {
                    max_py = py_sum;
                    pred_y = y;
                }
            }
            if pred_y == target_y {
                correct_coarse += 1;
            }

            let mut pred_s = 0;
            let mut max_ps = -1.0;
            for (s, &p) in probs[target_y].iter().enumerate() {
                if p > max_ps {
                    max_ps = p;
                    pred_s = s;
                }
            }
            if pred_s == target_s {
                correct_fine += 1;
            }

            if pred_y == target_y && pred_s == target_s {
                correct_joint += 1;
            }
        }

        let loss_coarse = sum_loss_coarse / n;
        let loss_fine = sum_loss_fine / n;
        let loss_decorr = sum_loss_decorr / n;
        let loss_total = loss_coarse + lambda_fine * loss_fine + lambda_decorr * loss_decorr;

        UltrametricLossReport {
            loss_total,
            loss_coarse,
            loss_fine,
            loss_decorr,
            coarse_accuracy: f64::from(correct_coarse) / n,
            fine_accuracy: f64::from(correct_fine) / n,
            joint_accuracy: f64::from(correct_joint) / n,
        }
    }

    /// Evaluates the analytical gradient with respect to branch parameters on an input statevector.
    pub fn parameter_shift_branch_grad_state(
        &mut self,
        input_state: &[Complex64],
        f_offset: usize,
        is_rz: bool,
        target_y: usize,
        target_s: usize,
    ) -> Vec<f64> {
        let p_idx = f_offset * 2 + if is_rz { 1 } else { 0 };
        let shift = std::f64::consts::FRAC_PI_2;
        let d_c = self.cfg.d_coarse();
        let mut grads = vec![0.0; d_c];

        // Parallel forward shift across all branches y
        for y in 0..d_c {
            self.branch_params[y][p_idx] += shift;
        }
        let state_plus = self.forward_state(input_state);
        let probs_plus = compute_ultrametric_branch_distribution(&state_plus, &self.cfg);

        // Parallel backward shift across all branches y
        for y in 0..d_c {
            self.branch_params[y][p_idx] -= 2.0 * shift;
        }
        let state_minus = self.forward_state(input_state);
        let probs_minus = compute_ultrametric_branch_distribution(&state_minus, &self.cfg);

        // Restore baseline parameter and record shift derivative
        for (y, grad) in grads.iter_mut().enumerate() {
            self.branch_params[y][p_idx] += shift;
            if y == target_y {
                let p_plus = probs_plus[target_y][target_s];
                let p_minus = probs_minus[target_y][target_s];
                *grad = 0.5 * (p_plus - p_minus);
            }
        }

        grads
    }

    /// Evaluates the analytical gradient with respect to branch parameters
    /// using the exact two-term Parameter-Shift Rule:
    /// d<P(y,s)>/d\theta_{y, k} = 0.5 * (P_{+\pi/2} - P_{-\pi/2}).
    pub fn parameter_shift_branch_grad(
        &mut self,
        features: &[f64],
        f_offset: usize,
        is_rz: bool,
        target_y: usize,
        target_s: usize,
    ) -> Vec<f64> {
        let total_q = self.cfg.total_qubits();
        let mut state = vec![Complex64::default(); self.cfg.dim()];
        state[0] = Complex64::new(1.0, 0.0);
        for q in 0..total_q {
            let angle = if q < features.len() { features[q] } else { 0.0 };
            apply_single_qubit_gate(&mut state, q, &gate_ry(angle));
        }
        self.parameter_shift_branch_grad_state(&state, f_offset, is_rz, target_y, target_s)
    }

    /// Evaluates the analytical gradient with respect to trunk parameters
    /// targeting coarse class routing using the Parameter-Shift Rule:
    /// d<P(target_y)>/d\theta_k = 0.5 * (P_{+\pi/2}(target_y) - P_{-\pi/2}(target_y)).
    pub fn parameter_shift_trunk_grad(
        &mut self,
        features: &[f64],
        param_idx: usize,
        target_y: usize,
    ) -> f64 {
        assert!(param_idx < self.trunk_params.len());
        let shift = std::f64::consts::FRAC_PI_2;

        self.trunk_params[param_idx] += shift;
        let s_plus = self.forward(features);
        let p_plus: f64 = compute_ultrametric_branch_distribution(&s_plus, &self.cfg)[target_y]
            .iter()
            .sum();

        self.trunk_params[param_idx] -= 2.0 * shift;
        let s_minus = self.forward(features);
        let p_minus: f64 = compute_ultrametric_branch_distribution(&s_minus, &self.cfg)[target_y]
            .iter()
            .sum();

        self.trunk_params[param_idx] += shift;
        0.5 * (p_plus - p_minus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_ultrametric_config_geometry() {
        let cfg = UltrametricTreeConfig::new(2, 4, 2);
        assert_eq!(cfg.total_qubits(), 8);
        assert_eq!(cfg.dim(), 256);
        assert_eq!(cfg.d_coarse(), 4);
        assert_eq!(cfg.d_fine(), 16);
        assert_eq!(cfg.d_env(), 4);
        assert_eq!(cfg.coarse_mask(), 0x03);
        assert_eq!(cfg.fine_mask(), 0x0F);
    }

    #[test]
    fn test_qmux_unitarity_preservation() {
        let cfg = UltrametricTreeConfig::new(1, 2, 1);
        let mut sorter = UltrametricTreeSorter::new(cfg, 1);
        sorter.init_seeded(42);

        let input_features = vec![0.5, -0.3, 1.2, 0.8];
        let state = sorter.forward(&input_features);

        let norm_sqr: f64 = state.iter().map(|a| a.norm_sqr()).sum();
        assert!(
            (norm_sqr - 1.0).abs() < 1e-12,
            "Statevector must remain strictly normalized (got {})",
            norm_sqr
        );
    }

    #[test]
    fn test_strong_triangle_inequality() {
        let gamma1 = 0.4;
        let gamma2 = 1.0;

        // Leaf 1: (0, 0), Leaf 2: (0, 1), Leaf 3: (1, 0)
        let d12 = ultrametric_tree_distance(0, 0, 0, 1, gamma1, gamma2); // Siblings => gamma1
        let d23 = ultrametric_tree_distance(0, 1, 1, 0, gamma1, gamma2); // Disjoint => gamma2
        let d13 = ultrametric_tree_distance(0, 0, 1, 0, gamma1, gamma2); // Disjoint => gamma2

        assert_eq!(d12, gamma1);
        assert_eq!(d23, gamma2);
        assert_eq!(d13, gamma2);
        assert!(d12 <= d23.max(d13));
        assert!(d13 <= d12.max(d23));
        assert!(d23 <= d12.max(d13));
    }

    #[test]
    fn test_qmux_branch_independent_routing() {
        let cfg = UltrametricTreeConfig::new(1, 1, 0); // 2 qubits: [Fine | Coarse]
        let mut state = vec![Complex64::default(); 4];
        // Uniform superposition across coarse: (|0> + |1>)/sqrt(2) on Coarse wire 0
        state[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0); // |00>
        state[1] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0); // |01> (y=1)

        // Branch 0 applies identity, Branch 1 applies X (Ry(pi))
        let branch_gates = [gate_ry(0.0), gate_ry(PI)];
        apply_qmux_single_qubit_gate(&mut state, &cfg, 0, &branch_gates);

        // Branch 0 (y=0, index 0) remains |00>
        // Branch 1 (y=1, index 1) flipped to |11> (index 3)
        assert!((state[0].norm_sqr() - 0.5).abs() < 1e-12);
        assert!((state[1].norm_sqr() - 0.0).abs() < 1e-12);
        assert!((state[2].norm_sqr() - 0.0).abs() < 1e-12);
        assert!((state[3].norm_sqr() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_parameter_shift_rule() {
        let cfg = UltrametricTreeConfig::new(1, 1, 0);
        let mut sorter = UltrametricTreeSorter::new(cfg, 1);
        sorter.init_seeded(123);

        let features = vec![0.3, -0.4];
        let grad = sorter.parameter_shift_branch_grad(&features, 0, false, 0, 0);
        assert_eq!(grad.len(), 2);
        let eps = 1e-6;
        sorter.branch_params[0][0] += eps;
        let s_plus = sorter.forward(&features);
        let p_plus = compute_ultrametric_branch_distribution(&s_plus, &cfg)[0][0];
        sorter.branch_params[0][0] -= 2.0 * eps;
        let s_minus = sorter.forward(&features);
        let p_minus = compute_ultrametric_branch_distribution(&s_minus, &cfg)[0][0];
        sorter.branch_params[0][0] += eps;
        let num_grad = (p_plus - p_minus) / (2.0 * eps);
        assert!((grad[0] - num_grad).abs() < 1e-4);

        // Trunk gradient verification
        let t_grad = sorter.parameter_shift_trunk_grad(&features, 0, 0);
        sorter.trunk_params[0] += eps;
        let p0_plus: f64 = compute_ultrametric_branch_distribution(&sorter.forward(&features), &cfg)[0].iter().sum();
        sorter.trunk_params[0] -= 2.0 * eps;
        let p0_minus: f64 = compute_ultrametric_branch_distribution(&sorter.forward(&features), &cfg)[0].iter().sum();
        sorter.trunk_params[0] += eps;
        let num_t_grad = (p0_plus - p0_minus) / (2.0 * eps);
        assert!((t_grad - num_t_grad).abs() < 1e-4);
    }
}
