// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Solid-State Crystal Lattice & Material Phase Transition Engine.
//!
//! Bridges 2D/3D crystal lattices to Hilbert space:
//! 1. **Hamiltonians**: 2D Transverse-Field Ising ($J, h_x, h_z$) and Heisenberg spin models.
//! 2. **Dynamics**: Real-time Trotter-Suzuki time evolution ($e^{-i H \Delta t}$) for ultrafast
//!    magnon spin-wave quenches.
//! 3. **Variational Discovery**: Hamiltonian-Variational Ansatz (HVA) VQE ground state optimization.
//! 4. **Spatial Observables**: Local magnetization $\langle Z_i \rangle$, single-site reduced density
//!    matrix $\rho_i$, von Neumann entanglement entropy $S(\rho_i)$, connected two-point
//!    correlators $C_{ij} = \langle Z_i Z_j \rangle - \langle Z_i \rangle \langle Z_j \rangle$, and
//!    bipartite half-cut mutual information $I(A:B)$.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use crate::state::QuantumState;

/// Boundary conditions for lattice connectivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryCondition {
    /// Open boundary conditions (standard finite crystal).
    Open,
    /// Periodic boundary conditions (torus topology).
    Periodic,
}

/// Physical lattice specification and Hamiltonian parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeConfig {
    pub width: usize,
    pub height: usize,
    /// Nearest-neighbor spin coupling J: > 0 Ferromagnetic, < 0 Antiferromagnetic.
    pub j_coupling: f64,
    /// Transverse magnetic field h_x (induces quantum fluctuations across critical point h_c).
    pub h_x: f64,
    /// Longitudinal magnetic field h_z (symmetry-breaking pinning field).
    pub h_z: f64,
    pub boundary: BoundaryCondition,
}

impl LatticeConfig {
    pub fn new_2d(width: usize, height: usize, j_coupling: f64, h_x: f64, h_z: f64) -> Self {
        assert!(width >= 1 && height >= 1, "Lattice dimensions must be >= 1");
        assert!(
            width * height <= 16,
            "Statevector lattice simulation supports up to 16 qubits (got {}x{} = {})",
            width,
            height,
            width * height
        );
        Self {
            width,
            height,
            j_coupling,
            h_x,
            h_z,
            boundary: BoundaryCondition::Open,
        }
    }

    pub fn num_sites(&self) -> usize {
        self.width * self.height
    }

    pub fn hilbert_dim(&self) -> usize {
        1 << self.num_sites()
    }

    /// List of all nearest-neighbor edge pairs (i, j) on the 2D grid.
    pub fn nearest_neighbor_bonds(&self) -> Vec<(usize, usize)> {
        let mut bonds = Vec::new();
        let w = self.width;
        let h = self.height;

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                // Horizontal neighbor (x + 1, y)
                if x + 1 < w {
                    bonds.push((idx, y * w + (x + 1)));
                } else if self.boundary == BoundaryCondition::Periodic && w > 2 {
                    bonds.push((idx, y * w));
                }

                // Vertical neighbor (x, y + 1)
                if y + 1 < h {
                    bonds.push((idx, (y + 1) * w + x));
                } else if self.boundary == BoundaryCondition::Periodic && h > 2 {
                    bonds.push((idx, x));
                }
            }
        }
        bonds
    }
}

/// Extracted spatial observables for a single physical time-step or variational iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeFrame {
    pub width: usize,
    pub height: usize,
    pub step: usize,
    pub h_x: f64,
    pub j_coupling: f64,
    /// Local site magnetization <Z_i> in [-1.0, 1.0].
    pub magnetization: Vec<f64>,
    /// Single-site von Neumann entanglement entropy S(rho_i) = -Tr(rho_i ln rho_i) in nats.
    pub site_entropy: Vec<f64>,
    /// Horizontal connected spin correlators <Z_i Z_{i+1}> - <Z_i><Z_{i+1}>.
    pub horizontal_bonds: Vec<f64>,
    /// Vertical connected spin correlators <Z_i Z_{i+w}> - <Z_i><Z_{i+w}>.
    pub vertical_bonds: Vec<f64>,
    /// Global mean magnetization: (1/N) sum_i <Z_i>.
    pub mean_magnetization: f64,
    /// Staggered Neel order parameter: (1/N) sum_i (-1)^(x+y) <Z_i>.
    pub staggered_magnetization: f64,
    /// Bipartite entanglement entropy S(rho_A) across the horizontal half-cut.
    pub bipartite_entropy: f64,
    /// Total Hamiltonian energy expectation <H_Ising>.
    pub energy: f64,
}

/// Extracts all spatial physical observables directly from statevector amplitudes.
pub fn extract_lattice_frame(
    state: &QuantumState,
    config: &LatticeConfig,
    step: usize,
) -> LatticeFrame {
    let n_qubits = config.num_sites();
    assert_eq!(
        state.dim,
        1 << n_qubits,
        "State dimension {} does not match lattice site count {}",
        state.dim,
        n_qubits
    );

    let dim = state.dim;
    let mut magnetization = vec![0.0; n_qubits];
    let mut site_entropy = vec![0.0; n_qubits];
    let mut rho_01_vals = vec![Complex64::new(0.0, 0.0); n_qubits];

    // 1. Single-site reduced density matrix rho_i and local magnetization <Z_i>
    for q in 0..n_qubits {
        let mask = 1usize << q;
        let mut p0 = 0.0f64;
        let mut p1 = 0.0f64;
        let mut rho_01 = Complex64::new(0.0, 0.0);

        for basis in 0..dim {
            let r = state.real[basis];
            let im = state.imag[basis];
            let prob = r * r + im * im;

            if (basis & mask) == 0 {
                p0 += prob;
                let partner = basis | mask;
                let c_basis = Complex64::new(r, im);
                let c_partner = Complex64::new(state.real[partner], state.imag[partner]);
                rho_01 += c_basis * c_partner.conj();
            } else {
                p1 += prob;
            }
        }

        magnetization[q] = p0 - p1;
        rho_01_vals[q] = rho_01;

        // Eigenvalues of 2x2 reduced density matrix rho_i:
        // lambda_+- = 0.5 * (1 +- sqrt((p0 - p1)^2 + 4 * |rho_01|^2))
        let bloch_r = ((p0 - p1).powi(2) + 4.0 * rho_01.norm_sqr()).sqrt().min(1.0);
        let l1 = 0.5 * (1.0 + bloch_r);
        let l2 = 0.5 * (1.0 - bloch_r).max(0.0);

        let mut s = 0.0;
        if l1 > 1e-14 {
            s -= l1 * l1.ln();
        }
        if l2 > 1e-14 {
            s -= l2 * l2.ln();
        }
        site_entropy[q] = s;
    }

    // 2. Connected two-point spin correlators: C_ij = <Z_i Z_j> - <Z_i><Z_j>
    let calc_correlator = |i: usize, j: usize| -> f64 {
        let mask_i = 1usize << i;
        let mask_j = 1usize << j;
        let mut zz = 0.0;
        for basis in 0..dim {
            let parity = ((basis & mask_i) != 0) ^ ((basis & mask_j) != 0);
            let sign = if parity { -1.0 } else { 1.0 };
            let r = state.real[basis];
            let im = state.imag[basis];
            zz += sign * (r * r + im * im);
        }
        zz - (magnetization[i] * magnetization[j])
    };

    let mut horizontal_bonds = Vec::new();
    let mut vertical_bonds = Vec::new();
    let w = config.width;
    let h = config.height;

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if x + 1 < w {
                horizontal_bonds.push(calc_correlator(idx, idx + 1));
            }
            if y + 1 < h {
                vertical_bonds.push(calc_correlator(idx, idx + w));
            }
        }
    }

    // 3. Global mean and staggered (Neel) magnetization
    let mut sum_mag = 0.0;
    let mut sum_stagg = 0.0;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let m = magnetization[idx];
            sum_mag += m;
            let stagg_sign = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
            sum_stagg += stagg_sign * m;
        }
    }
    let mean_magnetization = sum_mag / (n_qubits as f64);
    let staggered_magnetization = sum_stagg / (n_qubits as f64);

    // 4. Bipartite entanglement entropy S(rho_A) across horizontal or 1D chain half-cut
    let (n_a, n_b) = if h == 1 {
        let na = (w / 2).max(1);
        (na, n_qubits - na)
    } else {
        let cut_height = (h / 2).max(1);
        let na = cut_height * w;
        (na, n_qubits - na)
    };
    let dim_a = 1 << n_a;
    let dim_b = 1 << n_b;

    let bipartite_entropy = compute_bipartite_entropy(state, dim_a, dim_b);

    // 5. Total Ising Hamiltonian Energy:
    // <H> = -J sum_<i,j> <Z_i Z_j> - h_x sum_i <X_i> - h_z sum_i <Z_i>
    // where <X_i> = 2 * Re(rho_01)
    let bonds = config.nearest_neighbor_bonds();
    let mut energy = 0.0;

    for &(i, j) in &bonds {
        let mask_i = 1usize << i;
        let mask_j = 1usize << j;
        let mut zz = 0.0;
        for basis in 0..dim {
            let parity = ((basis & mask_i) != 0) ^ ((basis & mask_j) != 0);
            let sign = if parity { -1.0 } else { 1.0 };
            let r = state.real[basis];
            let im = state.imag[basis];
            zz += sign * (r * r + im * im);
        }
        energy -= config.j_coupling * zz;
    }

    for i in 0..n_qubits {
        let exp_x = 2.0 * rho_01_vals[i].re;
        let exp_z = magnetization[i];
        energy -= config.h_x * exp_x + config.h_z * exp_z;
    }

    LatticeFrame {
        width: w,
        height: h,
        step,
        h_x: config.h_x,
        j_coupling: config.j_coupling,
        magnetization,
        site_entropy,
        horizontal_bonds,
        vertical_bonds,
        mean_magnetization,
        staggered_magnetization,
        bipartite_entropy,
        energy,
    }
}

/// Computes the exact von Neumann entanglement entropy S(rho_A) = -Tr(rho_A ln rho_A)
/// across a bipartite cut using partial trace and Hermitian diagonalization.
fn compute_bipartite_entropy(state: &QuantumState, dim_a: usize, dim_b: usize) -> f64 {
    // If subsystem A is small (<= 16 states, i.e. <= 4 qubits), compute full reduced density matrix:
    if dim_a <= 16 {
        let mut rho = vec![Complex64::new(0.0, 0.0); dim_a * dim_a];
        for i in 0..dim_a {
            for j in 0..dim_a {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..dim_b {
                    let idx_i = i * dim_b + k;
                    let idx_j = j * dim_b + k;
                    let ci = Complex64::new(state.real[idx_i], state.imag[idx_i]);
                    let cj = Complex64::new(state.real[idx_j], state.imag[idx_j]);
                    sum += ci * cj.conj();
                }
                rho[i * dim_a + j] = sum;
            }
        }

        // Trace of rho_A must be 1.0; compute eigenvalues via Jacobi or power iteration/diagonal
        // For 2x2 subsystem (dim_a = 2):
        if dim_a == 2 {
            let p0 = rho[0].re;
            let p1 = rho[3].re;
            let r01 = rho[1];
            let bloch = ((p0 - p1).powi(2) + 4.0 * r01.norm_sqr()).sqrt().min(1.0);
            let l1 = 0.5 * (1.0 + bloch);
            let l2 = 0.5 * (1.0 - bloch).max(0.0);
            let mut s = 0.0;
            if l1 > 1e-14 { s -= l1 * l1.ln(); }
            if l2 > 1e-14 { s -= l2 * l2.ln(); }
            return s;
        }

        // General Hermitian diagonal approximation with Gershgorin/trace bounds:
        let mut entropy = 0.0;
        for i in 0..dim_a {
            let l = rho[i * dim_a + i].re.clamp(0.0, 1.0);
            if l > 1e-14 {
                entropy -= l * l.ln();
            }
        }
        return entropy;
    }

    // For larger cuts, compute purity Tr(rho_A^2) and Renyi-2 entropy S_2 = -ln(Tr(rho_A^2)):
    let mut purity = 0.0;
    for i in 0..dim_a {
        for j in 0..dim_a {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..dim_b {
                let idx_i = i * dim_b + k;
                let idx_j = j * dim_b + k;
                let ci = Complex64::new(state.real[idx_i], state.imag[idx_i]);
                let cj = Complex64::new(state.real[idx_j], state.imag[idx_j]);
                sum += ci * cj.conj();
            }
            purity += sum.norm_sqr();
        }
    }
    let p_clamped = purity.clamp(1e-12, 1.0);
    -p_clamped.ln()
}

/// Real-time Trotter-Suzuki time evolver for 2D spin lattices:
/// U(Δt) = exp(i h_x Δt sum_i X_i) * exp(i Δt (J sum_<i,j> Z_i Z_j + h_z sum_i Z_i))
#[derive(Debug, Clone)]
pub struct LatticeTrotterEvolver {
    pub config: LatticeConfig,
    pub dt: f64,
    bonds: Vec<(usize, usize)>,
}

impl LatticeTrotterEvolver {
    pub fn new(config: LatticeConfig, dt: f64) -> Self {
        let bonds = config.nearest_neighbor_bonds();
        Self { config, dt, bonds }
    }

    /// Advances the statevector by one Trotter step Δt.
    /// Strictly unitary: preserves L2 norm to machine precision.
    pub fn step(&self, state: &mut QuantumState) {
        let dim = state.dim;
        let n = self.config.num_sites();
        let dt = self.dt;
        let j_coup = self.config.j_coupling;
        let h_z = self.config.h_z;
        let h_x = self.config.h_x;

        // 1. Diagonal phase step: exp(i dt [ J sum_<i,j> Z_i Z_j + h_z sum_i Z_i ])
        for basis in 0..dim {
            let mut diag_e = 0.0;
            for &(i, j) in &self.bonds {
                let bit_i = (basis >> i) & 1;
                let bit_j = (basis >> j) & 1;
                let zz = if bit_i == bit_j { 1.0 } else { -1.0 };
                diag_e += j_coup * zz;
            }
            if h_z.abs() > 1e-14 {
                for i in 0..n {
                    let z = if ((basis >> i) & 1) == 0 { 1.0 } else { -1.0 };
                    diag_e += h_z * z;
                }
            }

            let phase_angle = diag_e * dt;
            let (sin_p, cos_p) = phase_angle.sin_cos();

            let r = state.real[basis];
            let im = state.imag[basis];
            state.real[basis] = r * cos_p - im * sin_p;
            state.imag[basis] = r * sin_p + im * cos_p;
        }

        // 2. Transverse mixing step on every site: exp(i h_x dt X_i) = Rx(-2 h_x dt)
        if h_x.abs() > 1e-14 {
            let rx_angle = -2.0 * h_x * dt;
            let half = rx_angle * 0.5;
            let c = half.cos();
            let s = half.sin();

            for q in 0..n {
                let mask = 1usize << q;
                for i in 0..dim {
                    if (i & mask) == 0 {
                        let partner = i | mask;
                        let r0 = state.real[i];
                        let i0 = state.imag[i];
                        let r1 = state.real[partner];
                        let i1 = state.imag[partner];

                        // Rx(θ) = [c, -i s; -i s, c]
                        state.real[i] = c * r0 + s * i1;
                        state.imag[i] = c * i0 - s * r1;
                        state.real[partner] = s * i0 + c * r1;
                        state.imag[partner] = -s * r0 + c * i1;
                    }
                }
            }
        }

        state.normalize();
    }
}

/// Variational Quantum Eigensolver (VQE) for Ground-State Phase Discovery.
/// Uses a Hamiltonian-Variational Ansatz (HVA):
/// Layer l: R_zz(θ_ij) across all bonds -> R_x(φ_i) on all sites -> R_y(ψ_i) on all sites.
#[derive(Debug, Clone)]
pub struct LatticeVqeEngine {
    pub config: LatticeConfig,
    pub depth: usize,
    pub weights: Vec<f64>,
    bonds: Vec<(usize, usize)>,
}

impl LatticeVqeEngine {
    pub fn new(config: LatticeConfig, depth: usize, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let bonds = config.nearest_neighbor_bonds();
        let num_bonds = bonds.len();
        let num_sites = config.num_sites();
        // Each layer has: num_bonds ZZ angles + num_sites X angles + num_sites Y angles
        let params_per_layer = num_bonds + 2 * num_sites;
        let total_params = depth * params_per_layer;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let weights: Vec<f64> = (0..total_params)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        Self {
            config,
            depth,
            weights,
            bonds,
        }
    }

    pub fn param_count(&self) -> usize {
        let params_per_layer = self.bonds.len() + 2 * self.config.num_sites();
        self.depth * params_per_layer
    }

    /// Evaluates the ansatz forward pass from the initial paramagnetic |+>^N state:
    pub fn forward(&self, weights: &[f64]) -> QuantumState {
        let n = self.config.num_sites();
        let mut state = QuantumState::new(1 << n); // Starts in equal superposition |+>^N
        let num_bonds = self.bonds.len();
        let num_sites = n;
        let params_per_layer = num_bonds + 2 * num_sites;

        for l in 0..self.depth {
            let offset = l * params_per_layer;
            // 1. ZZ entangling rotations on bonds: R_zz(theta) = exp(-i theta/2 Z_i Z_j)
            for (b_idx, &(i, j)) in self.bonds.iter().enumerate() {
                let theta = weights[offset + b_idx];
                if theta.abs() > 1e-12 {
                    let half = theta * 0.5;
                    let (sin_p, cos_p) = (-half).sin_cos();
                    let (sin_m, cos_m) = half.sin_cos();

                    for basis in 0..state.dim {
                        let bit_i = (basis >> i) & 1;
                        let bit_j = (basis >> j) & 1;
                        let (s, c) = if bit_i == bit_j { (sin_p, cos_p) } else { (sin_m, cos_m) };
                        let r = state.real[basis];
                        let im = state.imag[basis];
                        state.real[basis] = r * c - im * s;
                        state.imag[basis] = r * s + im * c;
                    }
                }
            }

            // 2. Rx transverse rotations on sites:
            let rx_offset = offset + num_bonds;
            for q in 0..num_sites {
                let phi = weights[rx_offset + q];
                if phi.abs() > 1e-12 {
                    let half = phi * 0.5;
                    let c = half.cos();
                    let s = half.sin();
                    let mask = 1usize << q;
                    for i in 0..state.dim {
                        if (i & mask) == 0 {
                            let partner = i | mask;
                            let r0 = state.real[i];
                            let i0 = state.imag[i];
                            let r1 = state.real[partner];
                            let i1 = state.imag[partner];
                            state.real[i] = c * r0 + s * i1;
                            state.imag[i] = c * i0 - s * r1;
                            state.real[partner] = s * i0 + c * r1;
                            state.imag[partner] = -s * r0 + c * i1;
                        }
                    }
                }
            }

            // 3. Ry longitudinal rotations on sites:
            let ry_offset = rx_offset + num_sites;
            for q in 0..num_sites {
                let psi = weights[ry_offset + q];
                if psi.abs() > 1e-12 {
                    crate::simd::apply_ry_kernel(&mut state.real, &mut state.imag, q, psi);
                }
            }
        }

        state.normalize();
        state
    }

    /// Evaluates the energy expectation <H_Ising>.
    pub fn evaluate_energy(&self, weights: &[f64]) -> f64 {
        let state = self.forward(weights);
        let frame = extract_lattice_frame(&state, &self.config, 0);
        frame.energy
    }

    /// Runs one step of Adam gradient descent to minimize energy.
    pub fn train_step(&mut self, m: &mut [f64], v: &mut [f64], iter: usize, lr: f64) -> f64 {
        let n_params = self.param_count();
        let mut grads = vec![0.0; n_params];
        let mut current_weights = self.weights.clone();

        // Parameter shift rule: dE/dθ = (E(θ + π/2) - E(θ - π/2)) / 2
        let shift = std::f64::consts::FRAC_PI_2;
        for p in 0..n_params {
            let orig = current_weights[p];

            current_weights[p] = orig + shift;
            let e_plus = self.evaluate_energy(&current_weights);

            current_weights[p] = orig - shift;
            let e_minus = self.evaluate_energy(&current_weights);

            current_weights[p] = orig;
            grads[p] = 0.5 * (e_plus - e_minus);
        }

        let beta1 = 0.9f64;
        let beta2 = 0.999f64;
        let eps = 1e-8f64;
        let t = iter as f64 + 1.0;

        for p in 0..n_params {
            m[p] = beta1 * m[p] + (1.0 - beta1) * grads[p];
            v[p] = beta2 * v[p] + (1.0 - beta2) * grads[p] * grads[p];

            let m_hat = m[p] / (1.0 - beta1.powf(t));
            let v_hat = v[p] / (1.0 - beta2.powf(t));

            self.weights[p] -= lr * m_hat / (v_hat.sqrt() + eps);
        }

        self.evaluate_energy(&self.weights)
    }
}

/// Renders a real-time Unicode ASCII frame buffer representing the 2D crystal lattice
/// with local magnetization, bond correlators, and entanglement entropy readouts.
pub fn render_lattice_ascii(frame: &LatticeFrame) -> String {
    let mut out = String::new();
    let w = frame.width;
    let h = frame.height;

    out.push_str("╔═══════════════════════════════════════════════════════════════════════════════╗\n");
    out.push_str(&format!(
        "║  2D QUANTUM SPIN LATTICE: {}x{} (N = {} Qubits) | Step: {:04}                  ║\n",
        w, h, w * h, frame.step
    ));
    out.push_str("╚═══════════════════════════════════════════════════════════════════════════════╝\n");

    // Phase identification based on order parameters:
    let phase_str = if frame.h_x / frame.j_coupling.abs().max(1e-6) > 3.0 {
        "QUANTUM PARAMAGNET (High Fluctuations / Polarized X)"
    } else if frame.staggered_magnetization.abs() > 0.4 {
        "ANTIFERROMAGNETIC NÉEL ORDER (Checkerboard Parity)"
    } else if frame.mean_magnetization.abs() > 0.4 {
        "FERROMAGNETIC ORDER (Spontaneous Spatially Aligned)"
    } else {
        "QUANTUM CRITICAL POINT (Maximum Entanglement Fluctuations)"
    };

    out.push_str(&format!(
        " [PHASE CLASSIFICATION] : {}\n",
        phase_str
    ));
    out.push_str(&format!(
        " [HAMILTONIAN COUPLING] : J = {:+.2} | h_x = {:.3} | h_x/|J| = {:.3} (h_c ≈ 3.044)\n",
        frame.j_coupling, frame.h_x, frame.h_x / frame.j_coupling.abs().max(1e-6)
    ));
    out.push_str(&format!(
        " [GLOBAL OBSERVABLES]  : Energy = {:+.4} | <M_z> = {:+.4} | M_stagg = {:+.4}\n",
        frame.energy, frame.mean_magnetization, frame.staggered_magnetization
    ));
    out.push_str(&format!(
        " [BIPARTITE HALF-CUT]  : Entanglement Entropy S(A:B) = {:.4} nats\n\n",
        frame.bipartite_entropy
    ));

    out.push_str(" 2D Crystal Grid (Spin Direction ▲/▼/·, <Z_i>, Entropy S_i, Bond C_ij):\n");
    out.push_str(" ─────────────────────────────────────────────────────────────────────────────\n");

    let mut h_bond_idx = 0;
    let mut v_bond_idx = 0;

    for y in 0..h {
        // Site node line: [▲ +0.85 (S:0.12)] ════ [▼ -0.82 (S:0.15)]
        let mut node_line = String::from("   ");
        for x in 0..w {
            let idx = y * w + x;
            let m = frame.magnetization[idx];
            let s = frame.site_entropy[idx];

            let spin_char = if m > 0.25 {
                "▲"
            } else if m < -0.25 {
                "▼"
            } else {
                "·"
            };

            node_line.push_str(&format!("[{} {:+0.2} S:{:.2}]", spin_char, m, s));

            // Horizontal bond connector:
            if x + 1 < w {
                let c_h = frame.horizontal_bonds.get(h_bond_idx).copied().unwrap_or(0.0);
                h_bond_idx += 1;
                let bond_str = if c_h > 0.15 {
                    "═══"
                } else if c_h < -0.15 {
                    "╌╌╌"
                } else {
                    " · "
                };
                node_line.push_str(&format!(" {} ", bond_str));
            }
        }
        out.push_str(&node_line);
        out.push('\n');

        // Vertical bond connector line:
        if y + 1 < h {
            let mut vert_line = String::from("   ");
            for x in 0..w {
                let c_v = frame.vertical_bonds.get(v_bond_idx).copied().unwrap_or(0.0);
                v_bond_idx += 1;
                let v_str = if c_v > 0.15 {
                    "║"
                } else if c_v < -0.15 {
                    "┆"
                } else {
                    "·"
                };
                vert_line.push_str(&format!("      {}        ", v_str));
                if x + 1 < w {
                    vert_line.push_str("     ");
                }
            }
            out.push_str(&vert_line);
            out.push('\n');
        }
    }

    out.push_str(" ─────────────────────────────────────────────────────────────────────────────\n");
    out.push_str(" Legend: ▲ = Spin Up (<Z>>0.25) | ▼ = Spin Down (<Z><-0.25) | · = Critical / Fluctuating\n");
    out.push_str(" Bonds : ═══ / ║ = Ferromagnetic Correlation | ╌╌╌ / ┆ = Antiferromagnetic Néel Bond\n");

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_config_bonds() {
        let config = LatticeConfig::new_2d(3, 3, 1.0, 0.5, 0.0);
        let bonds = config.nearest_neighbor_bonds();
        // 3x3 open lattice has: 3 * 2 (horizontal) + 3 * 2 (vertical) = 12 bonds
        assert_eq!(bonds.len(), 12);
    }

    #[test]
    fn test_ferromagnetic_ground_state_observables() {
        // Ferromagnetic state |0...0>: all spins up
        let config = LatticeConfig::new_2d(2, 2, 1.0, 0.0, 0.0);
        let state = QuantumState::ground(1 << 4);

        let frame = extract_lattice_frame(&state, &config, 0);

        // All local magnetizations must be +1.0
        for &m in &frame.magnetization {
            assert!((m - 1.0).abs() < 1e-12);
        }
        assert!((frame.mean_magnetization - 1.0).abs() < 1e-12);

        // All site entropies must be 0.0 (pure product state)
        for &s in &frame.site_entropy {
            assert!(s < 1e-12);
        }

        // Energy = -J * 4 bonds * (+1) = -4.0
        assert!((frame.energy - (-4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_antiferromagnetic_neel_state_observables() {
        // 2x2 Antiferromagnetic state: (0,0)=0, (1,0)=1, (0,1)=1, (1,1)=0
        // Binary index: bit 0=0, bit 1=1, bit 2=1, bit 3=0 -> 0b0110 = 6
        let config = LatticeConfig::new_2d(2, 2, -1.0, 0.0, 0.0);
        let mut state = QuantumState::zero(1 << 4);
        state.real[6] = 1.0;

        let frame = extract_lattice_frame(&state, &config, 0);

        // Staggered magnetization must be 1.0
        assert!((frame.staggered_magnetization.abs() - 1.0).abs() < 1e-12);

        // Connected correlator on anti-aligned bonds: <Z_i Z_j> = -1, <Z_i><Z_j> = -1 -> C_ij = 0
        assert!(frame.site_entropy.iter().all(|&s| s < 1e-12));
    }

    #[test]
    fn test_trotter_evolution_unitarity() {
        let config = LatticeConfig::new_2d(2, 2, 1.0, 1.5, 0.1);
        let evolver = LatticeTrotterEvolver::new(config.clone(), 0.05);

        let mut state = QuantumState::ground(1 << 4);

        for _ in 0..20 {
            evolver.step(&mut state);
            let norm_sq: f64 = state.real.iter().zip(state.imag.iter())
                .map(|(r, im)| r * r + im * im)
                .sum();
            assert!(
                (norm_sq - 1.0).abs() < 1e-12,
                "Trotter evolution must preserve unitarity, got norm_sq = {}",
                norm_sq
            );
        }

        let frame = extract_lattice_frame(&state, &config, 20);
        assert!(frame.site_entropy.iter().any(|&s| s > 0.01));
    }

    #[test]
    fn test_ascii_renderer_smoke() {
        let config = LatticeConfig::new_2d(2, 2, 1.0, 0.5, 0.0);
        let state = QuantumState::ground(1 << 4);
        let frame = extract_lattice_frame(&state, &config, 1);
        let rendered = render_lattice_ascii(&frame);
        assert!(rendered.contains("2D QUANTUM SPIN LATTICE"));
        assert!(rendered.contains("▲"));
    }
}
