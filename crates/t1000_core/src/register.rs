// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::porter_thomas::{
    exact_linear_xeb, naive_linear_xeb, verify_porter_thomas, PorterThomasReport,
};

/// Canonical radix for qudits in the Deen OS / GuddaNetworks architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum QuditRadix {
    /// Binary physical carrier (d = 2, qubits)
    Binary,
    /// Canonical Deen Heptal qudit (d = 7, Z_7 finite field)
    Heptal,
    /// Custom d-dimensional qudit
    Custom(usize),
}

impl QuditRadix {
    #[inline]
    pub const fn dim(&self) -> usize {
        match self {
            QuditRadix::Binary => 2,
            QuditRadix::Heptal => 7,
            QuditRadix::Custom(d) => *d,
        }
    }

    #[inline]
    pub const fn is_heptal(&self) -> bool {
        matches!(self, QuditRadix::Heptal)
    }

    #[inline]
    pub const fn is_binary(&self) -> bool {
        matches!(self, QuditRadix::Binary)
    }
}

/// Helper function to infer qudit radix and number of sites from basis size.
pub fn infer_radix_and_sites(basis: usize) -> (QuditRadix, usize) {
    if basis == 0 {
        return (QuditRadix::Custom(0), 0);
    }
    // Check if power of 7 (Heptal)
    let mut temp = basis;
    let mut sites_7 = 0;
    while temp > 1 && temp.is_multiple_of(7) {
        temp /= 7;
        sites_7 += 1;
    }
    if temp == 1 && sites_7 > 0 {
        return (QuditRadix::Heptal, sites_7);
    }

    // Check if power of 2 (Binary)
    if basis.is_power_of_two() {
        let sites_2 = basis.trailing_zeros() as usize;
        return (QuditRadix::Binary, sites_2);
    }

    (QuditRadix::Custom(basis), 1)
}

/// Amplitude-weighted basis state vector spanning a Hilbert space of dimension `basis`.
///
/// In the Deen OS architecture:
/// - Canonical logical qudit: `QuditRadix::Heptal` (d = 7, Z_7 finite field).
/// - Physical carrier / simulation: `QuditRadix::Binary` (d = 2, qubits).
///
/// Amplitudes are unnormalized by default until `normalize()` is called.
#[derive(Debug, Clone)]
pub struct QuditRegister {
    pub amplitudes: Vec<Complex64>,
    pub radix: QuditRadix,
    pub sites: usize,
    pub basis: usize,
    pub seed: u64,
}

/// Backwards compatibility alias for the codebase.
pub type QubitRegister = QuditRegister;

impl QuditRegister {
    /// Constructs a register with uniform superposition across all basis states.
    /// Radix and site count are inferred from the basis dimension.
    pub fn new(basis: usize, seed: u64) -> Self {
        let (radix, sites) = infer_radix_and_sites(basis);
        let amp = if basis > 0 {
            Complex64::new(1.0 / (basis as f64).sqrt(), 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        };
        Self {
            amplitudes: vec![amp; basis],
            radix,
            sites,
            basis,
            seed,
        }
    }

    /// Constructs a register directly from an amplitude vector, inferring radix and sites.
    pub fn from_amplitudes(amplitudes: Vec<Complex64>, seed: u64) -> Self {
        let basis = amplitudes.len();
        let (radix, sites) = infer_radix_and_sites(basis);
        Self {
            amplitudes,
            radix,
            sites,
            basis,
            seed,
        }
    }

    /// Constructs a canonical Heptal qudit register of `sites` qudits (D = 7^sites).
    pub fn new_heptal(sites: usize, seed: u64) -> Self {
        let basis = 7usize
            .checked_pow(sites as u32)
            .expect("Heptal dimension overflow");
        let amp = Complex64::new(1.0 / (basis as f64).sqrt(), 0.0);
        Self {
            amplitudes: vec![amp; basis],
            radix: QuditRadix::Heptal,
            sites,
            basis,
            seed,
        }
    }

    /// Constructs a binary qubit register of `qubits` sites (D = 2^qubits).
    pub fn new_binary(qubits: usize, seed: u64) -> Self {
        let basis = 1usize
            .checked_shl(qubits as u32)
            .expect("Binary dimension overflow");
        let amp = Complex64::new(1.0 / (basis as f64).sqrt(), 0.0);
        Self {
            amplitudes: vec![amp; basis],
            radix: QuditRadix::Binary,
            sites: qubits,
            basis,
            seed,
        }
    }

    /// Constructs a general qudit register of `sites` sites with explicit `radix`.
    pub fn new_qudit(radix: QuditRadix, sites: usize, seed: u64) -> Self {
        let d = radix.dim();
        let basis = d
            .checked_pow(sites as u32)
            .expect("Qudit dimension overflow");
        let amp = Complex64::new(1.0 / (basis as f64).sqrt(), 0.0);
        Self {
            amplitudes: vec![amp; basis],
            radix,
            sites,
            basis,
            seed,
        }
    }

    /// Generates an ideal Haar-random scrambled pure quantum state on the unit sphere
    /// in Hilbert space dimension D = radix^sites.
    ///
    /// Sampling independent standard complex Gaussian amplitudes guarantees that the
    /// resulting measurement probabilities strictly follow the Porter-Thomas distribution.
    pub fn new_haar_random(radix: QuditRadix, sites: usize, seed: u64) -> Self {
        let d = radix.dim();
        let basis = d
            .checked_pow(sites as u32)
            .expect("Qudit dimension overflow");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut amps = Vec::with_capacity(basis);

        // Box-Muller transform to sample complex standard normal random variables
        for _ in 0..basis {
            let u1: f64 = rng.gen::<f64>().clamp(1e-15, 1.0);
            let u2: f64 = rng.gen::<f64>();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            let real = r * theta.cos();
            let imag = r * theta.sin();
            amps.push(Complex64::new(real, imag));
        }

        let mut reg = Self {
            amplitudes: amps,
            radix,
            sites,
            basis,
            seed,
        };
        reg.normalize();
        reg
    }

    #[inline]
    pub fn amplitude(&self, i: usize) -> Complex64 {
        self.amplitudes[i]
    }

    /// Probability at basis index `i` is |amplitude_i|^2.
    #[inline]
    pub fn probability(&self, i: usize) -> f64 {
        self.amplitudes[i].norm_sqr()
    }

    /// Returns the full vector of measurement probabilities for all basis states.
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sqr()).collect()
    }

    /// Normalizes the state vector in L2 norm (sum |c_x|^2 = 1).
    pub fn normalize(&mut self) {
        let norm: f64 = self
            .amplitudes
            .iter()
            .map(|a| a.norm_sqr())
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for a in &mut self.amplitudes {
                *a /= norm;
            }
        }
    }

    /// Shannon entropy (nats) of the measurement probability distribution |amp_i|^2.
    pub fn entropy(&self) -> f64 {
        self.amplitudes
            .iter()
            .map(|a| {
                let p = a.norm_sqr();
                if p > 1e-18 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Computes the exact finite-D Linear XEB fidelity for this register.
    pub fn exact_linear_xeb(&self) -> f64 {
        let probs = self.probabilities();
        exact_linear_xeb(&probs)
    }

    /// Computes the naive Linear XEB fidelity (D * sum(p^2) - 1.0).
    pub fn naive_linear_xeb(&self) -> f64 {
        let probs = self.probabilities();
        naive_linear_xeb(&probs)
    }

    /// Performs full Porter-Thomas statistical validation on this register.
    pub fn verify_porter_thomas(&self) -> PorterThomasReport {
        let probs = self.probabilities();
        verify_porter_thomas(&probs)
    }

    /// Tensor product of two registers: joint basis `self.basis * other.basis`.
    pub fn tensor_product(&self, other: &Self) -> Self {
        let basis = self.basis * other.basis;
        let mut amps = Vec::with_capacity(basis);
        for a in &self.amplitudes {
            for b in &other.amplitudes {
                amps.push(*a * *b);
            }
        }
        let (radix, sites) = if self.radix == other.radix {
            (self.radix, self.sites + other.sites)
        } else {
            (QuditRadix::Custom(basis), 1)
        };
        Self {
            amplitudes: amps,
            radix,
            sites,
            basis,
            seed: self.seed.wrapping_add(other.seed),
        }
    }

    /// Embeds a Heptal qudit register (K qudits, D = 7^K) into a physical binary qubit
    /// register (3K qubits, D = 8^K = 2^{3K}).
    ///
    /// Each Heptal qudit |0>..|6> is mapped to the 3-qubit basis |000>..|110>.
    /// State |111> (7) forms the non-computational Somatic Leakage Subspace.
    /// Returns `(binary_register, leakage_probability)`.
    pub fn embed_heptal_into_binary(&self) -> Result<(Self, f64), &'static str> {
        if self.radix != QuditRadix::Heptal {
            return Err("Register is not a Heptal qudit register");
        }
        let k = self.sites;
        if k == 0 {
            return Err("Empty qudit register");
        }

        let binary_sites = k * 3;
        let binary_basis = 1usize
            .checked_shl(binary_sites as u32)
            .ok_or("Binary physical dimension overflow")?;

        let mut binary_amps = vec![Complex64::new(0.0, 0.0); binary_basis];

        for (h_idx, &amp) in self.amplitudes.iter().enumerate() {
            // Decompose Heptal index into base-7 digits: h_idx = sum_{j=0}^{k-1} d_j * 7^(k - 1 - j)
            let mut rem = h_idx;
            let mut bin_idx = 0usize;
            for j in 0..k {
                let power_7 = 7usize.pow((k - 1 - j) as u32);
                let digit = rem / power_7;
                rem %= power_7;

                let power_8 = 8usize.pow((k - 1 - j) as u32);
                bin_idx += digit * power_8;
            }
            binary_amps[bin_idx] = amp;
        }

        let reg = Self {
            amplitudes: binary_amps,
            radix: QuditRadix::Binary,
            sites: binary_sites,
            basis: binary_basis,
            seed: self.seed,
        };

        // For a freshly embedded pure logical state, leakage is strictly 0.0
        Ok((reg, 0.0))
    }

    /// Projects a physical binary register (3K qubits, D = 8^K) back onto the
    /// canonical Heptal computational subspace (K qudits, D = 7^K).
    ///
    /// Computes the total non-computational leakage probability Lambda into any state
    /// containing a `|111>` (digit 7) site, projects the remaining amplitude onto
    /// the Heptal subspace, and renormalizes.
    /// Returns `(heptal_register, somatic_leakage)`.
    pub fn project_binary_to_heptal(binary_reg: &Self) -> Result<(Self, f64), &'static str> {
        if !binary_reg.sites.is_multiple_of(3) {
            return Err("Binary qubit count must be a multiple of 3 to embed Heptal qudits");
        }
        let k = binary_reg.sites / 3;
        let heptal_basis = 7usize
            .checked_pow(k as u32)
            .ok_or("Heptal logical dimension overflow")?;

        let mut heptal_amps = vec![Complex64::new(0.0, 0.0); heptal_basis];
        let mut leakage_prob = 0.0f64;

        for (b_idx, &amp) in binary_reg.amplitudes.iter().enumerate() {
            let mut rem = b_idx;
            let mut has_leakage = false;
            let mut h_idx = 0usize;

            for j in 0..k {
                let power_8 = 8usize.pow((k - 1 - j) as u32);
                let digit = rem / power_8;
                rem %= power_8;

                if digit >= 7 {
                    has_leakage = true;
                    break;
                }
                let power_7 = 7usize.pow((k - 1 - j) as u32);
                h_idx += digit * power_7;
            }

            if has_leakage {
                leakage_prob += amp.norm_sqr();
            } else {
                heptal_amps[h_idx] = amp;
            }
        }

        let mut heptal_reg = Self {
            amplitudes: heptal_amps,
            radix: QuditRadix::Heptal,
            sites: k,
            basis: heptal_basis,
            seed: binary_reg.seed,
        };
        heptal_reg.normalize();

        Ok((heptal_reg, leakage_prob))
    }

    /// Evaluates the total somatic leakage probability Lambda of a physical binary register
    /// into the non-computational subspace.
    pub fn somatic_leakage(binary_reg: &Self) -> f64 {
        if !binary_reg.sites.is_multiple_of(3) {
            return 1.0;
        }
        let k = binary_reg.sites / 3;
        let mut leakage = 0.0f64;

        for (b_idx, &amp) in binary_reg.amplitudes.iter().enumerate() {
            let mut rem = b_idx;
            for j in 0..k {
                let power_8 = 8usize.pow((k - 1 - j) as u32);
                let digit = rem / power_8;
                rem %= power_8;
                if digit >= 7 {
                    leakage += amp.norm_sqr();
                    break;
                }
            }
        }
        leakage
    }
}
