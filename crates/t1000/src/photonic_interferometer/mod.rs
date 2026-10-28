// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Native Photonic Interferometer Unitary Compiler (Dyadic Beamsplitter Mesh)
//!
//! Compiles the 7-node directed graph transition matrix $P$ into a native linear optical
//! interferometer operating over 16 directed optical spatial/temporal modes:
//!
//! ### Zero-Analog Phase Advantage
//! Unlike general unitary synthesis (Reck/Clements schemes) which require arbitrary analog
//! phase shifters $\phi \in [0, 2\pi)$ prone to thermal and electro-optic drift, the directed
//! graph $P$ requires **exclusively balanced 50:50 directional couplers (beamsplitters)**:
//! - **Central Hub $v_1$ (4 outgoing modes):** Synthesized by a $4 \times 4$ Hadamard mesh
//!   $H_4 = H_2 \otimes H_2$ requiring **four 50:50 beamsplitters**, producing exact transmission
//!   amplitudes of $\pm 1/2$.
//! - **Rim & Intermediate Nodes $v_2 \dots v_7$ (2 outgoing modes each):** Synthesized by
//!   **six 50:50 beamsplitters** $H_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$,
//!   producing exact transmission amplitudes of $\pm 1/\sqrt{2}$.
//! - **Total Couplers:** Exactly $4 + 6 = 10$ physical 50:50 directional couplers.
//! - **Waveguide Interconnect Permutation $S_{16}$:** Lossless passive waveguide crossings
//!   routing output ports to target node input couplers.
//!
//! ### Quantum Multi-Photon & Boson Sampling
//! Supports arbitrary $N$-photon input Fock states $|n_0, \dots, n_{15}\rangle$, exact Glynn
//! matrix permanents, and Hong-Ou-Mandel (HOM) quantum interference with physical loss modeling.

pub mod fock;
pub mod loss;
pub mod mesh;
pub mod pockels;

pub use fock::{
    fock_transition_amplitude, generate_fock_basis, matrix_permanent, simulate_boson_sampling,
    FockState, HongOuMandelExperiment,
};
pub use loss::WaveguideLossModel;
pub use mesh::{compile_mesh_unitary, Beamsplitter2x2, PhotonicMode, PHOTONIC_MODES};
pub use pockels::MolecularPockelsModulator;

use serde::{Deserialize, Serialize};

/// 16-Mode Native Photonic Interferometer Engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhotonicInterferometer {
    /// 16x16 Unitary optical scattering matrix $U_{\text{opt}} = S_{16} C_{16}$.
    pub unitary_16x16: [[f64; 16]; 16],
    /// 16-element complex optical field amplitude vector (real part).
    pub field_re: [f64; 16],
    /// 16-element complex optical field amplitude vector (imag part).
    pub field_im: [f64; 16],
    /// 10 Physical 50:50 directional couplers in the mesh.
    pub couplers: Vec<Beamsplitter2x2>,
    /// Physical waveguide loss and fabrication tolerance model.
    pub loss_model: WaveguideLossModel,
}

impl Default for PhotonicInterferometer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhotonicInterferometer {
    /// Constructs and compiles the 16-mode photonic interferometer.
    pub fn new() -> Self {
        let (u_16, couplers) = compile_mesh_unitary();
        Self {
            unitary_16x16: u_16,
            field_re: [0.0; 16],
            field_im: [0.0; 16],
            couplers,
            loss_model: WaveguideLossModel::lossless(),
        }
    }

    /// Constructs interferometer with explicit physical loss parameters.
    pub fn with_loss_model(loss_model: WaveguideLossModel) -> Self {
        let mut inst = Self::new();
        inst.loss_model = loss_model;
        inst
    }

    /// Measures deviation from strict optical unitarity: $\|U^T U - I\|_{\max}$.
    pub fn unitarity_error(&self) -> f64 {
        let mut max_err = 0.0f64;
        for r in 0..16 {
            for c in 0..16 {
                let mut dot = 0.0;
                for k in 0..16 {
                    dot += self.unitary_16x16[k][r] * self.unitary_16x16[k][c];
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

    /// Injects optical field into a single input mode.
    pub fn inject_mode(&mut self, mode_idx: usize, amp_re: f64, amp_im: f64) {
        assert!(mode_idx < 16, "Mode index out of range");
        self.field_re = [0.0; 16];
        self.field_im = [0.0; 16];
        self.field_re[mode_idx] = amp_re;
        self.field_im[mode_idx] = amp_im;
    }

    /// Advances the optical fields through one full pass of the interferometer.
    pub fn step(&mut self) {
        let mut next_re = [0.0f64; 16];
        let mut next_im = [0.0f64; 16];
        for r in 0..16 {
            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for c in 0..16 {
                let u = self.unitary_16x16[r][c];
                acc_re += u * self.field_re[c];
                acc_im += u * self.field_im[c];
            }
            next_re[r] = acc_re;
            next_im[r] = acc_im;
        }

        // Apply physical field attenuation if configured
        let eta = self.loss_model.field_transmission_factor(self.couplers.len());
        if (eta - 1.0).abs() > 1e-12 {
            for k in 0..16 {
                next_re[k] *= eta;
                next_im[k] *= eta;
            }
        }

        self.field_re = next_re;
        self.field_im = next_im;
    }

    /// Total optical power across all 16 waveguide modes: $P_{\text{total}} = \sum_{k=0}^{15} |E_k|^2$.
    pub fn total_power(&self) -> f64 {
        let mut p = 0.0;
        for k in 0..16 {
            p += self.field_re[k] * self.field_re[k] + self.field_im[k] * self.field_im[k];
        }
        p
    }

    /// Optical power detected at the 4 Hub (v1) output modes (modes 0..3).
    pub fn hub_power(&self) -> f64 {
        let mut p = 0.0;
        for k in 0..4 {
            p += self.field_re[k] * self.field_re[k] + self.field_im[k] * self.field_im[k];
        }
        p
    }

    /// Applies an electro-optic phase shift delta_phi to a given optical mode.
    pub fn apply_phase_shift(&mut self, mode_idx: usize, phase_rad: f64) {
        assert!(mode_idx < 16, "Mode index out of range");
        let (sin_p, cos_p) = phase_rad.sin_cos();
        let re = self.field_re[mode_idx];
        let im = self.field_im[mode_idx];
        self.field_re[mode_idx] = re * cos_p - im * sin_p;
        self.field_im[mode_idx] = re * sin_p + im * cos_p;
    }

    /// Applies voltage-driven phase modulation via a physical molecular Pockels modulator.
    pub fn apply_molecular_voltage(
        &mut self,
        mode_idx: usize,
        voltage: f64,
        modulator: &MolecularPockelsModulator,
    ) {
        let phi = modulator.phase_shift(voltage);
        self.apply_phase_shift(mode_idx, phi);
    }

    /// Simulates multi-photon Boson Sampling through the 16-mode mesh.
    pub fn sample_boson_distribution(&self, in_state: &FockState) -> Vec<(FockState, f64)> {
        simulate_boson_sampling(&self.unitary_16x16, in_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_photonic_unitarity_machine_precision() {
        let interferometer = PhotonicInterferometer::new();
        let err = interferometer.unitarity_error();
        assert!(
            err < 1e-14,
            "Photonic unitary U_16 violated unitarity: error = {}",
            err
        );
    }

    #[test]
    fn test_optical_power_conservation() {
        let mut interferometer = PhotonicInterferometer::new();
        interferometer.inject_mode(0, 1.0, 0.0);
        let p_in = interferometer.total_power();

        for _ in 1..=10 {
            interferometer.step();
            let p_now = interferometer.total_power();
            assert!(
                (p_now - p_in).abs() < 1e-14,
                "Power not conserved at step: {}",
                p_now
            );
        }
    }

    #[test]
    fn test_waveguide_loss_attenuation() {
        let loss_model = WaveguideLossModel::silicon_on_insulator();
        let mut interferometer = PhotonicInterferometer::with_loss_model(loss_model.clone());
        interferometer.inject_mode(0, 1.0, 0.0);

        let p_in = interferometer.total_power();
        interferometer.step();
        let p_out = interferometer.total_power();

        let expected_t = loss_model.total_power_transmission(10);
        assert!((p_out / p_in - expected_t).abs() < 1e-10);
        assert!(p_out < p_in, "Loss model must strictly attenuate power");
    }

    #[test]
    fn test_hong_ou_mandel_two_photon_bunching() {
        // Ideal 50:50 Hadamard coupler
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        let coupler = [
            [inv_sqrt2, inv_sqrt2],
            [inv_sqrt2, -inv_sqrt2],
        ];

        let hom = HongOuMandelExperiment::simulate_2x2(coupler);
        assert!(
            hom.coincidence_prob < 1e-14,
            "HOM coincidence must vanish: {}",
            hom.coincidence_prob
        );
        assert!(
            (hom.bunching_prob_a - 0.5).abs() < 1e-14,
            "Bunching into port A must be 0.5"
        );
        assert!(
            (hom.bunching_prob_b - 0.5).abs() < 1e-14,
            "Bunching into port B must be 0.5"
        );
        assert_eq!(hom.visibility, 1.0, "Visibility must reach 100%");
    }

    #[test]
    fn test_boson_sampling_two_photon_distribution() {
        let interferometer = PhotonicInterferometer::new();
        // Two photons: one in mode 0, one in mode 1
        let in_state = FockState::two_photon(16, 0, 1);
        let dist = interferometer.sample_boson_distribution(&in_state);

        // Hilbert space dimension: binom(16 + 2 - 1, 2) = binom(17, 2) = 136 states
        assert_eq!(dist.len(), 136);

        let total_prob: f64 = dist.iter().map(|(_, p)| p).sum();
        assert!(
            (total_prob - 1.0).abs() < 1e-10,
            "Sum of Boson Sampling output probabilities must equal 1.0: {}",
            total_prob
        );
    }

    #[test]
    fn test_molecular_phase_shift_power_conservation() {
        let mut interferometer = PhotonicInterferometer::new();
        interferometer.inject_mode(0, 1.0, 0.0);
        let p_in = interferometer.total_power();

        let mol_mod = MolecularPockelsModulator::lead_1_soh();
        interferometer.apply_molecular_voltage(0, mol_mod.v_pi(), &mol_mod);

        let p_out = interferometer.total_power();
        assert!((p_out - p_in).abs() < 1e-14);
        assert!((interferometer.field_re[0] - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_grand_loop_molecular_consistency() {
        use crate::directed_heptal_walk::{BipartiteSzegedyWalk, MolecularDipoleGraph};
        use crate::smiles::SmilesFeaturizer;

        // Step 1: Ingest Lead #1 SMILES and featurize
        let smiles_lead1 = "CCCCN(CCCC)c1ccc(/C=C/C=C/c2cc[n+](CCCCS(=O)(=O)[O-])cc2)cc1";
        let features = SmilesFeaturizer::featurize(smiles_lead1)
            .expect("Lead #1 SMILES must featurize cleanly");
        assert_eq!(features.heavy_atoms, 33);
        assert!(features.estimated_mw > 470.0 && features.estimated_mw < 475.0);

        // Step 2: Option 2 Photonic Coupling - verify SOH modulation
        let mut interferometer = PhotonicInterferometer::new();
        interferometer.inject_mode(0, 1.0, 0.0);
        let mol_mod = MolecularPockelsModulator::lead_1_soh();
        assert!(mol_mod.v_pi() < 0.5, "Sub-0.5V switching voltage");
        interferometer.apply_molecular_voltage(0, mol_mod.v_pi(), &mol_mod);
        interferometer.step();
        assert!((interferometer.total_power() - 1.0).abs() < 1e-14);

        // Step 3: Option 3 Directed Heptal Coupling - verify Szegedy walk unitarity
        let dipoles = MolecularDipoleGraph::from_pitt_leads();
        let mut walk = BipartiteSzegedyWalk::with_molecular_dipole_drift(&dipoles);
        assert!(walk.unitarity_error() < 1e-13);
        walk.set_localized(0);
        for _ in 0..15 {
            walk.step();
        }
        let probs = walk.node_probabilities();
        let total_p: f64 = probs.iter().sum();
        assert!(
            (total_p - 1.0).abs() < 1e-12,
            "Quantum walk probability must sum to 1.0"
        );
        assert!(probs[6] > 0.0, "Sink node must receive quantum wavepacket");
    }
}

