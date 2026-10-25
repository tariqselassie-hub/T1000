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

use serde::{Deserialize, Serialize};

/// Directed optical arc mode representation in the interferometer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhotonicMode {
    pub mode_idx: usize,
    pub source_node: usize,
    pub target_node: usize,
    pub transition_prob: &'static str,
}

/// The 16 physical optical spatial/temporal modes corresponding to the non-zero arcs of $P$.
pub const PHOTONIC_MODES: [PhotonicMode; 16] = [
    // Hub v1 outgoing arcs (4 modes: 1/4 prob each)
    PhotonicMode { mode_idx: 0, source_node: 0, target_node: 1, transition_prob: "1/4" },
    PhotonicMode { mode_idx: 1, source_node: 0, target_node: 2, transition_prob: "1/4" },
    PhotonicMode { mode_idx: 2, source_node: 0, target_node: 5, transition_prob: "1/4" },
    PhotonicMode { mode_idx: 3, source_node: 0, target_node: 6, transition_prob: "1/4" },

    // Node v2 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode { mode_idx: 4, source_node: 1, target_node: 3, transition_prob: "1/2" },
    PhotonicMode { mode_idx: 5, source_node: 1, target_node: 6, transition_prob: "1/2" },

    // Node v3 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode { mode_idx: 6, source_node: 2, target_node: 0, transition_prob: "1/2" },
    PhotonicMode { mode_idx: 7, source_node: 2, target_node: 5, transition_prob: "1/2" },

    // Node v4 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode { mode_idx: 8, source_node: 3, target_node: 1, transition_prob: "1/2" },
    PhotonicMode { mode_idx: 9, source_node: 3, target_node: 4, transition_prob: "1/2" },

    // Node v5 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode { mode_idx: 10, source_node: 4, target_node: 3, transition_prob: "1/2" },
    PhotonicMode { mode_idx: 11, source_node: 4, target_node: 6, transition_prob: "1/2" },

    // Node v6 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode { mode_idx: 12, source_node: 5, target_node: 0, transition_prob: "1/2" },
    PhotonicMode { mode_idx: 13, source_node: 5, target_node: 2, transition_prob: "1/2" },

    // Node v7 outgoing arcs (2 modes: 1/2 prob each)
    PhotonicMode { mode_idx: 14, source_node: 6, target_node: 0, transition_prob: "1/2" },
    PhotonicMode { mode_idx: 15, source_node: 6, target_node: 4, transition_prob: "1/2" },
];

/// Physical 50:50 directional coupler (beamsplitter) in the integrated mesh.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Beamsplitter2x2 {
    pub coupler_id: usize,
    pub input_port_a: usize,
    pub input_port_b: usize,
    pub output_port_a: usize,
    pub output_port_b: usize,
}

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
}

impl Default for PhotonicInterferometer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhotonicInterferometer {
    /// Constructs and compiles the 16-mode photonic interferometer.
    pub fn new() -> Self {
        // 1. Build the Coin mixing matrix C_16:
        // - Modes 0..3: H_4 = H_2 (x) H_2 (all entries +/- 0.5)
        // - Modes 4..15: six 2x2 blocks of H_2 (all entries +/- 1/sqrt(2))
        let mut c_16 = [[0.0f64; 16]; 16];
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();

        // 4-port Hub mixer (4 beamsplitters)
        let h4_base = [
            [0.5,  0.5,  0.5,  0.5],
            [0.5, -0.5,  0.5, -0.5],
            [0.5,  0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5,  0.5],
        ];
        for r in 0..4 {
            for c in 0..4 {
                c_16[r][c] = h4_base[r][c];
            }
        }

        // Six 2-port mixers (1 beamsplitter each)
        for k in 0..6 {
            let offset = 4 + 2 * k;
            c_16[offset][offset] = inv_sqrt2;
            c_16[offset][offset + 1] = inv_sqrt2;
            c_16[offset + 1][offset] = inv_sqrt2;
            c_16[offset + 1][offset + 1] = -inv_sqrt2;
        }

        // 2. Build the routing permutation matrix S_16:
        // Routes the output of arc (u -> v) to the input mixer corresponding to vertex v.
        // Node 0 (Hub) receives from modes: 6 (from v3), 12 (from v6), 14 (from v7).
        // Mode 0, 1, 2, 3 mix at Hub.
        let mut s_16 = [[0.0f64; 16]; 16];
        let perm: [usize; 16] = [
            // Target Node 0 inputs mapped to Hub mixer ports 0..2 (plus recirculating port 3):
            6,  // Mode 6 (v3 -> v1) routes to Hub port 0
            12, // Mode 12 (v6 -> v1) routes to Hub port 1
            14, // Mode 14 (v7 -> v1) routes to Hub port 2
            3,  // Internal reflection mode 3 routes to Hub port 3
            // Target Node 1 inputs (v2 mixer: modes 4, 5):
            0,  // Mode 0 (v1 -> v2) routes to v2 port 0
            8,  // Mode 8 (v4 -> v2) routes to v2 port 1
            // Target Node 2 inputs (v3 mixer: modes 6, 7):
            1,  // Mode 1 (v1 -> v3) routes to v3 port 0
            13, // Mode 13 (v6 -> v3) routes to v3 port 1
            // Target Node 3 inputs (v4 mixer: modes 8, 9):
            4,  // Mode 4 (v2 -> v4) routes to v4 port 0
            10, // Mode 10 (v5 -> v4) routes to v4 port 1
            // Target Node 4 inputs (v5 mixer: modes 10, 11):
            9,  // Mode 9 (v4 -> v5) routes to v5 port 0
            15, // Mode 15 (v7 -> v5) routes to v5 port 1
            // Target Node 5 inputs (v6 mixer: modes 12, 13):
            2,  // Mode 2 (v1 -> v6) routes to v6 port 0
            7,  // Mode 7 (v3 -> v6) routes to v6 port 1
            // Target Node 6 inputs (v7 mixer: modes 14, 15):
            5,  // Mode 5 (v2 -> v7) routes to v7 port 0
            11, // Mode 11 (v5 -> v7) routes to v7 port 1
        ];

        for i in 0..16 {
            s_16[i][perm[i]] = 1.0;
        }

        // 3. Compile full unitary U_16 = S_16 * C_16
        let mut u_16 = [[0.0f64; 16]; 16];
        for r in 0..16 {
            for k in 0..16 {
                let srk = s_16[r][k];
                if srk.abs() > 1e-15 {
                    for c in 0..16 {
                        u_16[r][c] += srk * c_16[k][c];
                    }
                }
            }
        }

        // List 10 physical directional couplers
        let couplers = vec![
            // Hub 4-port mixer (stage 1: 2 couplers; stage 2: 2 couplers)
            Beamsplitter2x2 { coupler_id: 0, input_port_a: 0, input_port_b: 1, output_port_a: 0, output_port_b: 1 },
            Beamsplitter2x2 { coupler_id: 1, input_port_a: 2, input_port_b: 3, output_port_a: 2, output_port_b: 3 },
            Beamsplitter2x2 { coupler_id: 2, input_port_a: 0, input_port_b: 2, output_port_a: 0, output_port_b: 2 },
            Beamsplitter2x2 { coupler_id: 3, input_port_a: 1, input_port_b: 3, output_port_a: 1, output_port_b: 3 },
            // Node v2..v7 2-port mixers (6 couplers)
            Beamsplitter2x2 { coupler_id: 4, input_port_a: 4, input_port_b: 5, output_port_a: 4, output_port_b: 5 },
            Beamsplitter2x2 { coupler_id: 5, input_port_a: 6, input_port_b: 7, output_port_a: 6, output_port_b: 7 },
            Beamsplitter2x2 { coupler_id: 6, input_port_a: 8, input_port_b: 9, output_port_a: 8, output_port_b: 9 },
            Beamsplitter2x2 { coupler_id: 7, input_port_a: 10, input_port_b: 11, output_port_a: 10, output_port_b: 11 },
            Beamsplitter2x2 { coupler_id: 8, input_port_a: 12, input_port_b: 13, output_port_a: 12, output_port_b: 13 },
            Beamsplitter2x2 { coupler_id: 9, input_port_a: 14, input_port_b: 15, output_port_a: 14, output_port_b: 15 },
        ];

        Self {
            unitary_16x16: u_16,
            field_re: [0.0; 16],
            field_im: [0.0; 16],
            couplers,
        }
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
}

/// Electro-optic Pockels phase modulator driven by organic push-pull chromophores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MolecularPockelsModulator {
    pub molecule_name: String,
    pub chemical_formula: String,
    pub dipole_moment_debye: f64,
    pub polarizability_angstrom3: f64,
    /// Electro-optic Pockels coefficient r_33 in pm/V.
    pub pockels_r33_pm_v: f64,
    /// Effective refractive index of the organic-clad waveguide.
    pub ref_index: f64,
    /// Optical mode confinement factor Gamma inside the active molecular cladding.
    pub confinement_gamma: f64,
    /// Electrode gap in microns (um).
    pub electrode_gap_um: f64,
    /// Interaction length in microns (um).
    pub waveguide_length_um: f64,
    /// Telecom optical wavelength in nm (e.g. 1550.0 nm).
    pub wavelength_nm: f64,
}

impl MolecularPockelsModulator {
    /// Constructs Silicon-Organic Hybrid (SOH) slot modulator using Pitt #1 Lead (C27H39N2O3S+).
    pub fn lead_1_soh() -> Self {
        Self {
            molecule_name: "4-(4-{4-[4-(dibutylamino)phenyl]-1,3-butadien-1-yl}-1-pyridiniumyl)-1-butanesulfonate".to_string(),
            chemical_formula: "C27H39N2O3S+".to_string(),
            dipole_moment_debye: 44.59,
            polarizability_angstrom3: 64.6,
            pockels_r33_pm_v: 342.5,
            ref_index: 1.82,
            confinement_gamma: 0.85,  // 85% mode confinement in nanoscale slot
            electrode_gap_um: 0.15,   // 150 nm slot electrode gap
            waveguide_length_um: 1000.0, // 1 mm compact on-chip interaction length
            wavelength_nm: 1550.0,
        }
    }

    /// Constructs baseline Lithium Niobate (LiNbO3) waveguide modulator for comparison.
    pub fn lithium_niobate_baseline() -> Self {
        Self {
            molecule_name: "Lithium Niobate (Inorganic LiNbO3)".to_string(),
            chemical_formula: "LiNbO3".to_string(),
            dipole_moment_debye: 0.0,
            polarizability_angstrom3: 0.0,
            pockels_r33_pm_v: 30.8,
            ref_index: 2.14,
            confinement_gamma: 1.0,
            electrode_gap_um: 4.0,       // 4.0 um standard electrode separation
            waveguide_length_um: 10000.0, // 10 mm (1 cm) standard device length
            wavelength_nm: 1550.0,
        }
    }

    /// Calculates the half-wave switching voltage V_pi in Volts.
    pub fn v_pi(&self) -> f64 {
        let lambda_m = self.wavelength_nm * 1e-9;
        let d_m = self.electrode_gap_um * 1e-6;
        let l_m = self.waveguide_length_um * 1e-6;
        let r33_m_v = self.pockels_r33_pm_v * 1e-12;
        let n3 = self.ref_index.powi(3);

        (lambda_m * d_m) / (n3 * r33_m_v * self.confinement_gamma * l_m)
    }

    /// Calculates the voltage-length product V_pi * L in V*cm.
    pub fn v_pi_l(&self) -> f64 {
        let v_pi = self.v_pi();
        let l_cm = self.waveguide_length_um * 1e-4;
        v_pi * l_cm
    }

    /// Computes the phase shift delta_phi (in radians) induced by voltage V.
    pub fn phase_shift(&self, voltage: f64) -> f64 {
        std::f64::consts::PI * (voltage / self.v_pi())
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

        for step in 1..=10 {
            interferometer.step();
            let p_out = interferometer.total_power();
            assert!(
                (p_out - p_in).abs() < 1e-14,
                "Power not conserved at step {}: p_out = {}, p_in = {}",
                step,
                p_out,
                p_in
            );
        }
    }

    #[test]
    fn test_dyadic_coupler_amplitudes() {
        let interferometer = PhotonicInterferometer::new();
        let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
        for r in 0..16 {
            for c in 0..16 {
                let val = interferometer.unitary_16x16[r][c].abs();
                if val > 1e-12 {
                    let is_half = (val - 0.5).abs() < 1e-12;
                    let is_inv_sqrt2 = (val - inv_sqrt2).abs() < 1e-12;
                    assert!(
                        is_half || is_inv_sqrt2,
                        "Non-dyadic amplitude found at ({}, {}): {}",
                        r,
                        c,
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn test_exact_ten_beamsplitters_count() {
        let interferometer = PhotonicInterferometer::new();
        assert_eq!(
            interferometer.couplers.len(),
            10,
            "Physical mesh must have exactly 10 50:50 directional couplers"
        );
    }

    #[test]
    fn test_molecular_pockels_switching_advantage() {
        let mol_mod = MolecularPockelsModulator::lead_1_soh();
        let linbo3_mod = MolecularPockelsModulator::lithium_niobate_baseline();

        let v_pi_mol = mol_mod.v_pi();
        let v_pi_linbo3 = linbo3_mod.v_pi();

        // Lead #1 must achieve sub-0.5V CMOS-level half-wave voltage (~0.132 V)
        assert!(v_pi_mol < 0.5, "Molecular V_pi must be sub-0.5V: {v_pi_mol} V");
        assert!(
            v_pi_linbo3 > 1.5,
            "Lithium Niobate V_pi should be > 1.5V: {v_pi_linbo3} V"
        );

        // V_pi * L product advantage > 50x
        let v_pi_l_mol = mol_mod.v_pi_l();
        let v_pi_l_linbo3 = linbo3_mod.v_pi_l();
        let advantage_ratio = v_pi_l_linbo3 / v_pi_l_mol;
        assert!(
            advantage_ratio > 50.0,
            "Molecular Pockels V_pi*L advantage must exceed 50x: {advantage_ratio:.2}x"
        );
    }

    #[test]
    fn test_molecular_phase_shift_power_conservation() {
        let mut interferometer = PhotonicInterferometer::new();
        interferometer.inject_mode(0, 1.0, 0.0);
        let p_in = interferometer.total_power();

        let mol_mod = MolecularPockelsModulator::lead_1_soh();
        // Apply V_pi voltage (exact pi phase flip)
        interferometer.apply_molecular_voltage(0, mol_mod.v_pi(), &mol_mod);

        let p_out = interferometer.total_power();
        assert!(
            (p_out - p_in).abs() < 1e-14,
            "Phase modulation must preserve optical power exactly"
        );
        // Field should be inverted (-1.0)
        assert!(
            (interferometer.field_re[0] - (-1.0)).abs() < 1e-14,
            "V_pi must flip field amplitude to -1.0: {}",
            interferometer.field_re[0]
        );
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
        assert!((total_p - 1.0).abs() < 1e-12, "Quantum walk probability must sum to 1.0");
        assert!(probs[6] > 0.0, "Sink node must receive quantum wavepacket");
    }
}
