// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Physical Quantum Chip Noise Simulator
//!
//! Evaluates physical decoherence, Lindblad decay, gate infidelity, inter-chiplet seam penalties,
//! and asymmetric readout errors parameterized by a [`QuantumChip`]'s calibrated [`ChipErrorProfile`].

use serde::{Deserialize, Serialize};
use crate::device_noise::{apply_readout, depolarize, invert_readout, ReadoutError};
use super::QuantumChip;

/// Physical hardware noise execution telemetry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChipNoiseReport {
    /// Target physical quantum chip.
    pub chip: QuantumChip,
    /// Overall composite state fidelity after all noise channels ($F_{\text{total}}$).
    pub composite_fidelity: f64,
    /// Gate infidelity contribution: $(1 - p_1)^{n_1} (1 - p_2)^{n_2} (1 - p_{\text{seam}})^{n_{\text{seam}}}$.
    pub gate_fidelity: f64,
    /// Longitudinal relaxation fidelity: $\exp(-t / T_1)$.
    pub t1_fidelity: f64,
    /// Transverse dephasing fidelity: $\exp(-t / T_2^*)$.
    pub t2_fidelity: f64,
    /// Total circuit duration in nanoseconds (ns).
    pub duration_ns: f64,
    /// Number of gates executed across inter-chiplet seams.
    pub seam_crossings: usize,
    /// Readout error rate applied.
    pub readout_error: f64,
}

/// Noise simulator driven by hardware chip error budgets.
#[derive(Debug, Clone, Copy)]
pub struct ChipNoiseSimulator {
    pub chip: QuantumChip,
}

impl ChipNoiseSimulator {
    /// Initializes a noise simulator for the specified quantum chip.
    pub fn new(chip: QuantumChip) -> Self {
        Self { chip }
    }

    /// Evaluates physical noise on an ideal output probability distribution.
    ///
    /// Modulates the distribution with:
    /// 1. Depolarizing gate error ($F_{\text{gate}}$).
    /// 2. Thermal energy relaxation ($T_1$) and dephasing ($T_2^*$) across the total pulse duration.
    /// 3. Inter-chiplet modular boundary seam penalties.
    /// 4. Asymmetric tensor-product readout confusion.
    pub fn simulate_noise(
        &self,
        ideal_probs: &[f64],
        n_qubits: usize,
        n1_gates: usize,
        n2_gates: usize,
        seam_crossings: usize,
    ) -> (Vec<f64>, ChipNoiseReport) {
        assert_eq!(ideal_probs.len(), 1 << n_qubits);
        let err = self.chip.error_profile();

        // 1. Gate fidelity calculation
        let p1 = err.single_qubit_error;
        let p2 = err.two_qubit_error;
        let p_seam = err.seam_two_qubit_error();

        let intra_n2 = n2_gates.saturating_sub(seam_crossings);
        let f_gate = (1.0 - p1).powi(n1_gates as i32)
            * (1.0 - p2).powi(intra_n2 as i32)
            * (1.0 - p_seam).powi(seam_crossings as i32);

        // 2. Coherence decay calculation (T1 & T2*)
        let duration_ns = (n1_gates as f64 * err.single_qubit_duration_ns)
            + (n2_gates as f64 * err.two_qubit_duration_ns);
        let duration_micros = duration_ns * 1e-3;

        let f_t1 = (-duration_micros / err.t1_micros).exp().clamp(0.0, 1.0);
        let f_t2 = (-duration_micros / err.t2_micros).exp().clamp(0.0, 1.0);

        let composite_f = (f_gate * f_t1 * f_t2).clamp(0.0, 1.0);

        // 3. Apply depolarizing channel
        let mut noisy_dist = ideal_probs.to_vec();
        depolarize(&mut noisy_dist, composite_f);

        // 4. Apply asymmetric readout error
        let r_err = ReadoutError::typical(err.readout_error);
        apply_readout(&mut noisy_dist, n_qubits, r_err);

        let report = ChipNoiseReport {
            chip: self.chip,
            composite_fidelity: composite_f,
            gate_fidelity: f_gate,
            t1_fidelity: f_t1,
            t2_fidelity: f_t2,
            duration_ns,
            seam_crossings,
            readout_error: err.readout_error,
        };

        (noisy_dist, report)
    }

    /// Inverts readout error on the noisy distribution (readout error mitigation).
    pub fn mitigate_readout(&self, noisy_probs: &[f64], n_qubits: usize) -> Vec<f64> {
        let err = self.chip.error_profile();
        let r_err = ReadoutError::typical(err.readout_error);
        let mut mitigated = noisy_probs.to_vec();
        invert_readout(&mut mitigated, n_qubits, r_err);
        mitigated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chip_noise_simulation_on_cepheus() {
        let sim = ChipNoiseSimulator::new(QuantumChip::RigettiCepheus108Q);
        let n_qubits = 3;
        let mut ideal = vec![0.0; 1 << n_qubits];
        ideal[0] = 1.0; // Ground state |000>

        let (noisy, report) = sim.simulate_noise(&ideal, n_qubits, 10, 5, 2);

        assert_eq!(report.chip, QuantumChip::RigettiCepheus108Q);
        assert!(report.composite_fidelity < 1.0);
        assert!(report.gate_fidelity < 1.0);
        assert!(report.t1_fidelity <= 1.0);
        assert!(report.t2_fidelity <= 1.0);
        assert_eq!(report.seam_crossings, 2);
        assert!((noisy.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // Ground state probability should decrease due to noise
        assert!(noisy[0] < 1.0);

        // Mitigation should recover peak probability towards ground state
        let mitigated = sim.mitigate_readout(&noisy, n_qubits);
        assert!(mitigated[0] > noisy[0]);
    }

    #[test]
    fn test_zero_noise_on_fpga_qpu() {
        let sim = ChipNoiseSimulator::new(QuantumChip::T1000QpuArtix7);
        let n_qubits = 2;
        let mut ideal = vec![0.0; 4];
        ideal[0] = 1.0;

        let (_, report) = sim.simulate_noise(&ideal, n_qubits, 4, 2, 0);
        // T1 and T2 are effectively infinite on digital SRAM, readout is 0.0
        assert!(report.composite_fidelity > 0.99999);
        assert_eq!(report.readout_error, 0.0);
    }
}
