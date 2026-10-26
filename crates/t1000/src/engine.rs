// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Sovereign Master Engine Facade (`T1000Engine`)
//!
//! Provides a unified, thread-safe high-level API coordinating all quantum subsystems:
//! - **Canonical Circuits:** Automated execution of Grover, Bernstein-Vazirani, QPE, and Trotter chains.
//! - **Arbitrary Circuits:** DAG scheduling, single-qubit fusion, and SIMD SOA execution.
//! - **Molecular Exploration:** OpenSMILES parsing, 19D feature mapping, and VQC screening.
//! - **1D MPS Bloch-Zipper:** Cache-resident continuous probability steering across hundreds of qubits.
//! - **Heptal Qudit Waves:** Radix-7 ($d=7$) $SU(7)$ modular complementarity state spaces.
//! - **Unified Telemetry:** Single-call synthesis of coherence, chaos, and compiler metrics.

use serde::{Deserialize, Serialize};

use crate::canonical::{
    CanonicalAlgorithm, CanonicalExecutionReport, CanonicalRunner, ReadoutConfusionMatrix,
};
use crate::chemistry::{MolecularQuantumReport, Period12Molecule, Period12MoleculeMetadata};
use crate::clifford_t::{synthesize_and_profile, CliffordTSynthesisReport};
use crate::coherence::NoiseChannelParams;
use crate::compiler::CircuitBuilder;
use crate::heptal_wave::{HeptalZipperConfig, HeptalZipperEngine, HeptalZipperReport};
use crate::smiles::{
    SmilesError, SmilesExplorer, SmilesFeatures, SmilesFeaturizer, SmilesPropertyProfile,
};
use crate::state::QuantumState;
use crate::telemetry::QuantumTelemetryReport;
use crate::zipper::{BlochZipperEngine, ZipperConfig, ZipperExecutionReport};

/// Static description of the 1000_QPU design on the Basys 3 (`fpga/1000_qpu`). It does not probe the
/// board; `scripts/fpga/verify_1000_qpu.py` is the check that the hardware answers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FpgaDeviceInfo {
    pub target_device: String,
    pub bram_memory_kb: usize,
    pub dsp_slices: usize,
    pub uart_port: String,
    pub baud_rate: u32,
    pub is_emulated: bool,
}

/// System manifest summarizing active architecture capabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemManifestReport {
    pub version: String,
    pub author: String,
    pub license: String,
    pub sim_backend: String,
    pub fpga_target: String,
    pub max_dense_qubits: usize,
    pub zipper_max_qubits: usize,
    pub plaquette_2d_max_qubits: usize,
    pub heptal_qudit_radix: usize,
    pub period_molecules_max_qubits: usize,
    pub clifford_t_synthesis: bool,
    pub avx_vnni_accelerated: bool,
    pub igpu_bridge_available: bool,
    pub igpu_device_name: Option<String>,
}

/// Master engine facade coordinating all T-1000 quantum simulation and screening pipelines.
#[derive(Debug, Clone)]
pub struct T1000Engine {
    pub seed: u64,
    pub default_noise: Option<NoiseChannelParams>,
    pub default_confusion: Option<ReadoutConfusionMatrix>,
    explorer: SmilesExplorer,
}

impl Default for T1000Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl T1000Engine {
    /// Constructs a master engine with default configuration (`DEFAULT_SEED`, 10-qubit chemical VQC).
    pub fn new() -> Self {
        Self::new_seeded(crate::DEFAULT_SEED)
    }

    /// Constructs a master engine with an explicit deterministic PRNG seed.
    pub fn new_seeded(seed: u64) -> Self {
        let explorer = SmilesExplorer::new(10, 2);
        Self {
            seed,
            default_noise: None,
            default_confusion: None,
            explorer,
        }
    }

    /// Configures global open-system noise parameters.
    pub fn with_noise(mut self, noise: NoiseChannelParams) -> Self {
        self.default_noise = Some(noise);
        self
    }

    /// Configures global readout measurement confusion matrix.
    pub fn with_confusion(mut self, confusion: ReadoutConfusionMatrix) -> Self {
        self.default_confusion = Some(confusion);
        self
    }

    // =========================================================================
    // 1. Canonical Algorithmic Execution
    // =========================================================================

    /// Executes any canonical quantum algorithm through compilation, optional noise, and readout mitigation.
    pub fn run_canonical(&self, algorithm: CanonicalAlgorithm) -> CanonicalExecutionReport {
        let mut runner = CanonicalRunner::new(algorithm).with_seed(self.seed);
        if let Some(noise) = self.default_noise {
            runner = runner.with_noise(noise);
        }
        if let Some(confusion) = self.default_confusion {
            runner = runner.with_readout_mitigation(confusion);
        }
        runner.execute()
    }

    // =========================================================================
    // 2. Arbitrary Circuit DAG Simulation
    // =========================================================================

    /// Simulates a compiled circuit DAG and synthesizes an immutable `QuantumTelemetryReport`.
    pub fn simulate_circuit(
        &self,
        builder: &CircuitBuilder,
    ) -> (QuantumState, QuantumTelemetryReport) {
        let num_qubits = builder.dag.num_qubits;
        let dim = 1 << num_qubits;
        let mut state = QuantumState::ground(dim);

        let t0 = std::time::Instant::now();
        let compiled = if let Some(ref noise) = self.default_noise {
            builder.execute_with_noise(&mut state, noise, 0.05, self.seed)
        } else {
            builder.execute(&mut state)
        };
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        let telemetry = QuantumTelemetryReport::from_state(&state, num_qubits)
            .with_porter_thomas(&state)
            .with_compiler(compiled.telemetry)
            .with_execution_time(elapsed_us);

        (state, telemetry)
    }

    // =========================================================================
    // 3. Molecular Latent Exploration & ZINC Screening
    // =========================================================================

    /// Extracts the 19-dimensional continuous molecular descriptor vector from an OpenSMILES string.
    pub fn featurize_smiles(&self, smiles: &str) -> Result<SmilesFeatures, SmilesError> {
        SmilesFeaturizer::featurize(smiles)
    }

    /// Evaluates the quantum chemical profile (logP, QED, SAS) of a single molecule.
    pub fn evaluate_molecule(&self, smiles: &str) -> Result<SmilesPropertyProfile, SmilesError> {
        self.explorer.evaluate_smiles(smiles)
    }

    /// Screens a batch of SMILES molecules, returning sorted candidates and rejected parse errors.
    pub fn screen_molecules(
        &self,
        smiles_list: &[&str],
    ) -> (Vec<SmilesPropertyProfile>, Vec<(String, SmilesError)>) {
        self.explorer.screen_candidates(smiles_list)
    }

    /// Computes the quantum Hilbert space state similarity (fidelity) between two SMILES strings.
    pub fn molecule_similarity(&self, smiles_a: &str, smiles_b: &str) -> Result<f64, SmilesError> {
        self.explorer.quantum_smiles_similarity(smiles_a, smiles_b)
    }

    // =========================================================================
    // 4. 1D MPS Linear Bloch-Zipper Scaling
    // =========================================================================

    /// Executes the 1D Matrix Product State (MPS) linear Bloch-zipper engine across `chain_length` qubits.
    pub fn run_bloch_zipper(&self, chain_length: usize) -> ZipperExecutionReport {
        let config = ZipperConfig::default();
        let engine = BlochZipperEngine::new(config);
        engine.run_chain(chain_length)
    }

    // =========================================================================
    // 5. Heptal Qudit Wave (SU(7))
    // =========================================================================

    /// Evolves a chain of `chain_length` radix-7 ($d=7$) qudits under $SU(7)$ modular complementarity.
    pub fn run_heptal_wave(&self, chain_length: usize) -> HeptalZipperReport {
        let config = HeptalZipperConfig::default();
        let engine = HeptalZipperEngine::new(config);
        engine.run_chain(chain_length)
    }

    // =========================================================================
    // 6. Period 1 & 2 Molecular Simulation (up to 18 Qubits)
    // =========================================================================

    /// Simulates a Period 1 or 2 molecule in second-quantized spin-orbital space up to 18 qubits.
    pub fn simulate_period_molecule(&self, mol: Period12Molecule) -> MolecularQuantumReport {
        crate::chemistry::simulate_period_molecule(mol)
    }

    /// Lists metadata and orbital specifications for all supported Period 1 & 2 molecules.
    pub fn list_period_molecules(&self) -> Vec<Period12MoleculeMetadata> {
        crate::chemistry::list_period_molecules()
    }

    // =========================================================================
    // 7. Fault-Tolerant Clifford+T Synthesis
    // =========================================================================

    /// Synthesizes an arbitrary single-qubit rotation (Rz, Rx, Ry) into an optimal Clifford+T sequence.
    pub fn synthesize_clifford_t(
        &self,
        axis: &str,
        theta: f64,
        eps: f64,
    ) -> CliffordTSynthesisReport {
        synthesize_and_profile(axis, theta, eps)
    }

    // =========================================================================
    // 8. Hardware Silicon Coprocessor Status
    // =========================================================================

    /// Describes the FPGA design (resources from the Vivado 2026.1 implementation report).
    pub fn fpga_status(&self) -> FpgaDeviceInfo {
        FpgaDeviceInfo {
            target_device: "AMD Artix-7 XC7A35T-1CPG236C (Basys 3)".to_string(),
            bram_memory_kb: 128,
            dsp_slices: 40,
            uart_port: "COM6".to_string(),
            baud_rate: 3_000_000,
            is_emulated: false,
        }
    }

    /// Hardware-accelerated 8-stabilizer syndrome extraction for [[9, 1, 3]] surface code.
    pub fn extract_surface_syndromes(&self, error_x: u16, error_z: u16) -> [i8; 8] {
        crate::qec::vnni::SurfaceCode9QVnniMatrix::new().extract_syndromes_vnni(error_x, error_z)
    }

    /// Returns an instantiated Zero-Copy Intel Graphics Co-Processor Bridge.
    pub fn igpu_bridge(&self) -> crate::gpu::IGpuBridge {
        crate::gpu::IGpuBridge::default()
    }

    /// Executes the 2D Plaquette-Zipper tree contraction across a 2D gauge lattice.
    pub fn run_plaquette_zipper_2d(
        &self,
        lattice: crate::plaquette::PlaquetteLattice,
    ) -> crate::plaquette::PlaquetteZipperReport {
        let zipper = crate::plaquette::PlaquetteZipper2D::new(lattice);
        zipper.run_tree_contraction(self.seed as usize)
    }

    /// Evaluates discrete 2D Stokes theorem invariance over measurement shots.
    pub fn test_stokes_pressure(
        &self,
        lattice: crate::plaquette::PlaquetteLattice,
        shots: &[usize],
    ) -> crate::plaquette::StokesPressureReport {
        let engine = crate::plaquette::DiscreteStokesEngine::new(lattice);
        engine.stress_test_shots(shots)
    }

    /// Returns the system manifest and active capability profile.
    pub fn system_manifest(&self) -> SystemManifestReport {
        let igpu = crate::gpu::IGpuBridge::default();
        let igpu_avail = igpu.is_available();
        let igpu_device = igpu.device_name().map(|s| s.to_string());
        let vnni_active = crate::qec::vnni::VnniBackend::detect().is_accelerated();

        SystemManifestReport {
            version: env!("CARGO_PKG_VERSION").to_string(),
            author: "Terrence A. Jones SR".to_string(),
            license: "MIT OR Apache-2.0".to_string(),
            sim_backend: "256-bit AVX2 + FMA SOA".to_string(),
            fpga_target: "Artix-7 1000_QPU (14 qubits, 2 x 64 KB BRAM banks, 40 DSP48E1)"
                .to_string(),
            max_dense_qubits: 26,
            zipper_max_qubits: 200,
            plaquette_2d_max_qubits: 2000,
            heptal_qudit_radix: 7,
            period_molecules_max_qubits: 18,
            clifford_t_synthesis: true,
            avx_vnni_accelerated: vnni_active,
            igpu_bridge_available: igpu_avail,
            igpu_device_name: igpu_device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_master_engine_canonical_and_telemetry() {
        let engine = T1000Engine::new();
        let report = engine.run_canonical(CanonicalAlgorithm::BernsteinVazirani {
            n_inputs: 3,
            hidden_mask: 0b101,
        });
        assert_eq!(report.num_qubits, 4);
        assert!(report.success_probability > 0.99);

        let manifest = engine.system_manifest();
        assert_eq!(manifest.author, "Terrence A. Jones SR");
        assert_eq!(manifest.heptal_qudit_radix, 7);
    }

    #[test]
    fn test_master_engine_smiles_and_zipper() {
        let engine = T1000Engine::new();
        let aspirin = "CC(=O)Oc1ccccc1C(=O)O";
        let features = engine
            .featurize_smiles(aspirin)
            .expect("Aspirin featurize failed");
        assert_eq!(features.heavy_atoms, 13);

        let profile = engine
            .evaluate_molecule(aspirin)
            .expect("Aspirin evaluation failed");
        assert!(profile.qed > 0.0 && profile.qed <= 1.0);

        let zipper_report = engine.run_bloch_zipper(14);
        assert_eq!(zipper_report.total_qubits, 14);
        assert!((zipper_report.mean_anti_correlation_fidelity - 1.0).abs() < 1e-6);

        let heptal_report = engine.run_heptal_wave(14);
        assert_eq!(heptal_report.total_qudits, 14);
        assert!((heptal_report.mean_complementarity_fidelity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_master_engine_period_1_2_molecules() {
        let engine = T1000Engine::new();
        let mols = engine.list_period_molecules();
        assert_eq!(mols.len(), 9);

        // Verify H2 (4 qubits)
        let h2_rep = engine.simulate_period_molecule(Period12Molecule::H2);
        assert_eq!(h2_rep.qubits, 4);
        assert_eq!(h2_rep.electrons, 2);
        assert!(h2_rep.ground_state_energy < -1.13);

        // Verify Methane CH4 (flagship 18 qubits)
        let ch4_rep = engine.simulate_period_molecule(Period12Molecule::CH4);
        assert_eq!(ch4_rep.qubits, 18);
        assert_eq!(ch4_rep.electrons, 10);
        assert_eq!(ch4_rep.spin_orbital_labels.len(), 18);
        assert_eq!(ch4_rep.orbital_occupations.len(), 18);
        assert!(ch4_rep.ground_state_energy < -40.0);
        assert!(ch4_rep.total_orbital_entropy > 0.0);
    }

    #[test]
    fn test_master_engine_vnni_and_igpu() {
        let engine = T1000Engine::new();
        let manifest = engine.system_manifest();
        // Hardware capability flags must be resolved
        assert_eq!(manifest.author, "Terrence A. Jones SR");

        // Verify surface code syndrome extraction through engine facade
        let syndromes_no_error = engine.extract_surface_syndromes(0, 0);
        assert_eq!(syndromes_no_error, [1; 8]); // All stabilizers in +1 eigenspace

        let syndromes_x0 = engine.extract_surface_syndromes(1 << 0, 0);
        // Error X on qubit 0 anticommutes with Sz3 = Z0 Z3 (index 2)
        assert_eq!(syndromes_x0[2], -1);

        // Verify iGPU bridge access through facade
        let bridge = engine.igpu_bridge();
        if bridge.is_available() {
            assert!(manifest.igpu_bridge_available);
            assert!(manifest.igpu_device_name.is_some());
        }
    }
}
