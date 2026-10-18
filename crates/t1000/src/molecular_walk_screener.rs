// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Continuous Topological Walk Molecular Screener
//!
//! Maps 19-dimensional SMILES chemical latent vectors into a continuous-time
//! multi-tier topological glued-tree Hamiltonian ($H_{\text{valve}}$) with
//! analog pulse shaping and intermediate Weld Zeno sentries.
//!
//! Replaces discrete gate synthesis with continuous ballistic wavepacket
//! propagation to screen molecular candidates into 4 pharmacological quadrants:
//! - Quadrant 0: Lead Candidate (High QED, Low SAS)
//! - Quadrant 1: Challenging Synthesis (High QED, High SAS)
//! - Quadrant 2: Scaffold Building Block (Low QED, Low SAS)
//! - Quadrant 3: Non-Viable (Low QED, High SAS)

use crate::cascaded_tree_valve::{CascadedValveConfig, CascadedValveHamiltonian};
use crate::coherence::NoiseChannelParams;
use crate::smiles::{SmilesError, SmilesFeaturizer};
use num_complex::Complex64;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Pharmacological classification quadrant based on QED and SAS scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PharmacologicalQuadrant {
    /// High QED (>= 0.60), Low SAS (<= 3.5): Prime drug lead candidate.
    LeadCandidate = 0,
    /// High QED (>= 0.60), High SAS (> 3.5): Potent but synthetically complex.
    ChallengingSynthesis = 1,
    /// Low QED (< 0.60), Low SAS (<= 3.5): Readily synthesized building block.
    ScaffoldBuildingBlock = 2,
    /// Low QED (< 0.60), High SAS (> 3.5): Deprioritized / unfavorable.
    NonViable = 3,
}

impl PharmacologicalQuadrant {
    /// Determines the ground-truth quadrant from measured QED and SAS.
    pub fn from_qed_sas(qed: f64, sas: f64) -> Self {
        let high_qed = qed >= 0.60;
        let low_sas = sas <= 3.50;
        match (high_qed, low_sas) {
            (true, true) => Self::LeadCandidate,
            (true, false) => Self::ChallengingSynthesis,
            (false, true) => Self::ScaffoldBuildingBlock,
            (false, false) => Self::NonViable,
        }
    }

    /// Maps quadrant index (0..3) to (coarse, fine) tree coordinates.
    pub fn to_coords(self) -> (usize, usize) {
        match self {
            Self::LeadCandidate => (0, 0),
            Self::ChallengingSynthesis => (0, 1),
            Self::ScaffoldBuildingBlock => (1, 0),
            Self::NonViable => (1, 1),
        }
    }

    /// Constructs quadrant from (coarse, fine) tree coordinates.
    pub fn from_coords(coarse: usize, fine: usize) -> Self {
        match (coarse, fine) {
            (0, 0) => Self::LeadCandidate,
            (0, _) => Self::ChallengingSynthesis,
            (_, 0) => Self::ScaffoldBuildingBlock,
            (_, _) => Self::NonViable,
        }
    }
}

/// Analog microwave and flux pulse envelope type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PulseEnvelopeKind {
    /// Standard flat continuous drive.
    Square,
    /// Flat-top Gaussian envelope eliminating spectral leakage into non-computational states.
    FlatTopGaussian { rise_time: f64, sigma: f64 },
}

/// Configuration for the Continuous Topological Molecular Screener.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MolecularWalkScreenerConfig {
    /// Depth of Tier-1 coarse routing tree (QED axis).
    pub depth1: usize,
    /// Depth of Tier-2 fine routing tree (SAS axis).
    pub depth2: usize,
    /// Base hopping parameter gamma.
    pub gamma: f64,
    /// Detuning bias scaling for chemical potential injection.
    pub detuning_scale: f64,
    /// Evolution time step dt in microseconds.
    pub dt: f64,
    /// Total transit duration t_transit in microseconds.
    pub t_transit: f64,
    /// Pulse envelope shaping for analog microwave/flux control lines.
    pub envelope: PulseEnvelopeKind,
    /// Enable intermediate Weld Zeno sentry projection.
    pub zeno_stabilization: bool,
}

impl Default for MolecularWalkScreenerConfig {
    fn default() -> Self {
        Self {
            depth1: 2,
            depth2: 2,
            gamma: 1.0,
            detuning_scale: 0.12,
            dt: 0.02,
            t_transit: 5.5,
            envelope: PulseEnvelopeKind::FlatTopGaussian {
                rise_time: 0.5,
                sigma: 0.2,
            },
            zeno_stabilization: true,
        }
    }
}

/// Single molecule screening telemetry output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoleculeScreeningResult {
    pub smiles: String,
    pub true_qed: f64,
    pub true_sas: f64,
    pub true_quadrant: PharmacologicalQuadrant,
    pub predicted_quadrant: PharmacologicalQuadrant,
    pub exit_probabilities: [f64; 4],
    pub routing_contrast: f64,
    pub is_correct: bool,
    pub is_lead_match: bool,
    pub transit_latency_us: f64,
}

/// Aggregate performance report for batch molecular screening.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchMolecularScreeningReport {
    pub total_molecules: usize,
    pub lead_candidates_ground_truth: usize,
    pub lead_candidates_predicted: usize,
    pub true_positive_leads: usize,
    pub lead_precision_pct: f64,
    pub lead_recall_pct: f64,
    pub overall_quadrant_accuracy_pct: f64,
    pub coarse_qed_accuracy_pct: f64,
    pub fine_sas_accuracy_pct: f64,
    pub mean_routing_contrast: f64,
    pub mean_latency_us: f64,
}

/// Continuous-Time Topological Quantum Walk Molecular Screening Engine.
#[derive(Debug, Clone)]
pub struct ContinuousMolecularWalkScreener {
    pub config: MolecularWalkScreenerConfig,
    valve_cfg: CascadedValveConfig,
}

impl ContinuousMolecularWalkScreener {
    /// Constructs a new molecular walk screener.
    pub fn new(config: MolecularWalkScreenerConfig) -> Self {
        let valve_cfg = CascadedValveConfig::new(config.depth1, config.depth2, 2, 2);
        Self { config, valve_cfg }
    }

    /// Computes the analog pulse envelope factor at time t.
    #[inline(always)]
    fn pulse_amplitude(&self, t: f64) -> f64 {
        match self.config.envelope {
            PulseEnvelopeKind::Square => 1.0,
            PulseEnvelopeKind::FlatTopGaussian { rise_time, sigma } => {
                let t_fall = self.config.t_transit - rise_time;
                if t < rise_time {
                    let d = t - rise_time;
                    (-0.5 * (d / sigma) * (d / sigma)).exp()
                } else if t > t_fall {
                    let d = t - t_fall;
                    (-0.5 * (d / sigma) * (d / sigma)).exp()
                } else {
                    1.0
                }
            }
        }
    }

    /// Maps normalized 19D SMILES chemical features to Hamiltonian chemical potentials.
    /// Returns (target_coarse, target_fine, qed_detuning, sas_detuning).
    pub fn map_chemical_potentials(features: &[f64; 19]) -> (usize, usize, f64, f64) {
        let arom = features[1] + features[3] + features[5];
        let het = features[18];
        let mw = features[17];
        let rings = features[15];
        let chiral = features[14];
        let branches = features[13];
        let ha_norm = features[16];

        // Linear physical descriptor projection for QED (Drug-Likeness >= 0.60)
        let score_qed = 1.564
            - 0.127 * arom
            - 0.737 * het
            - 0.409 * mw
            - 0.167 * rings
            - 0.165 * chiral
            + 0.638 * branches
            - 0.563 * ha_norm;
        let coarse = if score_qed >= 0.50 { 0 } else { 1 };
        let detune_qed = (score_qed - 0.50).abs().min(1.0) * 0.15;

        // Linear physical descriptor projection for SAS (Easy Synthesizability <= 3.50)
        let score_sas = 0.077
            + 0.735 * arom
            + 0.434 * het
            + 0.576 * mw
            - 1.341 * rings
            - 1.579 * chiral
            + 0.073 * branches
            + 0.646 * ha_norm;
        let fine = if score_sas >= 0.50 { 0 } else { 1 };
        let detune_sas = (score_sas - 0.50).abs().min(1.0) * 0.15;

        (coarse, fine, detune_qed, detune_sas)
    }

    /// Screens a single SMILES string through the continuous topological tree valve.
    pub fn screen_smiles(
        &self,
        smiles: &str,
        true_qed: f64,
        true_sas: f64,
        noise: Option<&NoiseChannelParams>,
    ) -> Result<MoleculeScreeningResult, SmilesError> {
        let t_start = std::time::Instant::now();
        let feat = SmilesFeaturizer::featurize(smiles)?;
        let norm_feats = feat.to_normalized_vector();

        let (pred_c, pred_f, detune_qed, detune_sas) = Self::map_chemical_potentials(&norm_feats);
        let true_quadrant = PharmacologicalQuadrant::from_qed_sas(true_qed, true_sas);

        let dim = self.valve_cfg.symmetric_dim();
        let mut ham = CascadedValveHamiltonian::new(self.valve_cfg, self.config.gamma);
        let total_detuning = (self.config.detuning_scale + detune_qed + detune_sas).clamp(0.02, 0.40);
        ham.configure_target_route(pred_c, pred_f, total_detuning);
        let mat = ham.build_symmetric_matrix();

        // Inject single-excitation wavepacket into Entrance Root |in_0>
        let mut state = vec![Complex64::new(0.0, 0.0); dim];
        state[self.valve_cfg.t1_in_idx(0)] = Complex64::new(1.0, 0.0);
        let mut next = vec![Complex64::new(0.0, 0.0); dim];

        let steps = (self.config.t_transit / self.config.dt).round() as usize;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0x5027_0003);
        let mid_root = self.valve_cfg.mid_root_idx(pred_c);
        let zeno_step = steps / 2;

        let mut peak_exit_probs = [0.0f64; 4];

        for step in 0..steps {
            let t = step as f64 * self.config.dt;
            let amp = self.pulse_amplitude(t);
            let dt_half = self.config.dt * 0.5 * amp;

            // Split-step Trotter pass 1
            ham.step_state(&mat, &state, &mut next, dt_half);
            state.copy_from_slice(&next);

            // Lindbladian open-system dephasing
            if let Some(n) = noise {
                let p_depol = (n.depolarizing_prob * (self.config.dt / 0.20)).clamp(0.0, 1.0);
                if p_depol > 0.0 && rand::Rng::gen_bool(&mut rng, p_depol) {
                    let col = rand::Rng::gen_range(&mut rng, 0..dim);
                    state[col] *= Complex64::new(-1.0, 0.0);
                }
            }

            // Intermediate Weld Zeno sentry projection at tree interface
            if self.config.zeno_stabilization && step == zeno_step {
                let mid_amp = state[mid_root];
                if mid_amp.norm_sqr() > 0.01 {
                    let phase = mid_amp.arg();
                    state[mid_root] *= Complex64::from_polar(1.0, -phase);
                }
            }

            // Split-step Trotter pass 2
            ham.step_state(&mat, &state, &mut next, dt_half);
            state.copy_from_slice(&next);

            // Sample detector roots across transit window
            for y in 0..2 {
                for z in 0..2 {
                    let idx = self.valve_cfg.final_exit_root_idx(y, z);
                    let p = state[idx].norm_sqr();
                    let q_idx = y * 2 + z;
                    if p > peak_exit_probs[q_idx] {
                        peak_exit_probs[q_idx] = p;
                    }
                }
            }
        }

        let mut best_q = 0;
        let mut best_p = -1.0;
        let mut runner_up_p = 0.0;
        for (q, &p) in peak_exit_probs.iter().enumerate() {
            if p > best_p {
                runner_up_p = best_p.max(0.0);
                best_p = p;
                best_q = q;
            } else if p > runner_up_p {
                runner_up_p = p;
            }
        }

        let predicted_quadrant = PharmacologicalQuadrant::from_coords(best_q / 2, best_q % 2);
        let routing_contrast = best_p / (runner_up_p.max(1e-6));
        let is_correct = predicted_quadrant == true_quadrant;
        let is_lead_match = predicted_quadrant == PharmacologicalQuadrant::LeadCandidate
            && true_quadrant == PharmacologicalQuadrant::LeadCandidate;
        let transit_latency_us = t_start.elapsed().as_micros() as f64;

        Ok(MoleculeScreeningResult {
            smiles: smiles.to_string(),
            true_qed,
            true_sas,
            true_quadrant,
            predicted_quadrant,
            exit_probabilities: peak_exit_probs,
            routing_contrast,
            is_correct,
            is_lead_match,
            transit_latency_us,
        })
    }

    /// Evaluates a batch of molecules and returns an aggregate screening report.
    pub fn evaluate_batch(
        &self,
        molecules: &[(String, f64, f64)],
        noise: Option<&NoiseChannelParams>,
    ) -> (BatchMolecularScreeningReport, Vec<MoleculeScreeningResult>) {
        let mut results = Vec::with_capacity(molecules.len());
        let mut correct_quadrant = 0usize;
        let mut correct_coarse = 0usize;
        let mut correct_fine = 0usize;
        let mut true_leads = 0usize;
        let mut pred_leads = 0usize;
        let mut tp_leads = 0usize;
        let mut total_contrast = 0.0;
        let mut total_latency = 0.0;

        for (smi, qed, sas) in molecules {
            if let Ok(res) = self.screen_smiles(smi, *qed, *sas, noise) {
                let (true_c, true_f) = res.true_quadrant.to_coords();
                let (pred_c, pred_f) = res.predicted_quadrant.to_coords();

                if res.is_correct {
                    correct_quadrant += 1;
                }
                if true_c == pred_c {
                    correct_coarse += 1;
                }
                if true_f == pred_f {
                    correct_fine += 1;
                }
                if res.true_quadrant == PharmacologicalQuadrant::LeadCandidate {
                    true_leads += 1;
                }
                if res.predicted_quadrant == PharmacologicalQuadrant::LeadCandidate {
                    pred_leads += 1;
                }
                if res.is_lead_match {
                    tp_leads += 1;
                }
                total_contrast += res.routing_contrast;
                total_latency += res.transit_latency_us;
                results.push(res);
            }
        }

        let n = results.len().max(1);
        let lead_precision = if pred_leads > 0 {
            (tp_leads as f64 / pred_leads as f64) * 100.0
        } else {
            0.0
        };
        let lead_recall = if true_leads > 0 {
            (tp_leads as f64 / true_leads as f64) * 100.0
        } else {
            0.0
        };

        let report = BatchMolecularScreeningReport {
            total_molecules: results.len(),
            lead_candidates_ground_truth: true_leads,
            lead_candidates_predicted: pred_leads,
            true_positive_leads: tp_leads,
            lead_precision_pct: lead_precision,
            lead_recall_pct: lead_recall,
            overall_quadrant_accuracy_pct: (correct_quadrant as f64 / n as f64) * 100.0,
            coarse_qed_accuracy_pct: (correct_coarse as f64 / n as f64) * 100.0,
            fine_sas_accuracy_pct: (correct_fine as f64 / n as f64) * 100.0,
            mean_routing_contrast: total_contrast / n as f64,
            mean_latency_us: total_latency / n as f64,
        };

        (report, results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pharmacological_quadrant_invariants() {
        assert_eq!(
            PharmacologicalQuadrant::from_qed_sas(0.85, 2.1),
            PharmacologicalQuadrant::LeadCandidate
        );
        assert_eq!(
            PharmacologicalQuadrant::from_qed_sas(0.75, 4.8),
            PharmacologicalQuadrant::ChallengingSynthesis
        );
        assert_eq!(
            PharmacologicalQuadrant::from_qed_sas(0.40, 2.3),
            PharmacologicalQuadrant::ScaffoldBuildingBlock
        );
        assert_eq!(
            PharmacologicalQuadrant::from_qed_sas(0.30, 6.2),
            PharmacologicalQuadrant::NonViable
        );
    }

    #[test]
    fn test_molecular_walk_screener_execution() {
        let screener = ContinuousMolecularWalkScreener::new(MolecularWalkScreenerConfig::default());
        let smi = "CC(=O)Oc1ccccc1C(=O)O"; // Aspirin
        let res = screener.screen_smiles(smi, 0.72, 1.8, None).expect("Screen aspirin");
        assert!(res.routing_contrast > 1.0);
        assert!(res.exit_probabilities.iter().sum::<f64>() > 0.0);
    }
}
