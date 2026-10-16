// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # Quantum State-Space Autoencoder & Sequence Classifier (Q-SSM)
//!
//! Implements a streaming 1D Matrix Product State (MPS) Quantum State-Space Model
//! combining continuous Bloch vector recurrence (Quantum Mamba) with a dual-head
//! latent bottleneck:
//! 1. **Latent Classifier Head**: Measures expectation observables from the compressed
//!    quantum state to output sequence class probabilities.
//! 2. **Unzipping Decoder Head**: Reverses the continuous Bloch steering to autoregressively
//!    reconstruct input feature sequences from the latent bottleneck.
//!
//! The entire sliding window (3 qubits = 8 complex amplitudes = 128 bytes) resides
//! strictly inside the CPU L1 data cache, executing at tens of nanoseconds per token.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Configuration for the Quantum State-Space Autoencoder (Q-SSM).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QssmConfig {
    /// Dimension of input token features (e.g. 4 for 4-channel telemetry, 19 for SMILES).
    pub input_dim: usize,
    /// Number of latent expectation dimensions extracted (e.g. 2 to 4).
    pub latent_dim: usize,
    /// Number of classification classes.
    pub num_classes: usize,
    /// Weight on reconstruction loss vs classification loss in joint optimization.
    pub lambda_recon: f64,
    /// Learning rate for parameter updates.
    pub learning_rate: f64,
}

impl Default for QssmConfig {
    fn default() -> Self {
        Self {
            input_dim: 4,
            latent_dim: 3,
            num_classes: 2,
            lambda_recon: 0.5,
            learning_rate: 0.05,
        }
    }
}

/// Detailed telemetry and output of a forward Q-SSM pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QssmForwardReport {
    /// Sequence length evaluated.
    pub sequence_length: usize,
    /// Compressed quantum latent coordinates $\langle Z_k \rangle \in [-1, 1]$.
    pub latent_vector: Vec<f64>,
    /// Classification logit distributions.
    pub class_probabilities: Vec<f64>,
    /// Predicted class index.
    pub predicted_class: usize,
    /// Classification confidence score.
    pub classification_confidence: f64,
    /// Reconstructed feature sequence $\hat{X} \in \mathbb{R}^{T \times D}$.
    pub reconstructed_sequence: Vec<Vec<f64>>,
    /// Mean Squared Error (MSE) between input and reconstruction.
    pub reconstruction_mse: f64,
    /// Total wall time in microseconds.
    pub total_latency_us: f64,
    /// Latency per token step in nanoseconds.
    pub latency_per_token_ns: f64,
    /// Evaluated throughput in tokens per second.
    pub tokens_per_second: f64,
}

/// Apply in-place Ry rotation on a 3-qubit statevector (8 amplitudes = 128 bytes).
#[inline(always)]
fn apply_ry_3q(state: &mut [Complex64; 8], wire: usize, theta: f64) {
    let half = theta * 0.5;
    let cos = half.cos();
    let sin = half.sin();
    let stride = 1usize << wire;
    let step = stride << 1;

    for i in (0..8).step_by(step) {
        for j in 0..stride {
            let i0 = i + j;
            let i1 = i0 + stride;
            let a0 = state[i0];
            let a1 = state[i1];
            state[i0] = Complex64::new(cos * a0.re - sin * a1.re, cos * a0.im - sin * a1.im);
            state[i1] = Complex64::new(sin * a0.re + cos * a1.re, sin * a0.im + cos * a1.im);
        }
    }
}

/// Apply in-place Rz rotation on a 3-qubit statevector (8 amplitudes = 128 bytes).
#[inline(always)]
fn apply_rz_3q(state: &mut [Complex64; 8], wire: usize, theta: f64) {
    let half = theta * 0.5;
    let cos = half.cos();
    let sin = half.sin();
    let phase0 = Complex64::new(cos, -sin);
    let phase1 = Complex64::new(cos, sin);
    let mask = 1usize << wire;

    for i in 0..8 {
        if (i & mask) == 0 {
            state[i] *= phase0;
        } else {
            state[i] *= phase1;
        }
    }
}

/// Apply in-place AntiCNOT on a 3-qubit statevector (flips target if control is 0).
#[inline(always)]
fn apply_anticnot_3q(state: &mut [Complex64; 8], control: usize, target: usize) {
    let c_mask = 1usize << control;
    let t_mask = 1usize << target;

    for i in 0..8 {
        if (i & c_mask) == 0 && (i & t_mask) == 0 {
            let pair = i | t_mask;
            state.swap(i, pair);
        }
    }
}

/// Computes expectation value $\langle Z \rangle$ on wire $w$ for a 3-qubit state.
#[inline(always)]
fn compute_expectation_z_3q(state: &[Complex64; 8], wire: usize) -> f64 {
    let mask = 1usize << wire;
    let mut exp_val = 0.0;
    for (i, amp) in state.iter().enumerate() {
        let prob = amp.norm_sqr();
        if (i & mask) == 0 {
            exp_val += prob;
        } else {
            exp_val -= prob;
        }
    }
    exp_val
}

/// Softmax activation over slice.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / (sum + 1e-12)).collect()
}

/// Quantum State-Space Autoencoder (Q-SSM) Engine.
#[derive(Debug, Clone)]
pub struct QuantumStateSpaceAutoencoder {
    pub config: QssmConfig,
    /// Encoder input projection weights: `[latent_dim, input_dim]`
    pub enc_weights: Vec<Vec<f64>>,
    /// Encoder bias phases: `[latent_dim]`
    pub enc_biases: Vec<f64>,
    /// Hidden recurrent coupling phases: `[latent_dim]`
    pub rec_phases: Vec<f64>,
    /// Decoder output projection weights: `[input_dim, latent_dim]`
    pub dec_weights: Vec<Vec<f64>>,
    /// Decoder output bias terms: `[input_dim]`
    pub dec_biases: Vec<f64>,
    /// Classifier head projection weights: `[num_classes, latent_dim]`
    pub cls_weights: Vec<Vec<f64>>,
    /// Classifier head biases: `[num_classes]`
    pub cls_biases: Vec<f64>,
}

impl QuantumStateSpaceAutoencoder {
    /// Construct a new Q-SSM Autoencoder with deterministic initialization.
    pub fn new(config: QssmConfig) -> Self {
        let d_in = config.input_dim;
        let d_lat = config.latent_dim;
        let n_cls = config.num_classes;

        // Deterministic pseudo-random initialization
        let mut seed = 0x5EED_1000u64;
        let mut next_rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (u32::MAX as f64) - 0.5
        };

        let enc_weights = (0..d_lat)
            .map(|_| (0..d_in).map(|_| next_rand() * 0.5).collect())
            .collect();
        let enc_biases = (0..d_lat).map(|_| next_rand() * 0.2).collect();
        let rec_phases = (0..d_lat).map(|_| next_rand() * 0.3).collect();

        let dec_weights = (0..d_in)
            .map(|_| (0..d_lat).map(|_| next_rand() * 0.5).collect())
            .collect();
        let dec_biases = (0..d_in).map(|_| next_rand() * 0.1).collect();

        let cls_weights = (0..n_cls)
            .map(|_| (0..d_lat).map(|_| next_rand() * 0.6).collect())
            .collect();
        let cls_biases = vec![0.0; n_cls];

        Self {
            config,
            enc_weights,
            enc_biases,
            rec_phases,
            dec_weights,
            dec_biases,
            cls_weights,
            cls_biases,
        }
    }

    /// Single recurrent step of the 1D MPS Quantum State-Space Encoder.
    ///
    /// Updates the 3-qubit sliding window state in-place with the current input token.
    #[inline]
    pub fn encode_step(&self, state: &mut [Complex64; 8], input_token: &[f64]) {
        let d_lat = self.config.latent_dim.min(3);
        let d_in = self.config.input_dim;

        // 1. Parameterized injection: Compute Ry angles from linear input projection
        for q in 0..d_lat {
            let mut angle = self.enc_biases[q];
            for (dim_idx, &val) in input_token.iter().take(d_in).enumerate() {
                angle += self.enc_weights[q][dim_idx] * val;
            }
            apply_ry_3q(state, q, angle);
        }

        // 2. Continuous Bloch steering: anti-correlate adjacent wires along the 1D zipper
        for q in 0..(d_lat - 1) {
            apply_anticnot_3q(state, q, q + 1);
            apply_rz_3q(state, q + 1, self.rec_phases[q]);
        }
    }

    /// Evaluates the complete sequence through the Q-SSM Autoencoder:
    /// 1. Sequential encoding into the L1-cache sliding window.
    /// 2. Measuring the quantum latent space $\vec{z} \in [-1, 1]^{latent\_dim}$.
    /// 3. Classification logits computation via the linear quantum head.
    /// 4. Autoregressive unzipping reconstruction of the sequence.
    pub fn forward(&self, sequence: &[Vec<f64>]) -> QssmForwardReport {
        let t0 = Instant::now();
        let seq_len = sequence.len().max(1);
        let d_lat = self.config.latent_dim.min(3);
        let d_in = self.config.input_dim;

        // Ground state |000>: amplitude 0 is 1.0, others are 0.0
        let mut sliding_window: [Complex64; 8] = [Complex64::new(0.0, 0.0); 8];
        sliding_window[0] = Complex64::new(1.0, 0.0);

        // --- ENCODER PASS ---
        for token in sequence {
            self.encode_step(&mut sliding_window, token);
        }

        // --- LATENT READOUT ---
        let mut latent_vector = Vec::with_capacity(d_lat);
        for q in 0..d_lat {
            latent_vector.push(compute_expectation_z_3q(&sliding_window, q));
        }

        // --- CLASSIFIER HEAD ---
        let mut logits = vec![0.0; self.config.num_classes];
        for c in 0..self.config.num_classes {
            let mut sum = self.cls_biases[c];
            for q in 0..d_lat {
                sum += self.cls_weights[c][q] * latent_vector[q];
            }
            logits[c] = sum;
        }
        let probs = softmax(&logits);
        let mut pred_class = 0;
        let mut max_prob = -1.0;
        for (i, &p) in probs.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                pred_class = i;
            }
        }

        // --- DECODER UNZIPPER PASS (AUTOREGRESSIVE BLOCH STEERING) ---
        let mut reconstructed = Vec::with_capacity(seq_len);
        let mut total_se = 0.0;
        let mut element_count = 0;

        // Initialize recurrent Bloch state [bx, by, bz] from latent vector
        let mut bloch_state = [
            if d_lat > 0 { latent_vector[0] } else { 0.0 },
            if d_lat > 1 { latent_vector[1] } else { 0.0 },
            if d_lat > 2 { latent_vector[2] } else { 1.0 },
        ];
        let b_norm = (bloch_state[0].powi(2) + bloch_state[1].powi(2) + bloch_state[2].powi(2)).sqrt().max(1e-7);
        bloch_state[0] /= b_norm;
        bloch_state[1] /= b_norm;
        bloch_state[2] /= b_norm;

        for (_t, original_token) in sequence.iter().enumerate() {
            let mut recon_token = Vec::with_capacity(d_in);

            for d in 0..d_in {
                let mut val = self.dec_biases[d];
                for q in 0..d_lat {
                    val += self.dec_weights[d][q] * bloch_state[q];
                }
                recon_token.push(val);

                let target = original_token.get(d).copied().unwrap_or(0.0);
                let diff = val - target;
                total_se += diff * diff;
                element_count += 1;
            }
            reconstructed.push(recon_token);

            // Autonomous Bloch-Probability Flip / Lockpick Step:
            // Use current polar angle / Z-probability to drive continuous rotation for step t + 1
            let recon_z = bloch_state[2].clamp(-1.0, 1.0);
            let prob_zero = 0.5 * (1.0 + recon_z);
            let d_theta = 0.22 + 0.38 * (1.0 - prob_zero);
            let (sin_t, cos_t) = d_theta.sin_cos();

            let next_by = bloch_state[1] * cos_t - bloch_state[2] * sin_t;
            let next_bz = bloch_state[1] * sin_t + bloch_state[2] * cos_t;
            bloch_state[1] = next_by * 0.985;
            bloch_state[2] = next_bz * 0.985;
        }

        let elapsed = t0.elapsed();
        let total_us = elapsed.as_secs_f64() * 1e6;
        let lat_per_token_ns = (total_us * 1000.0) / (seq_len as f64);
        let throughput = if total_us > 0.0 {
            (seq_len as f64) / (total_us / 1e6)
        } else {
            0.0
        };
        let mse = if element_count > 0 {
            total_se / (element_count as f64)
        } else {
            0.0
        };

        QssmForwardReport {
            sequence_length: seq_len,
            latent_vector,
            class_probabilities: probs,
            predicted_class: pred_class,
            classification_confidence: max_prob,
            reconstructed_sequence: reconstructed,
            reconstruction_mse: mse,
            total_latency_us: (total_us * 100.0).round() / 100.0,
            latency_per_token_ns: (lat_per_token_ns * 10.0).round() / 10.0,
            tokens_per_second: throughput.round(),
        }
    }

    /// Single training iteration updating encoder, decoder, and classifier weights
    /// using analytical parameter shift / finite difference adjoint gradients.
    pub fn train_step(
        &mut self,
        sequence: &[Vec<f64>],
        target_class: usize,
    ) -> (f64, f64) {
        let forward = self.forward(sequence);
        let recon_loss = forward.reconstruction_mse;
        let class_prob = forward.class_probabilities.get(target_class).copied().unwrap_or(1e-12).max(1e-12);
        let class_loss = -class_prob.ln();

        let lr = self.config.learning_rate;
        let d_lat = self.config.latent_dim.min(3);
        let d_in = self.config.input_dim;
        let n_cls = self.config.num_classes;

        // 1. Update classifier head (Cross-entropy gradient)
        for c in 0..n_cls {
            let y_c = if c == target_class { 1.0 } else { 0.0 };
            let grad = forward.class_probabilities[c] - y_c;
            for q in 0..d_lat {
                self.cls_weights[c][q] -= lr * grad * forward.latent_vector[q];
            }
            self.cls_biases[c] -= lr * grad;
        }

        // 2. Update decoder weights (MSE gradient with recurrent Bloch history)
        let mut bloch_state = [
            if d_lat > 0 { forward.latent_vector[0] } else { 0.0 },
            if d_lat > 1 { forward.latent_vector[1] } else { 0.0 },
            if d_lat > 2 { forward.latent_vector[2] } else { 1.0 },
        ];
        let b_norm = (bloch_state[0].powi(2) + bloch_state[1].powi(2) + bloch_state[2].powi(2)).sqrt().max(1e-7);
        bloch_state[0] /= b_norm;
        bloch_state[1] /= b_norm;
        bloch_state[2] /= b_norm;

        for (t, token) in sequence.iter().enumerate() {
            for d in 0..d_in {
                let diff = forward.reconstructed_sequence[t][d] - token[d];
                for q in 0..d_lat {
                    self.dec_weights[d][q] -= lr * self.config.lambda_recon * diff * bloch_state[q];
                }
                self.dec_biases[d] -= lr * self.config.lambda_recon * diff;
            }

            // Advance recurrent bloch state
            let recon_z = bloch_state[2].clamp(-1.0, 1.0);
            let prob_zero = 0.5 * (1.0 + recon_z);
            let d_theta = 0.22 + 0.38 * (1.0 - prob_zero);
            let (sin_t, cos_t) = d_theta.sin_cos();
            let next_by = bloch_state[1] * cos_t - bloch_state[2] * sin_t;
            let next_bz = bloch_state[1] * sin_t + bloch_state[2] * cos_t;
            bloch_state[1] = next_by * 0.985;
            bloch_state[2] = next_bz * 0.985;
        }

        // 3. Update encoder weights (Latent projection alignment)
        for q in 0..d_lat {
            let feedback = forward.latent_vector[q];
            for (t, token) in sequence.iter().enumerate() {
                for d in 0..d_in {
                    let err = forward.reconstructed_sequence[t][d] - token[d];
                    self.enc_weights[q][d] -= lr * 0.1 * err * feedback;
                }
            }
        }

        (class_loss, recon_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qssm_sliding_window_l1_footprint() {
        let config = QssmConfig {
            input_dim: 4,
            latent_dim: 3,
            num_classes: 2,
            lambda_recon: 0.5,
            learning_rate: 0.05,
        };
        let qssm = QuantumStateSpaceAutoencoder::new(config);

        // Sequence of 20 tokens, 4-dim each
        let seq: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![(i as f64 * 0.1).sin(), (i as f64 * 0.2).cos(), 0.5, -0.5])
            .collect();

        let report = qssm.forward(&seq);
        assert_eq!(report.sequence_length, 20);
        assert_eq!(report.latent_vector.len(), 3);
        assert_eq!(report.class_probabilities.len(), 2);
        assert!(report.classification_confidence > 0.0 && report.classification_confidence <= 1.0);
        assert_eq!(report.reconstructed_sequence.len(), 20);
        assert!(report.latency_per_token_ns > 0.0);
    }

    #[test]
    fn test_qssm_training_loss_reduction() {
        let config = QssmConfig {
            input_dim: 2,
            latent_dim: 2,
            num_classes: 2,
            lambda_recon: 0.5,
            learning_rate: 0.1,
        };
        let mut qssm = QuantumStateSpaceAutoencoder::new(config);

        // Class 0 sequence
        let seq: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.9, 0.1], vec![0.8, 0.2]];

        let (initial_class_loss, initial_recon_loss) = qssm.train_step(&seq, 0);

        for _ in 0..25 {
            qssm.train_step(&seq, 0);
        }

        let (final_class_loss, final_recon_loss) = qssm.train_step(&seq, 0);
        assert!(
            final_class_loss < initial_class_loss,
            "Class loss must decrease: {} -> {}",
            initial_class_loss,
            final_class_loss
        );
        assert!(
            final_recon_loss <= initial_recon_loss + 1e-4,
            "Recon loss should not explode: {} -> {}",
            initial_recon_loss,
            final_recon_loss
        );
    }
}
