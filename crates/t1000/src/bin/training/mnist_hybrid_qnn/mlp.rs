// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Differentiable 2-Layer Non-Linear MLP Head with LeakyReLU Activation.

pub const N_CLASSES: usize = 10;
pub const N_OBSERVABLES: usize = 55; // 10 Z + 45 ZZ
pub const N_FUSED: usize = N_OBSERVABLES + super::features::N_SPATIAL; // 120
pub const N_HIDDEN: usize = 128;

#[derive(Debug, Clone)]
pub struct HybridMlpHead {
    pub w1: Vec<f64>, // N_HIDDEN x N_FUSED
    pub b1: Vec<f64>, // N_HIDDEN
    pub w2: Vec<f64>, // N_CLASSES x N_HIDDEN
    pub b2: Vec<f64>, // N_CLASSES
}

impl HybridMlpHead {
    pub fn new(seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let scale1 = (2.0 / N_FUSED as f64).sqrt();
        let w1 = (0..N_HIDDEN * N_FUSED)
            .map(|_| rng.gen_range(-scale1..scale1))
            .collect();
        let b1 = vec![0.01; N_HIDDEN];

        let scale2 = (2.0 / N_HIDDEN as f64).sqrt();
        let w2 = (0..N_CLASSES * N_HIDDEN)
            .map(|_| rng.gen_range(-scale2..scale2))
            .collect();
        let b2 = vec![0.0; N_CLASSES];

        Self { w1, b1, w2, b2 }
    }

    pub fn param_count(&self) -> usize {
        N_HIDDEN * N_FUSED + N_HIDDEN + N_CLASSES * N_HIDDEN + N_CLASSES
    }

    pub fn pack_params(&self) -> Vec<f64> {
        let mut p = Vec::with_capacity(self.param_count());
        p.extend_from_slice(&self.w1);
        p.extend_from_slice(&self.b1);
        p.extend_from_slice(&self.w2);
        p.extend_from_slice(&self.b2);
        p
    }

    pub fn unpack_params(&mut self, p: &[f64]) {
        let mut off = 0;
        let s_w1 = N_HIDDEN * N_FUSED;
        self.w1.copy_from_slice(&p[off..off + s_w1]);
        off += s_w1;
        self.b1.copy_from_slice(&p[off..off + N_HIDDEN]);
        off += N_HIDDEN;
        let s_w2 = N_CLASSES * N_HIDDEN;
        self.w2.copy_from_slice(&p[off..off + s_w2]);
        off += s_w2;
        self.b2.copy_from_slice(&p[off..off + N_CLASSES]);
    }

    pub fn forward(&self, x: &[f64; N_FUSED]) -> ([f64; N_CLASSES], [f64; N_HIDDEN], [f64; N_HIDDEN]) {
        let mut z1 = [0.0; N_HIDDEN];
        let mut a1 = [0.0; N_HIDDEN];
        for h in 0..N_HIDDEN {
            let mut sum = self.b1[h];
            let row = h * N_FUSED;
            for (i, &xi) in x.iter().enumerate().take(N_FUSED) {
                sum += self.w1[row + i] * xi;
            }
            z1[h] = sum;
            a1[h] = if sum > 0.0 { sum } else { 0.05 * sum };
        }

        let mut logits = [0.0; N_CLASSES];
        for (c, logit) in logits.iter_mut().enumerate().take(N_CLASSES) {
            let mut sum = self.b2[c];
            let row = c * N_HIDDEN;
            for (h, &ah) in a1.iter().enumerate().take(N_HIDDEN) {
                sum += self.w2[row + h] * ah;
            }
            *logit = sum;
        }
        (logits, z1, a1)
    }

    pub fn backward(
        &self,
        x: &[f64; N_FUSED],
        z1: &[f64; N_HIDDEN],
        a1: &[f64; N_HIDDEN],
        d_logits: &[f64; N_CLASSES],
        grad_out: &mut [f64],
    ) -> [f64; N_FUSED] {
        let off_w1 = 0;
        let off_b1 = N_HIDDEN * N_FUSED;
        let off_w2 = off_b1 + N_HIDDEN;
        let off_b2 = off_w2 + N_CLASSES * N_HIDDEN;

        for c in 0..N_CLASSES {
            let row = c * N_HIDDEN;
            let d_c = d_logits[c];
            grad_out[off_b2 + c] += d_c;
            for (h, &ah) in a1.iter().enumerate().take(N_HIDDEN) {
                grad_out[off_w2 + row + h] += d_c * ah;
            }
        }

        let mut d_a1 = [0.0; N_HIDDEN];
        for (h, da) in d_a1.iter_mut().enumerate().take(N_HIDDEN) {
            let mut sum = 0.0;
            for (c, &dc) in d_logits.iter().enumerate().take(N_CLASSES) {
                sum += dc * self.w2[c * N_HIDDEN + h];
            }
            *da = sum;
        }

        let mut d_z1 = [0.0; N_HIDDEN];
        for (h, dz) in d_z1.iter_mut().enumerate().take(N_HIDDEN) {
            *dz = if z1[h] > 0.0 { d_a1[h] } else { 0.05 * d_a1[h] };
            grad_out[off_b1 + h] += *dz;
        }

        let mut d_x = [0.0; N_FUSED];
        for (h, &d_zh) in d_z1.iter().enumerate().take(N_HIDDEN) {
            let row = h * N_FUSED;
            for (i, &xi) in x.iter().enumerate().take(N_FUSED) {
                grad_out[off_w1 + row + i] += d_zh * xi;
                d_x[i] += d_zh * self.w1[row + i];
            }
        }

        d_x
    }
}

pub fn softmax_cross_entropy(logits: &[f64; N_CLASSES], label: usize) -> (f64, [f64; N_CLASSES], usize) {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut sum_exp = 0.0;
    let mut probs = [0.0; 10];
    for c in 0..10 {
        let e = (logits[c] - max_logit).exp();
        probs[c] = e;
        sum_exp += e;
    }
    let inv_sum = 1.0 / sum_exp.max(1e-12);
    let mut pred_class = 0;
    let mut best_prob = -1.0;
    for (c, p) in probs.iter_mut().enumerate().take(10) {
        *p *= inv_sum;
        if *p > best_prob {
            best_prob = *p;
            pred_class = c;
        }
    }

    let loss = -(probs[label].max(1e-12)).ln();
    let mut d_logits = [0.0; 10];
    d_logits.copy_from_slice(&probs);
    d_logits[label] -= 1.0;

    (loss, d_logits, pred_class)
}
