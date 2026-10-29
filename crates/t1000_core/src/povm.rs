// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use num_complex::Complex64;

/// Generalized measurement without full collapse.
#[derive(Debug, Clone)]
pub struct POVM {
    pub operators: Vec<Vec<Complex64>>,
}

impl POVM {
    pub fn new(operators: Vec<Vec<Complex64>>) -> Self {
        Self { operators }
    }

    /// Soft measurement outcome probabilities, loosely `Tr(E_k rho)`.
    pub fn measure_soft(&self, reg: &crate::QubitRegister) -> Vec<f64> {
        let mut out = vec![0.0; self.operators.len()];
        self.measure_soft_into(&reg.amplitudes, &mut out);
        out
    }

    /// Zero-allocation in-place soft measurement into a pre-allocated probabilities slice.
    pub fn measure_soft_into(&self, amplitudes: &[Complex64], out_probs: &mut [f64]) {
        assert_eq!(self.operators.len(), out_probs.len());
        let n = amplitudes.len() as f64;
        for (op, prob) in self.operators.iter().zip(out_probs.iter_mut()) {
            let mut proj = Complex64::default();
            for (i, amp) in amplitudes.iter().enumerate() {
                if let Some(o) = op.get(i) {
                    proj += o.conj() * amp;
                }
            }
            *prob = proj.norm_sqr() / n;
        }
    }

    /// Zero-allocation in-place soft measurement directly from SoA real and imaginary buffers.
    pub fn measure_soft_soa_into(&self, real: &[f64], imag: &[f64], out_probs: &mut [f64]) {
        assert_eq!(real.len(), imag.len());
        assert_eq!(self.operators.len(), out_probs.len());
        let n = real.len() as f64;
        for (op, prob) in self.operators.iter().zip(out_probs.iter_mut()) {
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for (i, &o) in op.iter().enumerate().take(real.len()) {
                let r = real[i];
                let m = imag[i];
                // o.conj() * amp = (o.re - i o.im) * (r + i m)
                // = (o.re * r + o.im * m) + i (o.re * m - o.im * r)
                sum_re += o.re * r + o.im * m;
                sum_im += o.re * m - o.im * r;
            }
            *prob = (sum_re * sum_re + sum_im * sum_im) / n;
        }
    }
}

pub fn povm_measure_soft(povm: &POVM, reg: &crate::QubitRegister) -> Vec<f64> {
    povm.measure_soft(reg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_povm_soa_equivalence() {
        let op1 = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        let op2 = vec![Complex64::new(0.0, -1.0), Complex64::new(1.0, 0.0)];
        let povm = POVM::new(vec![op1, op2]);

        let mut reg = crate::QubitRegister::new(2, 42);
        reg.amplitudes = vec![Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0)];

        let real = vec![0.6, 0.8];
        let imag = vec![0.0, 0.0];

        let probs_ref = povm.measure_soft(&reg);
        let mut probs_soa = vec![0.0; 2];
        povm.measure_soft_soa_into(&real, &imag, &mut probs_soa);

        for (r, s) in probs_ref.iter().zip(&probs_soa) {
            assert!((r - s).abs() < 1e-14);
        }
    }
}
