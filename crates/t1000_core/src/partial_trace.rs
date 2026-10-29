// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use num_complex::Complex64;

#[derive(Debug, Clone, Copy)]
pub struct PartialTrace;

impl PartialTrace {
    /// Computes the reduced density matrix rho_A by tracing out subsystem B.
    pub fn trace_out_b(tensor_state: &[Complex64], dim_a: usize, dim_b: usize) -> Vec<Complex64> {
        let mut rho_a = vec![Complex64::default(); dim_a * dim_a];
        Self::trace_out_b_into(tensor_state, dim_a, dim_b, &mut rho_a);
        rho_a
    }

    /// Zero-allocation partial trace into a pre-allocated output density matrix buffer.
    pub fn trace_out_b_into(
        tensor_state: &[Complex64],
        dim_a: usize,
        dim_b: usize,
        out_rho_a: &mut [Complex64],
    ) {
        assert_eq!(tensor_state.len(), dim_a * dim_b);
        assert_eq!(out_rho_a.len(), dim_a * dim_a);
        for i in 0..dim_a {
            let row_offset = i * dim_a;
            let i_base = i * dim_b;
            for j in 0..dim_a {
                let j_base = j * dim_b;
                let mut sum = Complex64::default();
                for k in 0..dim_b {
                    sum += tensor_state[i_base + k] * tensor_state[j_base + k].conj();
                }
                out_rho_a[row_offset + j] = sum;
            }
        }
    }

    /// Zero-allocation in-place partial trace directly from Structure-of-Arrays (SoA) amplitude buffers.
    ///
    /// Eliminates intermediate interleaving allocation: operates directly on separated real and imaginary buffers.
    pub fn trace_out_b_soa_into(
        real: &[f64],
        imag: &[f64],
        dim_a: usize,
        dim_b: usize,
        out_rho_a: &mut [Complex64],
    ) {
        assert_eq!(real.len(), dim_a * dim_b);
        assert_eq!(imag.len(), dim_a * dim_b);
        assert_eq!(out_rho_a.len(), dim_a * dim_a);

        for i in 0..dim_a {
            let row_offset = i * dim_a;
            let i_base = i * dim_b;
            for j in 0..dim_a {
                let j_base = j * dim_b;
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;
                for k in 0..dim_b {
                    let r_i = real[i_base + k];
                    let m_i = imag[i_base + k];
                    let r_j = real[j_base + k];
                    let m_j = imag[j_base + k];
                    // (r_i + i m_i) * (r_j - i m_j) = (r_i*r_j + m_i*m_j) + i(m_i*r_j - r_i*m_j)
                    sum_re += r_i * r_j + m_i * m_j;
                    sum_im += m_i * r_j - r_i * m_j;
                }
                out_rho_a[row_offset + j] = Complex64::new(sum_re, sum_im);
            }
        }
    }

    pub fn partial_trace_b(
        tensor_state: &[Complex64],
        dim_a: usize,
        dim_b: usize,
    ) -> Vec<Complex64> {
        Self::trace_out_b(tensor_state, dim_a, dim_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_trace_soa_equivalence() {
        let dim_a = 2;
        let dim_b = 3;
        let dim = dim_a * dim_b;

        let real: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.7).sin()).collect();
        let imag: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.4).cos()).collect();
        let interleaved: Vec<Complex64> = real
            .iter()
            .zip(&imag)
            .map(|(&r, &i)| Complex64::new(r, i))
            .collect();

        let rho_ref = PartialTrace::trace_out_b(&interleaved, dim_a, dim_b);
        let mut rho_soa = vec![Complex64::default(); dim_a * dim_a];
        PartialTrace::trace_out_b_soa_into(&real, &imag, dim_a, dim_b, &mut rho_soa);

        for (r, s) in rho_ref.iter().zip(&rho_soa) {
            assert!((r - s).norm() < 1e-14);
        }
    }
}
