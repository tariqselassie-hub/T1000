//! Fused diagonal layer: all Rz rotations and CZ gates of a layer in one statevector pass.
//!
//! Rz and CZ are diagonal in the computational basis, so a layer's Rz(theta_q) on every qubit
//! followed by any set of CZ pairs multiplies amplitude `x` by
//!
//! ```text
//! d(x) = exp(i * sum_q s_q(x) * theta_q / 2) * (-1)^(number of CZ pairs with both bits set in x)
//! ```
//!
//! with `s_q(x) = -1` if bit `q` of `x` is 0 and `+1` if it is 1. Applying `d` directly costs one
//! pass instead of one pass per gate.
//!
//! Implementation: the index is split into a low part `j` (the lowest `LO_BITS` qubits) and a
//! high part `h`. A table over `j` holds the low-qubit phase and the sign from CZ pairs inside the
//! low qubits. For each block of fixed `h` the high-qubit phase, the sign from CZ pairs inside the
//! high qubits, and a mask of low qubits coupled by CZ to a set high bit are computed once; the
//! block's multiplier row is then the low table times that factor with the cross-pair parity
//! applied, and amplitudes are multiplied by the row.

const LO_BITS: usize = 8;

/// Pre-computed tables shared by every block.
struct DiagonalPlan {
    n_qubits: usize,
    lo_bits: usize,
    /// Low-qubit phase times low-only CZ sign, split into (re, im).
    lo_re: Vec<f64>,
    lo_im: Vec<f64>,
    /// Half-angles for the high qubits (index `q - lo_bits`).
    hi_half_angles: Vec<f64>,
    /// CZ pairs with both qubits in the high part, as high-index masks.
    hi_pair_masks: Vec<usize>,
    /// CZ pairs with one low and one high qubit: (high-index bit mask, low-index bit mask).
    cross_pairs: Vec<(usize, usize)>,
}

impl DiagonalPlan {
    fn new(n_qubits: usize, theta_rz: &[f64], cz_pairs: &[(usize, usize)]) -> Self {
        let lo_bits = n_qubits.min(LO_BITS);
        let lo_size = 1usize << lo_bits;
        let half = |q: usize| theta_rz.get(q).copied().unwrap_or(0.0) * 0.5;

        let mut lo_pair_masks = Vec::new();
        let mut hi_pair_masks = Vec::new();
        let mut cross_pairs = Vec::new();
        for &(a, b) in cz_pairs {
            let (a, b) = (a.min(b), a.max(b));
            if b < lo_bits {
                lo_pair_masks.push((1usize << a) | (1usize << b));
            } else if a >= lo_bits {
                hi_pair_masks.push((1usize << (a - lo_bits)) | (1usize << (b - lo_bits)));
            } else {
                cross_pairs.push((1usize << (b - lo_bits), 1usize << a));
            }
        }

        let mut lo_re = vec![0.0; lo_size];
        let mut lo_im = vec![0.0; lo_size];
        for j in 0..lo_size {
            let angle: f64 = (0..lo_bits)
                .map(|q| if (j >> q) & 1 == 1 { half(q) } else { -half(q) })
                .sum();
            let negative = lo_pair_masks.iter().filter(|&&m| j & m == m).count() % 2 == 1;
            let sign = if negative { -1.0 } else { 1.0 };
            let (s, c) = angle.sin_cos();
            lo_re[j] = sign * c;
            lo_im[j] = sign * s;
        }

        Self {
            n_qubits,
            lo_bits,
            lo_re,
            lo_im,
            hi_half_angles: (lo_bits..n_qubits).map(half).collect(),
            hi_pair_masks,
            cross_pairs,
        }
    }

    /// Multiplies the block of amplitudes with high index `h` (length `2^lo_bits`) by `d(x)`.
    fn apply_block(&self, h: usize, re: &mut [f64], im: &mut [f64]) {
        let angle: f64 = self
            .hi_half_angles
            .iter()
            .enumerate()
            .map(|(k, &a)| if (h >> k) & 1 == 1 { a } else { -a })
            .sum();
        let hi_negative = self.hi_pair_masks.iter().filter(|&&m| h & m == m).count() % 2 == 1;
        let sign = if hi_negative { -1.0 } else { 1.0 };
        let (s, c) = angle.sin_cos();
        let (fr, fi) = (sign * c, sign * s);
        let cross_mask = self
            .cross_pairs
            .iter()
            .filter(|&&(hm, _)| h & hm != 0)
            .fold(0usize, |acc, &(_, lm)| acc ^ lm);

        for j in 0..re.len() {
            let flip = if (j & cross_mask).count_ones() & 1 == 1 { -1.0 } else { 1.0 };
            let (tr, ti) = (self.lo_re[j], self.lo_im[j]);
            let (dr, di) = (flip * (tr * fr - ti * fi), flip * (tr * fi + ti * fr));
            let (ar, ai) = (re[j], im[j]);
            re[j] = ar * dr - ai * di;
            im[j] = ar * di + ai * dr;
        }
    }
}

/// Applies `Rz(theta_rz[q])` to every qubit `q < theta_rz.len()` and then CZ on every pair in
/// `cz_pairs`, in a single pass over the statevector. Equivalent to applying the gates one by one.
pub fn apply_rz_cz_layer(real: &mut [f64], imag: &mut [f64], theta_rz: &[f64], cz_pairs: &[(usize, usize)]) {
    let n_qubits = crate::simd::assert_statevector(real, imag);
    assert!(
        theta_rz.len() <= n_qubits,
        "{} Rz angles for a {}-qubit state",
        theta_rz.len(),
        n_qubits
    );
    for &(a, b) in cz_pairs {
        crate::simd::assert_qubit_pair(a, b, n_qubits);
    }
    let plan = DiagonalPlan::new(n_qubits, theta_rz, cz_pairs);
    let block = 1usize << plan.lo_bits;
    debug_assert_eq!(plan.n_qubits, n_qubits);
    crate::parallel::for_each_block(real, imag, block, |h, re, im| plan.apply_block(h, re, im));
}
