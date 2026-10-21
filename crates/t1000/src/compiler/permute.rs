// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Run-based kernels for CNOT, AntiCNOT, SWAP and controlled-phase.
//!
//! With `lo < hi` the two qubits split an index into a prefix above `hi`, a middle between the
//! qubits and a suffix below `lo`. For a fixed prefix and middle, the amplitudes of each
//! (bit `lo`, bit `hi`) combination form one contiguous run of `2^lo` elements, so each gate is
//! a swap or phase of whole runs instead of a branch per index.

/// Visits `(base, run_len)` for every prefix/middle combination of the pair `(lo, hi)`.
#[inline(always)]
fn for_each_base(dim: usize, lo: usize, hi: usize, mut f: impl FnMut(usize)) {
    let middle = 1usize << (hi - lo - 1);
    for prefix in 0..(dim >> (hi + 1)) {
        let top = prefix << (hi + 1);
        for mid in 0..middle {
            f(top | (mid << (lo + 1)));
        }
    }
}

/// Swaps run `base + a` with run `base + b` in both planes for every base.
fn swap_runs(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize, pick: impl Fn(usize, usize) -> (usize, usize)) {
    let (lo, hi) = (q1.min(q2), q1.max(q2));
    let run = 1usize << lo;
    let (a_off, b_off) = pick(1 << lo, 1 << hi);
    let (rp, ip) = (real.as_mut_ptr(), imag.as_mut_ptr());
    for_each_base(real.len(), lo, hi, |base| {
        // SAFETY: both runs lie inside the statevector (base + offset + run <= dim), and the
        // two offsets differ in a bit >= lo, so the runs never overlap.
        unsafe {
            std::ptr::swap_nonoverlapping(rp.add(base + a_off), rp.add(base + b_off), run);
            std::ptr::swap_nonoverlapping(ip.add(base + a_off), ip.add(base + b_off), run);
        }
    });
}

/// CNOT: swap the target halves inside the control = 1 subspace.
pub fn cnot(real: &mut [f64], imag: &mut [f64], control: usize, target: usize) {
    let (cm, tm) = (1usize << control, 1usize << target);
    swap_runs(real, imag, control, target, |_, _| (cm, cm | tm));
}

/// AntiCNOT: swap the target halves inside the control = 0 subspace.
pub fn anticnot(real: &mut [f64], imag: &mut [f64], control: usize, target: usize) {
    let tm = 1usize << target;
    swap_runs(real, imag, control, target, |_, _| (0, tm));
}

/// SWAP: exchange the (q1 = 1, q2 = 0) and (q1 = 0, q2 = 1) runs.
pub fn swap(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let (m1, m2) = (1usize << q1, 1usize << q2);
    swap_runs(real, imag, q1, q2, |_, _| (m1, m2));
}

/// Controlled phase: multiplies the (1, 1) runs by `exp(i * theta)`.
pub fn controlled_phase(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize, theta: f64) {
    let (lo, hi) = (q1.min(q2), q1.max(q2));
    let run = 1usize << lo;
    let both = (1usize << lo) | (1usize << hi);
    let (s, c) = theta.sin_cos();
    let dim = real.len();
    for_each_base(dim, lo, hi, |base| {
        let start = base + both;
        for (r, i) in real[start..start + run].iter_mut().zip(&mut imag[start..start + run]) {
            let (x, y) = (*r, *i);
            *r = x * c - y * s;
            *i = x * s + y * c;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(n: usize) -> (Vec<f64>, Vec<f64>) {
        let dim = 1usize << n;
        ((0..dim).map(|i| (i as f64 * 0.37).sin()).collect(), (0..dim).map(|i| (i as f64 * 0.91).cos()).collect())
    }

    /// Index-by-index reference: `map` gives the source index for each destination index.
    fn permuted(n: usize, map: impl Fn(usize) -> usize) -> (Vec<f64>, Vec<f64>) {
        let (re, im) = state(n);
        ((0..re.len()).map(|i| re[map(i)]).collect(), (0..re.len()).map(|i| im[map(i)]).collect())
    }

    #[test]
    fn kernels_match_naive_references_for_all_pairs() {
        let n = 6;
        for a in 0..n {
            for b in (0..n).filter(|&b| b != a) {
                let (ma, mb) = (1usize << a, 1usize << b);
                let cases: [(&str, Box<dyn Fn(&mut [f64], &mut [f64])>, Box<dyn Fn(usize) -> usize>); 3] = [
                    ("cnot", Box::new(move |r, i| cnot(r, i, a, b)),
                     Box::new(move |i| if i & ma != 0 { i ^ mb } else { i })),
                    ("anticnot", Box::new(move |r, i| anticnot(r, i, a, b)),
                     Box::new(move |i| if i & ma == 0 { i ^ mb } else { i })),
                    ("swap", Box::new(move |r, i| swap(r, i, a, b)),
                     Box::new(move |i| if (i & ma != 0) != (i & mb != 0) { i ^ ma ^ mb } else { i })),
                ];
                for (name, apply, map) in cases {
                    let (mut re, mut im) = state(n);
                    apply(&mut re, &mut im);
                    assert_eq!((re, im), permuted(n, map), "{name}({a},{b})");
                }
                let (mut re, mut im) = state(n);
                controlled_phase(&mut re, &mut im, a, b, 0.7);
                let (r0, i0) = state(n);
                let (s, c) = 0.7f64.sin_cos();
                for k in 0..re.len() {
                    let on = k & ma != 0 && k & mb != 0;
                    let (wr, wi) = if on { (r0[k] * c - i0[k] * s, r0[k] * s + i0[k] * c) } else { (r0[k], i0[k]) };
                    assert!((re[k] - wr).abs() < 1e-15 && (im[k] - wi).abs() < 1e-15, "cphase({a},{b}) at {k}");
                }
            }
        }
    }
}
