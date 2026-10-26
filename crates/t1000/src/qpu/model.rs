// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Bit-exact software model of the 1000_QPU statevector engine.
//!
//! The model is the specification the RTL is built and checked against: every operation is a
//! pure function of the stored integers, so the hardware's pair ordering, banking and pipelining
//! cannot change a result.

use super::fixed::{round_and_saturate, COEF_BITS, STATE_MAX, STATE_MIN};

/// Largest register: `2^14` amplitudes.
pub const MAX_QUBITS: usize = 14;
pub const MAX_DIM: usize = 1 << MAX_QUBITS;

/// A controlled single-qubit gate: the 2x2 complex matrix is applied to `target` on every basis
/// pair whose index has all `ctrl` bits set. Coefficients are 25-bit signed integers in the order
/// `m00r, m00i, m01r, m01i, m10r, m10i, m11r, m11i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gate {
    pub target: u8,
    pub ctrl: u16,
    pub m: [i64; 8],
}

/// Sign-extends the low 25 bits, as the hardware does when it receives a 32-bit coefficient word.
pub fn sign_extend_coef(raw: i64) -> i64 {
    let shift = 64 - COEF_BITS;
    (raw << shift) >> shift
}

/// The statevector engine state: real and imaginary components of `2^n` amplitudes.
#[derive(Debug, Clone)]
pub struct QpuModel {
    n: usize,
    re: Vec<i64>,
    im: Vec<i64>,
    circuit: Vec<Gate>,
}

impl Default for QpuModel {
    fn default() -> Self {
        QpuModel {
            n: MAX_QUBITS,
            re: vec![0; MAX_DIM],
            im: vec![0; MAX_DIM],
            circuit: Vec::new(),
        }
    }
}

impl QpuModel {
    pub fn new() -> Self {
        Self::default()
    }

    /// Active register size.
    pub fn qubits(&self) -> usize {
        self.n
    }

    pub fn dim(&self) -> usize {
        1 << self.n
    }

    /// Selects the active register size (amplitudes are kept); `false` if outside `1..=14`.
    pub fn set_qubits(&mut self, n: usize) -> bool {
        if (1..=MAX_QUBITS).contains(&n) {
            self.n = n;
            true
        } else {
            false
        }
    }

    /// Zeroes the active amplitudes and sets amplitude 0 to the real value `one`.
    pub fn reset(&mut self, one: i64) {
        let dim = self.dim();
        self.re[..dim].fill(0);
        self.im[..dim].fill(0);
        self.re[0] = one.clamp(STATE_MIN, STATE_MAX);
    }

    /// Writes amplitudes starting at `addr`; `false` (no change) if the range leaves the register.
    pub fn load(&mut self, addr: usize, data: &[(i64, i64)]) -> bool {
        if addr + data.len() > self.dim() {
            return false;
        }
        for (k, &(r, i)) in data.iter().enumerate() {
            self.re[addr + k] = r.clamp(STATE_MIN, STATE_MAX);
            self.im[addr + k] = i.clamp(STATE_MIN, STATE_MAX);
        }
        true
    }

    /// Reads `count` amplitudes from `addr`.
    pub fn read(&self, addr: usize, count: usize) -> Option<Vec<(i64, i64)>> {
        if addr + count > self.dim() {
            return None;
        }
        Some(
            (addr..addr + count)
                .map(|k| (self.re[k], self.im[k]))
                .collect(),
        )
    }

    /// Whether a gate is legal for the active register.
    pub fn gate_is_valid(&self, g: &Gate) -> bool {
        let t = usize::from(g.target);
        t < self.n && usize::from(g.ctrl) < self.dim() && g.ctrl & (1 << t) == 0
    }

    /// Applies a controlled single-qubit gate; `false` (no change) if it is not legal.
    pub fn apply_gate(&mut self, g: &Gate) -> bool {
        if !self.gate_is_valid(g) {
            return false;
        }
        let m: Vec<i128> =
            g.m.iter()
                .map(|&c| i128::from(sign_extend_coef(c)))
                .collect();
        let bit = 1usize << g.target;
        let ctrl = usize::from(g.ctrl);
        for i0 in 0..self.dim() {
            if i0 & bit != 0 || i0 & ctrl != ctrl {
                continue;
            }
            let i1 = i0 | bit;
            let (a0r, a0i) = (i128::from(self.re[i0]), i128::from(self.im[i0]));
            let (a1r, a1i) = (i128::from(self.re[i1]), i128::from(self.im[i1]));
            self.re[i0] = round_and_saturate(m[0] * a0r - m[1] * a0i + m[2] * a1r - m[3] * a1i);
            self.im[i0] = round_and_saturate(m[0] * a0i + m[1] * a0r + m[2] * a1i + m[3] * a1r);
            self.re[i1] = round_and_saturate(m[4] * a0r - m[5] * a0i + m[6] * a1r - m[7] * a1i);
            self.im[i1] = round_and_saturate(m[4] * a0i + m[5] * a0r + m[6] * a1i + m[7] * a1r);
        }
        true
    }

    /// `Σ re² + im²` over the active register, exactly.
    pub fn norm_squared(&self) -> i128 {
        (0..self.dim())
            .map(|k| i128::from(self.re[k]).pow(2) + i128::from(self.im[k]).pow(2))
            .sum()
    }

    /// `Σ (-1)^{popcount(i ∧ mask)} (re² + im²)`, the unnormalised expectation of a Z string;
    /// `None` if `mask` reaches outside the register.
    pub fn expect_z(&self, mask: usize) -> Option<i128> {
        if mask >= self.dim() {
            return None;
        }
        Some(
            (0..self.dim())
                .map(|k| {
                    let p = i128::from(self.re[k]).pow(2) + i128::from(self.im[k]).pow(2);
                    if (k & mask).count_ones() % 2 == 0 {
                        p
                    } else {
                        -p
                    }
                })
                .sum(),
        )
    }

    /// Loads a microcode circuit sequence (up to 128 gates).
    pub fn load_circuit(&mut self, gates: Vec<Gate>) -> bool {
        if gates.len() > 128 {
            return false;
        }
        self.circuit = gates;
        true
    }

    /// Executes the loaded microcode circuit burst sequence.
    pub fn run_burst(&mut self) -> bool {
        let circuit = self.circuit.clone();
        for g in &circuit {
            if !self.apply_gate(g) {
                return false;
            }
        }
        true
    }

    /// Executes the loaded microcode circuit burst, then evaluates Expect-Z.
    pub fn run_burst_expect(&mut self, mask: usize) -> Option<i128> {
        if !self.run_burst() {
            return None;
        }
        self.expect_z(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixed::{dequantize_state, quantize_coef, quantize_state};
    use super::*;

    const F: u32 = 28;

    fn gate(target: u8, ctrl: u16, m: [f64; 8]) -> Gate {
        Gate {
            target,
            ctrl,
            m: m.map(quantize_coef),
        }
    }

    fn hadamard(t: u8) -> Gate {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        gate(t, 0, [s, 0.0, s, 0.0, s, 0.0, -s, 0.0])
    }

    fn cnot(c: u8, t: u8) -> Gate {
        gate(t, 1 << c, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    }

    fn amp(q: &QpuModel, k: usize) -> (f64, f64) {
        let v = q.read(k, 1).unwrap()[0];
        (dequantize_state(v.0, F), dequantize_state(v.1, F))
    }

    fn zero_state(n: usize) -> QpuModel {
        let mut q = QpuModel::new();
        q.set_qubits(n);
        q.reset(quantize_state(1.0, F));
        q
    }

    #[test]
    fn bell_state_from_hadamard_and_cnot() {
        let mut q = zero_state(3);
        assert!(q.apply_gate(&hadamard(0)));
        assert!(q.apply_gate(&cnot(0, 2)));
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((amp(&q, 0).0 - s).abs() < 1e-6);
        assert!((amp(&q, 0b101).0 - s).abs() < 1e-6);
        for k in [1, 2, 3, 4, 6, 7] {
            assert!(amp(&q, k).0.abs() < 1e-6, "k = {k}");
        }
    }

    #[test]
    fn controls_gate_only_the_matching_pairs() {
        let mut q = zero_state(2);
        // X on qubit 1 controlled by qubit 0 does nothing while qubit 0 is |0>.
        assert!(q.apply_gate(&cnot(0, 1)));
        assert_eq!(q.read(0, 4).unwrap()[0].0, 1 << F);
        let x0 = gate(0, 0, [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        assert!(q.apply_gate(&x0));
        assert!(q.apply_gate(&cnot(0, 1)));
        assert_eq!(q.read(3, 1).unwrap()[0].0, 1 << F);
    }

    #[test]
    fn complex_coefficients_rotate_the_phase() {
        let mut q = zero_state(1);
        q.apply_gate(&hadamard(0));
        // S gate: diag(1, i) on |1>.
        q.apply_gate(&gate(0, 0, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]));
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let (r, i) = amp(&q, 1);
        assert!(r.abs() < 1e-6 && (i - s).abs() < 1e-6, "{r} {i}");
    }

    #[test]
    fn a_hadamard_pair_returns_the_state_to_within_quantisation() {
        let mut q = zero_state(4);
        for t in 0..4 {
            q.apply_gate(&hadamard(t));
        }
        for t in 0..4 {
            q.apply_gate(&hadamard(t));
        }
        let (r, _) = amp(&q, 0);
        assert!((r - 1.0).abs() < 1e-5, "{r}");
        let norm =
            dequantize_state(1, 0) * 0.0 + q.norm_squared() as f64 / f64::powi(2.0, 2 * F as i32);
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn norm_and_z_expectation_are_exact_integer_sums() {
        let mut q = zero_state(2);
        q.apply_gate(&hadamard(0));
        let unit = 1i128 << (2 * F);
        let norm = q.norm_squared();
        assert!((norm - unit).abs() < unit / 100_000, "{norm}");
        // |+> on qubit 0: <Z0> = 0, <Z1> = 1.
        assert!(q.expect_z(0b01).unwrap().abs() < unit / 100_000);
        assert!((q.expect_z(0b10).unwrap() - unit).abs() < unit / 100_000);
        assert!(q.expect_z(4).is_none());
    }

    #[test]
    fn illegal_operations_change_nothing() {
        let mut q = zero_state(3);
        let before = q.read(0, 8).unwrap();
        assert!(!q.apply_gate(&hadamard(3)), "target outside the register");
        assert!(
            !q.apply_gate(&Gate {
                ctrl: 0b100,
                ..hadamard(2)
            }),
            "control equals target"
        );
        assert!(
            !q.apply_gate(&Gate {
                ctrl: 0b1000,
                ..hadamard(0)
            }),
            "control outside register"
        );
        assert!(!q.load(6, &[(1, 1), (2, 2), (3, 3)]));
        assert!(q.read(7, 2).is_none());
        assert_eq!(q.read(0, 8).unwrap(), before);
        assert!(!q.set_qubits(0) && !q.set_qubits(15));
    }

    #[test]
    fn results_saturate_instead_of_wrapping() {
        let mut q = QpuModel::new();
        q.set_qubits(1);
        q.load(0, &[(STATE_MAX, 0), (STATE_MAX, 0)]);
        // 2 * x overflows 32 bits.
        q.apply_gate(&gate(0, 0, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]));
        assert_eq!(q.read(0, 1).unwrap()[0].0, STATE_MAX);
    }

    #[test]
    fn only_the_low_25_bits_of_a_coefficient_word_are_used() {
        assert_eq!(sign_extend_coef(0x0200_0000 | 5), 5);
        assert_eq!(sign_extend_coef(0x01FF_FFFF), -1);
        assert_eq!(sign_extend_coef(-3), -3);
    }
}
