// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use std::f64::consts::FRAC_1_SQRT_2;

/// 4-Qubit Density Matrix (16x16 complex matrix stored as flat real/imag arrays)
#[derive(Clone, Debug)]
struct DensityMatrix4Q {
    re: [[f64; 16]; 16],
    im: [[f64; 16]; 16],
}

impl DensityMatrix4Q {
    fn new_zero_state() -> Self {
        let mut dm = Self {
            re: [[0.0; 16]; 16],
            im: [[0.0; 16]; 16],
        };
        dm.re[0][0] = 1.0; // |0000><0000|
        dm
    }

    /// Apply Hadamard on qubit `q` (0=Q1, 1=Q2, 2=Q3, 3=Q4)
    fn apply_h(&mut self, q: usize) {
        let mask = 1 << (3 - q);
        let mut next = self.clone();
        // U * rho
        for i in 0..16 {
            for j in 0..16 {
                let sign = if (i & mask) != 0 { -1.0 } else { 1.0 };
                next.re[i][j] =
                    FRAC_1_SQRT_2 * (self.re[i & !mask][j] + sign * self.re[i | mask][j]);
                next.im[i][j] =
                    FRAC_1_SQRT_2 * (self.im[i & !mask][j] + sign * self.im[i | mask][j]);
            }
        }
        // (U * rho) * U^\dagger
        for i in 0..16 {
            for j in 0..16 {
                let sign = if (j & mask) != 0 { -1.0 } else { 1.0 };
                self.re[i][j] =
                    FRAC_1_SQRT_2 * (next.re[i][j & !mask] + sign * next.re[i][j | mask]);
                self.im[i][j] =
                    FRAC_1_SQRT_2 * (next.im[i][j & !mask] + sign * next.im[i][j | mask]);
            }
        }
    }

    /// Apply CNOT with control `c` and target `t`, followed by depolarizing noise `p`
    fn apply_noisy_cnot(&mut self, c: usize, t: usize, p: f64) {
        let c_mask = 1 << (3 - c);
        let t_mask = 1 << (3 - t);
        let mut next = self.clone();

        for i in 0..16 {
            let i_mapped = if (i & c_mask) != 0 { i ^ t_mask } else { i };
            for j in 0..16 {
                let j_mapped = if (j & c_mask) != 0 { j ^ t_mask } else { j };
                next.re[i_mapped][j_mapped] = self.re[i][j];
                next.im[i_mapped][j_mapped] = self.im[i][j];
            }
        }
        *self = next;
        self.depolarize_qubit(c, p);
        self.depolarize_qubit(t, p);
    }

    /// Single-qubit depolarizing channel: rho -> (1 - p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
    fn depolarize_qubit(&mut self, q: usize, p: f64) {
        let mask = 1 << (3 - q);
        let mut next = self.clone();
        for i in 0..16 {
            for j in 0..16 {
                let same_bit = (i & mask) == (j & mask);
                // X rho X flips both indices at mask
                let x_re = self.re[i ^ mask][j ^ mask];
                let x_im = self.im[i ^ mask][j ^ mask];
                // Z rho Z negates off-diagonal elements on qubit q
                let z_sign = if same_bit { 1.0 } else { -1.0 };
                let z_re = z_sign * self.re[i][j];
                let z_im = z_sign * self.im[i][j];
                // Y rho Y = -Z X rho X Z
                let y_re = z_sign * x_re;
                let y_im = z_sign * x_im;

                next.re[i][j] = (1.0 - p) * self.re[i][j] + (p / 3.0) * (x_re + y_re + z_re);
                next.im[i][j] = (1.0 - p) * self.im[i][j] + (p / 3.0) * (x_im + y_im + z_im);
            }
        }
        *self = next;
    }

    /// Compute Bell state fidelity <Phi+| Tr_{2,3}(rho) |Phi+> on Qubits 1 and 4
    #[allow(clippy::identity_op)]
    fn bell_fidelity_1_and_4(&self, post_select_syndrome_zero: bool) -> (f64, f64) {
        let mut fid_sum = 0.0;
        let mut prob_sum = 0.0;

        // Iterate over middle mediator states m in {00, 01, 10, 11} for (Q2, Q3)
        for m in 0..4 {
            if post_select_syndrome_zero && m != 0 {
                continue; // Only keep syndrome (s2=0, s3=0)
            }
            let idx_0m0 = (0 << 3) | (m << 1) | 0; // |0, m2, m3, 0>
            let idx_1m1 = (1 << 3) | (m << 1) | 1; // |1, m2, m3, 1>
            let idx_0m1 = (0 << 3) | (m << 1) | 1;
            let idx_1m0 = (1 << 3) | (m << 1) | 0;

            let p_subspace = self.re[idx_0m0][idx_0m0]
                + self.re[idx_1m1][idx_1m1]
                + self.re[idx_0m1][idx_0m1]
                + self.re[idx_1m0][idx_1m0];

            let bell_overlap = 0.5
                * (self.re[idx_0m0][idx_0m0]
                    + self.re[idx_1m1][idx_1m1]
                    + self.re[idx_0m0][idx_1m1]
                    + self.re[idx_1m1][idx_0m0]);

            fid_sum += bell_overlap;
            prob_sum += p_subspace;
        }
        (fid_sum / prob_sum, prob_sum)
    }

    /// Evaluate Teleportation + Syndrome Check:
    /// - Q2 (bit 1 of m) is the X-basis Teleportation Bridge:
    ///   m_q2 == 0 -> |Phi+>, m_q2 == 1 -> |Phi-> (corrected to |Phi+> via classical Z-frame!)
    /// - Q3 (bit 0 of m) is the Z1*Z4 Syndrome Ancilla (must be 0 for no bit-flip fault)
    #[allow(clippy::identity_op)]
    fn teleported_bell_fidelity_1_and_4(&self) -> (f64, f64) {
        let mut fid_sum = 0.0;
        let mut prob_sum = 0.0;

        // Only accept shots where Syndrome Ancilla Q3 == 0 (m = 0b00 or m = 0b10)
        for &m in &[0usize, 2usize] {
            let phase_sign = if m == 2 { -1.0 } else { 1.0 }; // Classical Pauli-Z frame correction when Q2=1!

            let idx_0m0 = (0 << 3) | (m << 1) | 0;
            let idx_1m1 = (1 << 3) | (m << 1) | 1;
            let idx_0m1 = (0 << 3) | (m << 1) | 1;
            let idx_1m0 = (1 << 3) | (m << 1) | 0;

            let p_subspace = self.re[idx_0m0][idx_0m0]
                + self.re[idx_1m1][idx_1m1]
                + self.re[idx_0m1][idx_0m1]
                + self.re[idx_1m0][idx_1m0];

            // With classical Z-frame correction, <Phi-| Z_1 |Phi-> = <Phi+|Phi+>
            let bell_overlap = 0.5
                * (self.re[idx_0m0][idx_0m0]
                    + self.re[idx_1m1][idx_1m1]
                    + phase_sign * (self.re[idx_0m0][idx_1m1] + self.re[idx_1m1][idx_0m0]));

            fid_sum += bell_overlap;
            prob_sum += p_subspace;
        }
        (fid_sum / prob_sum, prob_sum)
    }
}

fn main() {
    let p_gate = 0.01; // 1% per-qubit gate error
    let p_idle = 0.005; // 0.5% idle decoherence per step

    // --- 1. Standard Sequential SWAP Chain (1 -> 2 -> 4) ---
    let mut dm_swap = DensityMatrix4Q::new_zero_state();
    dm_swap.apply_h(0);
    dm_swap.apply_noisy_cnot(0, 1, p_gate); // Step 1: Entangle 1->2
    dm_swap.depolarize_qubit(2, p_idle);
    dm_swap.depolarize_qubit(3, p_idle);
    // SWAP 2 and 4 (3 CNOTs)
    dm_swap.apply_noisy_cnot(1, 3, p_gate);
    dm_swap.apply_noisy_cnot(3, 1, p_gate);
    dm_swap.apply_noisy_cnot(1, 3, p_gate);
    for _ in 0..3 {
        dm_swap.depolarize_qubit(0, p_idle); // Q1 idles during 3-CNOT SWAP
    }
    let (fid_swap, _) = dm_swap.bell_fidelity_1_and_4(false);

    // --- 2. 1 + (2,3) = 4 Dual-Path Plaquette with Syndrome Verification ---
    let mut dm_plaq = DensityMatrix4Q::new_zero_state();
    dm_plaq.apply_h(0);
    // Step 1: Parallel fan-out across 1->2 and 1->3
    dm_plaq.apply_noisy_cnot(0, 1, p_gate);
    dm_plaq.apply_noisy_cnot(1, 3, p_gate); // Route to 4
                                            // Step 2: Parity uncompute & syndrome check on Q2 and Q3
    dm_plaq.apply_noisy_cnot(3, 1, p_gate);
    dm_plaq.apply_noisy_cnot(0, 2, p_gate);
    dm_plaq.apply_noisy_cnot(3, 2, p_gate); // Q3 checks Z1*Z4 parity

    let (fid_raw, _) = dm_plaq.bell_fidelity_1_and_4(false);
    let (fid_post, keep_rate) = dm_plaq.bell_fidelity_1_and_4(true);

    // --- 4. True Dual-Stabilizer 1 + (2,3) = 4: Q2=X1*X4 (Control), Q3=Z1*Z4 (Target) ---
    let mut dm_dual = DensityMatrix4Q::new_zero_state();
    // Step 1: Put Q2 in |+> to act as the X-stabilizer bus across (1,4)
    dm_dual.apply_h(1);

    // Interleaved Plaquette Schedule (prevents Hook Error propagation):
    dm_dual.apply_noisy_cnot(1, 0, p_gate); // Q2 -> Q1 (X-branch leg 1)
    dm_dual.apply_noisy_cnot(0, 2, p_gate); // Q1 -> Q3 (Z-branch leg 1)
    dm_dual.apply_noisy_cnot(1, 3, p_gate); // Q2 -> Q4 (X-branch leg 2)
    dm_dual.apply_noisy_cnot(3, 2, p_gate); // Q4 -> Q3 (Z-branch leg 2)

    // Close X-basis check on Q2
    dm_dual.apply_h(1);

    let (fid_dual_post, keep_dual) = dm_dual.bell_fidelity_1_and_4(true);

    // --- 5. Purified 1 + (2,3) = 4 with Final Parity Verification ---
    // Start with the 2-CNOT coherent route via Q2, then verify both X and Z via (Q2, Q3)
    let mut dm_purified = DensityMatrix4Q::new_zero_state();
    dm_purified.apply_h(0);
    dm_purified.apply_noisy_cnot(0, 1, p_gate); // 1 -> 2
    dm_purified.apply_noisy_cnot(1, 3, p_gate); // 2 -> 4
                                                // Uncompute Q2 in Z, then check Z1*Z4 on Q3 AND X-phase on Q2
    dm_purified.apply_noisy_cnot(0, 1, p_gate); // Uncompute Q2 using Q1 (catches 1->2 faults!)
    dm_purified.apply_noisy_cnot(0, 2, p_gate); // Q1 -> Q3
    dm_purified.apply_noisy_cnot(3, 2, p_gate); // Q4 -> Q3

    let (fid_purified, keep_purified) = dm_purified.bell_fidelity_1_and_4(true);

    // --- 6. Measurement-Based Teleportation (Q2) + Z-Syndrome Check (Q3) + Pauli Frame ---
    // Uses only 4 CNOTs total (1 per edge of the square!), zero SWAPs, and classical Pauli frame recovery
    let mut dm_mbqc = DensityMatrix4Q::new_zero_state();
    dm_mbqc.apply_h(1); // Q2 in |+> distributes entanglement to Q1 and Q4
    dm_mbqc.apply_noisy_cnot(1, 0, p_gate); // Edge 1: Q2 -> Q1
    dm_mbqc.apply_noisy_cnot(1, 3, p_gate); // Edge 2: Q2 -> Q4

    // Q3 checks Z1*Z4 across the opposite two edges of the square
    dm_mbqc.apply_noisy_cnot(0, 2, p_gate); // Edge 3: Q1 -> Q3
    dm_mbqc.apply_noisy_cnot(3, 2, p_gate); // Edge 4: Q4 -> Q3

    // Measure Q2 in X-basis (teleports 1-4 link with classical Pauli frame update)
    dm_mbqc.apply_h(1);

    let (fid_mbqc, keep_mbqc) = dm_mbqc.teleported_bell_fidelity_1_and_4();

    println!("=== 2x2 Plaquette Routing & Syndrome Benchmark: 1 + (2,3) = 4 ===");
    println!(
        "1. Sequential SWAP (1 -> 2 -> 4) Fidelity      : {:.4}",
        fid_swap
    );
    println!(
        "2. Dual-Path Raw Fidelity (No Post-Selection)  : {:.4}",
        fid_raw
    );
    println!(
        "3. Dual-Path + (2,3) Syndrome Checked Fidelity : {:.4} (Acceptance: {:.1}%)",
        fid_post,
        keep_rate * 100.0
    );
    println!(
        "4. True Dual-Stabilizer (Q2=X1X4, Q3=Z1Z4)     : {:.4} (Acceptance: {:.1}%)",
        fid_dual_post,
        keep_dual * 100.0
    );
    println!(
        "5. Symmetric Dual-Bridge (Q2=1->2, Q3=1->4)    : {:.4} (Acceptance: {:.1}%)",
        fid_purified,
        keep_purified * 100.0
    );
    println!(
        "6. 4-Edge Plaquette Teleport + Pauli Frame     : {:.4} (Acceptance: {:.1}%)",
        fid_mbqc,
        keep_mbqc * 100.0
    );
}
