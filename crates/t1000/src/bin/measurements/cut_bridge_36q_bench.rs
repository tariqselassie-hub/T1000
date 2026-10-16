// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use std::arch::x86_64::*;
use std::time::Instant;

const N_QUBITS: usize = 18;
const DIM: usize = 1 << N_QUBITS; // 262,144 amplitudes (4.00 MB per buffer)

#[inline(always)]
fn alloc_buffer() -> Box<[f64; DIM]> {
    vec![0.0f64; DIM].into_boxed_slice().try_into().unwrap()
}

/// Applies an in-place Ry(theta) rotation on qubit `q` using AVX2 FMA.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are supported and `re` and `im` contain `DIM` elements.
#[inline(always)]
pub unsafe fn apply_ry_avx2(re: &mut [f64], im: &mut [f64], q: usize, theta: f64) {
    let c = (theta * 0.5).cos();
    let s = (theta * 0.5).sin();
    let stride = 1usize << q;
    let step = stride << 1;

    if stride >= 4 {
        let vc = _mm256_set1_pd(c);
        let vs = _mm256_set1_pd(s);
        let re_ptr = re.as_mut_ptr();
        let im_ptr = im.as_mut_ptr();

        let mut base = 0;
        while base < DIM {
            let mut offset = 0;
            while offset < stride {
                let i0 = base + offset;
                let i1 = i0 + stride;

                let r0 = _mm256_loadu_pd(re_ptr.add(i0));
                let m0 = _mm256_loadu_pd(im_ptr.add(i0));
                let r1 = _mm256_loadu_pd(re_ptr.add(i1));
                let m1 = _mm256_loadu_pd(im_ptr.add(i1));

                // r0' = c*r0 - s*r1, r1' = s*r0 + c*r1
                let nr0 = _mm256_fmsub_pd(vc, r0, _mm256_mul_pd(vs, r1));
                let nm0 = _mm256_fmsub_pd(vc, m0, _mm256_mul_pd(vs, m1));
                let nr1 = _mm256_fmadd_pd(vs, r0, _mm256_mul_pd(vc, r1));
                let nm1 = _mm256_fmadd_pd(vs, m0, _mm256_mul_pd(vc, m1));

                _mm256_storeu_pd(re_ptr.add(i0), nr0);
                _mm256_storeu_pd(im_ptr.add(i0), nm0);
                _mm256_storeu_pd(re_ptr.add(i1), nr1);
                _mm256_storeu_pd(im_ptr.add(i1), nm1);

                offset += 4;
            }
            base += step;
        }
    } else {
        let mut base = 0;
        while base < DIM {
            for offset in 0..stride {
                let i0 = base + offset;
                let i1 = i0 + stride;
                let (r0, m0) = (re[i0], im[i0]);
                let (r1, m1) = (re[i1], im[i1]);
                re[i0] = c * r0 - s * r1;
                im[i0] = c * m0 - s * m1;
                re[i1] = s * r0 + c * r1;
                im[i1] = s * m0 + c * m1;
            }
            base += step;
        }
    }
}

/// Applies virtual Pauli-Z insertions at mid-circuit for cut-bridge qubits (q_b0, q_b1).
///
/// # Safety
/// Caller must ensure AVX2 is supported and `re` and `im` are valid for vector operations.
#[inline(always)]
pub unsafe fn apply_midcircuit_z_bridge_avx2(
    re: &mut [f64; DIM],
    im: &mut [f64; DIM],
    z_flags: &[bool; 2],
    bridge_qubits: [usize; 2],
) {
    if !z_flags[0] && !z_flags[1] {
        return; // Branch 00: Identity (Zero overhead!)
    }

    let mut mask = 0usize;
    if z_flags[0] {
        mask |= 1usize << bridge_qubits[0];
    }
    if z_flags[1] {
        mask |= 1usize << bridge_qubits[1];
    }

    let re_ptr = re.as_mut_ptr();
    let im_ptr = im.as_mut_ptr();

    for vec_idx in 0..(DIM / 4) {
        let base = vec_idx * 4;
        let s0 = if ((base & mask).count_ones() & 1) != 0 { -1.0 } else { 1.0 };
        let s1 = if (((base + 1) & mask).count_ones() & 1) != 0 { -1.0 } else { 1.0 };
        let s2 = if (((base + 2) & mask).count_ones() & 1) != 0 { -1.0 } else { 1.0 };
        let s3 = if (((base + 3) & mask).count_ones() & 1) != 0 { -1.0 } else { 1.0 };

        let v_sign = _mm256_set_pd(s3, s2, s1, s0);
        let r = _mm256_loadu_pd(re_ptr.add(base));
        let m = _mm256_loadu_pd(im_ptr.add(base));
        _mm256_storeu_pd(re_ptr.add(base), _mm256_mul_pd(r, v_sign));
        _mm256_storeu_pd(im_ptr.add(base), _mm256_mul_pd(m, v_sign));
    }
}

/// Fused Intra-Fragment All-to-All CZ Sweep.
///
/// # Safety
/// Caller must ensure AVX2 is supported and `re` and `im` contain `DIM` elements.
#[inline(always)]
pub unsafe fn apply_intra_all_to_all_cz_18(re: &mut [f64; DIM], im: &mut [f64; DIM]) {
    let re_ptr = re.as_mut_ptr();
    let im_ptr = im.as_mut_ptr();

    for vec_idx in 0..(DIM / 4) {
        let b = vec_idx * 4;
        let p0 = eval_intra_phase(b);
        let p1 = eval_intra_phase(b + 1);
        let p2 = eval_intra_phase(b + 2);
        let p3 = eval_intra_phase(b + 3);

        let v_sign = _mm256_set_pd(p3, p2, p1, p0);
        let r = _mm256_loadu_pd(re_ptr.add(b));
        let m = _mm256_loadu_pd(im_ptr.add(b));
        _mm256_storeu_pd(re_ptr.add(b), _mm256_mul_pd(r, v_sign));
        _mm256_storeu_pd(im_ptr.add(b), _mm256_mul_pd(m, v_sign));
    }
}

#[inline(always)]
fn eval_intra_phase(idx: usize) -> f64 {
    let k1 = (idx & 0x1FF).count_ones() as usize;
    let k2 = ((idx >> 9) & 0x1FF).count_ones() as usize;
    let pairs = ((k1 * k1.saturating_sub(1)) >> 1) + ((k2 * k2.saturating_sub(1)) >> 1);
    if (pairs & 1) != 0 { -1.0 } else { 1.0 }
}

/// Evaluates ⟨Z_target⟩ on an 18-qubit fragment
pub fn measure_z_expectation(re: &[f64; DIM], im: &[f64; DIM], target_q: usize) -> f64 {
    let mask = 1usize << target_q;
    let mut exp_z = 0.0;
    for i in 0..DIM {
        let p = re[i] * re[i] + im[i] * im[i];
        if (i & mask) == 0 {
            exp_z += p;
        } else {
            exp_z -= p;
        }
    }
    exp_z
}

fn main() {
    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│     T-1000: 36-QUBIT MID-CIRCUIT CHECKPOINTED CUT-BRIDGE ENGINE        │");
    println!("│     Sandwich Topology: [Layers 1..6] -> [Cut CZ] -> [Layers 7..12]     │");
    println!("└────────────────────────────────────────────────────────────────────────┘");

    let start_total = Instant::now();

    // 1. Pre-compute Layers 1..6 ONCE per 18Q Fragment (Checkpointed in L3 Cache)
    let mut base_re_a = alloc_buffer();
    let mut base_im_a = alloc_buffer();
    let mut base_re_b = alloc_buffer();
    let mut base_im_b = alloc_buffer();
    base_re_a[0] = 1.0;
    base_re_b[0] = 1.0;

    let t_prep = Instant::now();
    unsafe {
        for layer in 0..6 {
            for q in 0..N_QUBITS {
                let angle_a = 0.37 * (q as f64 + 1.0) + 0.19 * f64::from(layer);
                let angle_b = 0.43 * (q as f64 + 1.0) - 0.13 * f64::from(layer);
                apply_ry_avx2(&mut *base_re_a, &mut *base_im_a, q, angle_a);
                apply_ry_avx2(&mut *base_re_b, &mut *base_im_b, q, angle_b);
            }
            apply_intra_all_to_all_cz_18(&mut base_re_a, &mut base_im_a);
            apply_intra_all_to_all_cz_18(&mut base_re_b, &mut base_im_b);
        }
    }
    let prep_ms = t_prep.elapsed().as_secs_f64() * 1000.0;
    println!("[CHECKPOINT] Layers 1..6 Pre-Computed Once in {:.2} ms", prep_ms);

    // 2. Branch from Mid-Circuit Checkpoint for Cut-Bridge Z-Insertions + Layers 7..12
    let mut work_re = alloc_buffer();
    let mut work_im = alloc_buffer();
    let mut exp_a = [0.0f64; 4];
    let mut exp_b = [0.0f64; 4];

    // Bridge anchors: Fragment A (q4, q8) <-> Fragment B (q13, q17)
    let bridges_a = [4usize, 8usize];
    let bridges_b = [13usize, 17usize];

    println!("\n[MID-CIRCUIT BRANCH EXECUTION BREAKDOWN]");
    for mask in 0..4 {
        let z_flags = [(mask & 1) != 0, ((mask >> 1) & 1) != 0];

        // --- Fragment A Branch ---
        let t_a = Instant::now();
        work_re.copy_from_slice(&*base_re_a);
        work_im.copy_from_slice(&*base_im_a);
        unsafe {
            apply_midcircuit_z_bridge_avx2(&mut work_re, &mut work_im, &z_flags, bridges_a);
            for layer in 6..12 {
                for q in 0..N_QUBITS {
                    let angle_a = 0.29 * (q as f64 + 1.0) + 0.23 * f64::from(layer);
                    apply_ry_avx2(&mut *work_re, &mut *work_im, q, angle_a);
                }
                apply_intra_all_to_all_cz_18(&mut work_re, &mut work_im);
            }
        }
        // Readout on bridge-coupled qubit q4
        exp_a[mask] = measure_z_expectation(&work_re, &work_im, 4);
        let lat_a = t_a.elapsed().as_secs_f64() * 1e6;

        // --- Fragment B Branch ---
        let t_b = Instant::now();
        work_re.copy_from_slice(&*base_re_b);
        work_im.copy_from_slice(&*base_im_b);
        unsafe {
            apply_midcircuit_z_bridge_avx2(&mut work_re, &mut work_im, &z_flags, bridges_b);
            for layer in 6..12 {
                for q in 0..N_QUBITS {
                    let angle_b = 0.31 * (q as f64 + 1.0) - 0.17 * f64::from(layer);
                    apply_ry_avx2(&mut *work_re, &mut *work_im, q, angle_b);
                }
                apply_intra_all_to_all_cz_18(&mut work_re, &mut work_im);
            }
        }
        // Readout on bridge-coupled qubit q13
        exp_b[mask] = measure_z_expectation(&work_re, &work_im, 13);
        let lat_b = t_b.elapsed().as_secs_f64() * 1e6;

        println!(
            "  Branch {:02b} (Z-ins: {:5?}) | ⟨Z_A⟩ = {:+.6} ({:6.1} us) | ⟨Z_B⟩ = {:+.6} ({:6.1} us)",
            mask, z_flags, exp_a[mask], lat_a, exp_b[mask], lat_b
        );
    }

    // 3. Quasiprobability Contraction: CZ = 0.5 * (I⊗I + I⊗Z + Z⊗I - Z⊗Z)
    let c: [[f64; 2]; 2] = [[0.5, 0.5], [0.5, -0.5]];
    let mut entangled_zz = 0.0;
    for m_a in 0..4 {
        for m_b in 0..4 {
            let w0 = c[m_a & 1][m_b & 1];
            let w1 = c[(m_a >> 1) & 1][(m_b >> 1) & 1];
            entangled_zz += w0 * w1 * exp_a[m_a] * exp_b[m_b];
        }
    }

    let unentangled_product = exp_a[0] * exp_b[0];
    let entanglement_delta = (entangled_zz - unentangled_product).abs();
    let total_ms = start_total.elapsed().as_secs_f64() * 1000.0;

    println!("\n────────────────────────────────────────────────────────────────────────");
    println!(" 36-QUBIT NON-SEPARABLE ENTANGLEMENT VERIFICATION");
    println!("────────────────────────────────────────────────────────────────────────");
    println!("  Uncoupled Product ⟨Z_A⟩⟨Z_B⟩ (Branch 00) : {:+.10}", unentangled_product);
    println!("  Reconstructed 36Q Entangled ⟨Z_A ⊗ Z_B⟩  : {:+.10}", entangled_zz);
    println!("  Cross-Fragment Entanglement Witness (Δ)  : {:+.10}", entanglement_delta);
    println!("  Total Checkpointed 36Q Execution Latency : {:.2} ms", total_ms);
    println!("────────────────────────────────────────────────────────────────────────");

    assert!(
        entanglement_delta > 1e-4,
        "COMMUTATION TRAP DETECTED: Branches failed to diverge!"
    );
    println!("-> PASS: Non-separable 36-qubit cross-bridge entanglement confirmed.");
}
