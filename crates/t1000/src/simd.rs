//! 256-bit AVX2 & FMA Hardware-Accelerated Vectorization Kernels.
//! Tailored for Intel Arrow Lake (Lion Cove / Skymont) and x86_64 processors.
//! Processes 4 double-precision complex amplitudes (8 x f64) per vector instruction.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Maximum qubit count supported by the fused terminal-readout kernels (fixed stack buffers).
pub const MAX_FUSED_READOUT_QUBITS: usize = 16;

// =========================================================================
// PRECONDITION GUARDS
// The AVX2 paths below perform unchecked loads/stores, so every public entry
// point validates its inputs first. Violations panic instead of corrupting memory.
// =========================================================================

/// Asserts a valid statevector buffer pair and returns its qubit count.
#[inline(always)]
pub(crate) fn assert_statevector(real: &[f64], imag: &[f64]) -> usize {
    assert_eq!(
        real.len(),
        imag.len(),
        "real/imag length mismatch ({} vs {})",
        real.len(),
        imag.len()
    );
    assert!(
        real.len() >= 2 && real.len().is_power_of_two(),
        "statevector length must be a power of two >= 2 (got {})",
        real.len()
    );
    real.len().trailing_zeros() as usize
}

#[inline(always)]
pub(crate) fn assert_qubit(qubit: usize, n_qubits: usize) {
    assert!(
        qubit < n_qubits,
        "qubit index {} out of range for {}-qubit state",
        qubit,
        n_qubits
    );
}

#[inline(always)]
pub(crate) fn assert_qubit_pair(q1: usize, q2: usize, n_qubits: usize) {
    assert_qubit(q1, n_qubits);
    assert_qubit(q2, n_qubits);
    assert_ne!(q1, q2, "two-qubit gate requires distinct qubits (got {} twice)", q1);
}

#[inline(always)]
pub(crate) fn assert_fused_readout_width(n: usize) {
    assert!(
        n <= MAX_FUSED_READOUT_QUBITS,
        "fused readout supports at most {} qubits (got {})",
        MAX_FUSED_READOUT_QUBITS,
        n
    );
}

/// Vectorized Single-Qubit Ry rotation:
/// [new_r0, new_r1]^T = [c, -s; s, c] [r0, r1]^T
/// Executes via 256-bit FMA instructions (4 x f64 per SIMD lane).
#[inline(always)]
pub fn apply_ry_kernel(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit(qubit, n_qubits);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        apply_ry_avx2(real, imag, qubit, theta);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                apply_ry_avx2(real, imag, qubit, theta);
                return;
            }
        }
        apply_ry_scalar(real, imag, qubit, theta);
    }
}

/// Vectorized Single-Qubit Rz rotation:
/// Multiplies |0> by exp(-i theta/2) and |1> by exp(i theta/2).
#[inline(always)]
pub fn apply_rz_kernel(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit(qubit, n_qubits);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        apply_rz_avx2(real, imag, qubit, theta);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                apply_rz_avx2(real, imag, qubit, theta);
                return;
            }
        }
        apply_rz_scalar(real, imag, qubit, theta);
    }
}

/// Vectorized Ring Controlled-Z (CZ) gate:
/// Flips sign of amplitudes where both bit q1 and bit q2 are 1.
#[inline(always)]
pub fn apply_cz_kernel(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit_pair(q1, q2, n_qubits);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        apply_cz_avx2(real, imag, q1, q2);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                apply_cz_avx2(real, imag, q1, q2);
                return;
            }
        }
        apply_cz_scalar(real, imag, q1, q2);
    }
}

/// Vectorized L2 Norm normalization: sum_i (r_i^2 + im_i^2) via FMA.
#[inline(always)]
pub fn normalize_kernel(real: &mut [f64], imag: &mut [f64]) {
    assert_eq!(real.len(), imag.len(), "real/imag length mismatch");
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        normalize_avx2(real, imag);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                normalize_avx2(real, imag);
                return;
            }
        }
        normalize_scalar(real, imag);
    }
}

/// Vectorized Complex Inner Product <state_a | state_b>.
#[inline(always)]
pub fn inner_product_kernel(
    r1: &[f64],
    i1: &[f64],
    r2: &[f64],
    i2: &[f64],
) -> (f64, f64) {
    let dim = r1.len();
    assert!(
        i1.len() == dim && r2.len() == dim && i2.len() == dim,
        "inner product operands must all have length {} (got {}, {}, {})",
        dim,
        i1.len(),
        r2.len(),
        i2.len()
    );
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        return inner_product_avx2(r1, i1, r2, i2);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return inner_product_avx2(r1, i1, r2, i2);
            }
        }
        inner_product_scalar(r1, i1, r2, i2)
    }
}

/// Vectorized Fused Walsh-Rademacher Phase Layer:
/// Transforms basis amplitudes according to the diagonal unitary:
/// U_Rad |x> = C * (-1)^popcount(x) |x>
/// where C = (-i)^n * prod_q s_q in {+1, -1, +i, -i}.
/// Executes via a single linear memory pass using 256-bit AVX2 XOR sign-bit inversions.
#[inline(always)]
pub fn apply_rademacher_phase_kernel(real: &mut [f64], imag: &mut [f64], signs: &[f64]) {
    let n_qubits = assert_statevector(real, imag);
    assert_eq!(
        signs.len(),
        n_qubits,
        "Rademacher layer needs one sign per qubit ({} signs for {} qubits)",
        signs.len(),
        n_qubits
    );
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        apply_rademacher_phase_avx2(real, imag, signs);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                apply_rademacher_phase_avx2(real, imag, signs);
                return;
            }
        }
        apply_rademacher_phase_scalar(real, imag, signs);
    }
}

/// Vectorized Dual Pauli-Z and Pauli-X Read-Only Reduction Kernel:
/// Simultaneously accumulates <Z_q> and <X_q> for each qubit q across the statevector.
/// Eliminates all writeback traffic and cache line dirtying.
#[inline(always)]
pub fn measure_dual_zx_expectations(
    real: &[f64],
    imag: &[f64],
    z_out: &mut [f64],
    x_out: &mut [f64],
) {
    let n_qubits = assert_statevector(real, imag);
    let n = z_out.len().min(x_out.len());
    assert!(
        n <= n_qubits,
        "requested {} qubit expectations from a {}-qubit state",
        n,
        n_qubits
    );
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    unsafe {
        measure_dual_zx_avx2(real, imag, &mut z_out[..n], &mut x_out[..n]);
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                measure_dual_zx_avx2(real, imag, &mut z_out[..n], &mut x_out[..n]);
                return;
            }
        }
        measure_dual_zx_scalar(real, imag, &mut z_out[..n], &mut x_out[..n]);
    }
}

/// Zero-Allocation Fused Terminal Ry + Pauli-Z Observable Readout Kernel:
/// Evaluates:
///   <Z_q>_final = cos(θ_q) <Z_q>_L2 - sin(θ_q) <X_q>_L2
/// directly from the unmutated Layer 2 statevector |ψ_L2> via read-only AVX2 reduction.
/// Never mutates the 64 KB statevector in Layer 3.
#[inline(always)]
pub fn measure_fused_ry_z_readout(
    real: &[f64],
    imag: &[f64],
    theta_y: &[f64],
    z_final: &mut [f64],
) {
    let n = theta_y.len().min(z_final.len());
    assert_fused_readout_width(n);
    let mut z_l2 = [0.0; 16];
    let mut x_l2 = [0.0; 16];
    measure_dual_zx_expectations(real, imag, &mut z_l2[..n], &mut x_l2[..n]);
    for q in 0..n {
        let (s, c) = theta_y[q].sin_cos();
        z_final[q] = c * z_l2[q] - s * x_l2[q];
    }
}

/// Read-Only Single-Qubit Pauli-Z Expectation Kernel:
/// Computes ⟨Z_q⟩ = Σ_k |ψ_k|² · (-1)^(bit_q(k)) directly from statevector in a single pass.
#[inline(always)]
pub fn measure_single_qubit_z(real: &[f64], imag: &[f64], qubit: usize) -> f64 {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit(qubit, n_qubits);
    let mask = 1usize << qubit;
    let mut diff = 0.0;
    for idx in 0..real.len() {
        let prob = real[idx] * real[idx] + imag[idx] * imag[idx];
        if (idx & mask) == 0 {
            diff += prob;
        } else {
            diff -= prob;
        }
    }
    diff
}

/// Read-Only 2-Body Ising Correlator Measurement Kernel:
/// Computes all C(n,2) two-body correlators ⟨Z_i Z_j⟩ in a single linear pass over the
/// statevector. For 12 qubits this produces 66 correlator values.
///
/// ⟨Z_i Z_j⟩ = Σ_k |α_k|² · (-1)^(bit_i(k) ⊕ bit_j(k))
///            = Σ_k |α_k|² · [1 if bit_i == bit_j, -1 otherwise]
///
/// Output layout: zz_out[pair_idx] where pair_idx = i * (i-1)/2 + j for i > j.
/// Total pairs = n * (n-1) / 2.
#[inline(always)]
pub fn measure_zz_correlators(
    real: &[f64],
    imag: &[f64],
    n_qubits: usize,
    zz_out: &mut [f64],
) {
    let state_qubits = assert_statevector(real, imag);
    assert!(
        n_qubits <= state_qubits,
        "requested correlators over {} qubits from a {}-qubit state",
        n_qubits,
        state_qubits
    );
    let dim = real.len();
    let n_pairs = n_qubits * n_qubits.saturating_sub(1) / 2;
    assert!(zz_out.len() >= n_pairs, "zz_out must have room for {} pairs", n_pairs);

    // Zero output
    for v in zz_out[..n_pairs].iter_mut() {
        *v = 0.0;
    }

    for idx in 0..dim {
        let prob = real[idx] * real[idx] + imag[idx] * imag[idx];
        if prob < 1e-30 { continue; }

        let mut pair_idx = 0;
        for i in 1..n_qubits {
            let bit_i = (idx >> i) & 1;
            for j in 0..i {
                let bit_j = (idx >> j) & 1;
                // (-1)^(bit_i ⊕ bit_j): same parity → +1, different → -1
                let sign = if bit_i == bit_j { 1.0 } else { -1.0 };
                zz_out[pair_idx] += sign * prob;
                pair_idx += 1;
            }
        }
    }
}

/// Read-Only 2-Body Cross-Observable Measurement Kernel:
/// Computes the four 2-qubit Pauli correlators needed for fused Ry-rotated ZZ readout:
///   ⟨Z_i Z_j⟩, ⟨Z_i X_j⟩, ⟨X_i Z_j⟩, ⟨X_i X_j⟩
/// for a single pair (qi, qj) from the L2 statevector.
///
/// These are used by the identity:
///   ⟨Z_i Z_j⟩_fused = c_i·c_j·⟨Z_i Z_j⟩ - c_i·s_j·⟨Z_i X_j⟩
///                     - s_i·c_j·⟨X_i Z_j⟩ + s_i·s_j·⟨X_i X_j⟩
/// where c_q = cos(θ_q), s_q = sin(θ_q).
#[inline(always)]
pub fn measure_2qubit_pauli_correlators(
    real: &[f64],
    imag: &[f64],
    qi: usize,
    qj: usize,
) -> (f64, f64, f64, f64) {
    let n_qubits = assert_statevector(real, imag);
    assert_qubit_pair(qi, qj, n_qubits);
    let dim = real.len();
    let mask_i = 1usize << qi;
    let mask_j = 1usize << qj;

    let mut zz = 0.0;
    let mut zx = 0.0;
    let mut xz = 0.0;
    let mut xx = 0.0;

    for idx in 0..dim {
        let prob = real[idx] * real[idx] + imag[idx] * imag[idx];

        let bi = (idx & mask_i) != 0;
        let bj = (idx & mask_j) != 0;
        let sign_zi: f64 = if bi { -1.0 } else { 1.0 };
        let sign_zj: f64 = if bj { -1.0 } else { 1.0 };

        // ⟨Z_i Z_j⟩ contribution
        zz += sign_zi * sign_zj * prob;

        // ⟨Z_i X_j⟩: Z on qubit i, X flips qubit j
        let partner_j = idx ^ mask_j;
        if idx < partner_j {
            let cross_j = real[idx] * real[partner_j] + imag[idx] * imag[partner_j];
            zx += 2.0 * sign_zi * cross_j;
            // Partner has opposite Z_i sign only if flipping j changes i's bit (impossible since j≠i)
            // So partner's Z_i sign is: sign of bit_i at (idx ^ mask_j) = same as at idx
            // Already handled by using sign_zi for both
        }

        // ⟨X_i Z_j⟩: X flips qubit i, Z on qubit j
        let partner_i = idx ^ mask_i;
        if idx < partner_i {
            let cross_i = real[idx] * real[partner_i] + imag[idx] * imag[partner_i];
            xz += 2.0 * sign_zj * cross_i;
        }

        // ⟨X_i X_j⟩: X flips both qubits i and j
        let partner_ij = idx ^ mask_i ^ mask_j;
        if idx < partner_ij {
            let cross_ij = real[idx] * real[partner_ij] + imag[idx] * imag[partner_ij];
            // Sign is always +1 for XX (no diagonal operator)
            xx += 2.0 * cross_ij;
        }
    }

    (zz, zx, xz, xx)
}

/// Zero-Allocation Fused Terminal Ry + Full 2-Body Correlator Readout Kernel:
/// Computes both the 12 single-qubit ⟨Z_q⟩ AND all 66 two-body ⟨Z_i Z_j⟩
/// expectations after terminal Ry rotation, entirely from the unmutated L2 statevector.
///
/// For single-qubit:
///   ⟨Z_q⟩_final = cos(θ_q) ⟨Z_q⟩_L2 - sin(θ_q) ⟨X_q⟩_L2
///
/// For two-body:
///   ⟨Z_i Z_j⟩_final = c_i·c_j·⟨Z_i Z_j⟩ - c_i·s_j·⟨Z_i X_j⟩
///                     - s_i·c_j·⟨X_i Z_j⟩ + s_i·s_j·⟨X_i X_j⟩
///
/// Output layout: z_final[0..n] = single-qubit, zz_final[0..n*(n-1)/2] = two-body pairs.
/// Pair indexing: pair_idx(i,j) = i*(i-1)/2 + j for i > j.
#[inline(always)]
pub fn measure_fused_ry_full_readout(
    real: &[f64],
    imag: &[f64],
    theta_y: &[f64],
    z_final: &mut [f64],
    zz_final: &mut [f64],
) {
    let n = theta_y.len().min(z_final.len());
    assert_fused_readout_width(n);
    let n_pairs = n * n.saturating_sub(1) / 2;
    assert!(zz_final.len() >= n_pairs);

    // Step 1: Single-qubit fused readout (existing fast path)
    let mut z_l2 = [0.0; 16];
    let mut x_l2 = [0.0; 16];
    measure_dual_zx_expectations(real, imag, &mut z_l2[..n], &mut x_l2[..n]);

    // Precompute sin/cos for terminal Ry layer
    let mut sin_t = [0.0; 16];
    let mut cos_t = [0.0; 16];
    for q in 0..n {
        let (s, c) = theta_y[q].sin_cos();
        sin_t[q] = s;
        cos_t[q] = c;
        z_final[q] = c * z_l2[q] - s * x_l2[q];
    }

    // Step 2: Two-body fused readout via per-pair 4-correlator measurement
    let mut pair_idx = 0;
    for i in 1..n {
        for j in 0..i {
            let (zz, zx, xz, xx) = measure_2qubit_pauli_correlators(real, imag, i, j);
            zz_final[pair_idx] = cos_t[i] * cos_t[j] * zz
                               - cos_t[i] * sin_t[j] * zx
                               - sin_t[i] * cos_t[j] * xz
                               + sin_t[i] * sin_t[j] * xx;
            pair_idx += 1;
        }
    }
}

// =========================================================================
// AVX2 + FMA INTRINSIC IMPLEMENTATIONS
// =========================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_ry_avx2(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let dim = real.len();
    let (s, c) = (theta * 0.5).sin_cos();

    // For qubit >= 2, blocks of size 2^qubit >= 4 align with 256-bit AVX2 vectors
    if qubit >= 2 {
        let block_size = 1 << qubit;
        let stride = block_size << 1;
        let c_vec = _mm256_set1_pd(c);
        let s_vec = _mm256_set1_pd(s);
        let neg_s_vec = _mm256_set1_pd(-s);

        let use_stream = dim >= (1 << 22)
            && ((real.as_ptr() as usize) & 31 == 0)
            && ((imag.as_ptr() as usize) & 31 == 0);

        let mut base = 0;
        while base < dim {
            let p_base = base + block_size;
            let mut k = 0;
            while k < block_size {
                let idx0 = base + k;
                let idx1 = p_base + k;

                let r0 = _mm256_loadu_pd(real.as_ptr().add(idx0));
                let r1 = _mm256_loadu_pd(real.as_ptr().add(idx1));
                let i0 = _mm256_loadu_pd(imag.as_ptr().add(idx0));
                let i1 = _mm256_loadu_pd(imag.as_ptr().add(idx1));

                // new_r0 = c * r0 - s * r1
                let new_r0 = _mm256_fmadd_pd(r0, c_vec, _mm256_mul_pd(r1, neg_s_vec));
                // new_i0 = c * i0 - s * i1
                let new_i0 = _mm256_fmadd_pd(i0, c_vec, _mm256_mul_pd(i1, neg_s_vec));

                // new_r1 = s * r0 + c * r1
                let new_r1 = _mm256_fmadd_pd(r0, s_vec, _mm256_mul_pd(r1, c_vec));
                // new_i1 = s * i0 + c * i1
                let new_i1 = _mm256_fmadd_pd(i0, s_vec, _mm256_mul_pd(i1, c_vec));

                if use_stream {
                    _mm256_stream_pd(real.as_mut_ptr().add(idx0), new_r0);
                    _mm256_stream_pd(real.as_mut_ptr().add(idx1), new_r1);
                    _mm256_stream_pd(imag.as_mut_ptr().add(idx0), new_i0);
                    _mm256_stream_pd(imag.as_mut_ptr().add(idx1), new_i1);
                } else {
                    _mm256_storeu_pd(real.as_mut_ptr().add(idx0), new_r0);
                    _mm256_storeu_pd(real.as_mut_ptr().add(idx1), new_r1);
                    _mm256_storeu_pd(imag.as_mut_ptr().add(idx0), new_i0);
                    _mm256_storeu_pd(imag.as_mut_ptr().add(idx1), new_i1);
                }

                k += 4;
            }
            base += stride;
        }

        if use_stream {
            _mm_sfence();
        }
    } else {
        // Fallback to scalar for low qubit strides (q=0, q=1)
        apply_ry_scalar(real, imag, qubit, theta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn apply_rz_avx2(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let dim = real.len();
    let (s_pos, c_pos) = (theta * 0.5).sin_cos();
    let (s_neg, c_neg) = (-theta * 0.5).sin_cos();

    if qubit >= 2 {
        let block_size = 1 << qubit;
        let stride = block_size << 1;

        let c_neg_v = _mm256_set1_pd(c_neg);
        let s_neg_v = _mm256_set1_pd(s_neg);
        let c_pos_v = _mm256_set1_pd(c_pos);
        let s_pos_v = _mm256_set1_pd(s_pos);

        let use_stream = dim >= (1 << 22)
            && ((real.as_ptr() as usize) & 31 == 0)
            && ((imag.as_ptr() as usize) & 31 == 0);

        let mut base = 0;
        while base < dim {
            // Block 0: bit q is 0 -> multiply by exp(-i theta/2)
            let mut k = 0;
            while k < block_size {
                let idx = base + k;
                let r = _mm256_loadu_pd(real.as_ptr().add(idx));
                let im = _mm256_loadu_pd(imag.as_ptr().add(idx));

                // new_r = r * c_neg - im * s_neg
                let new_r = _mm256_fmsub_pd(r, c_neg_v, _mm256_mul_pd(im, s_neg_v));
                // new_im = r * s_neg + im * c_neg
                let new_im = _mm256_fmadd_pd(r, s_neg_v, _mm256_mul_pd(im, c_neg_v));

                if use_stream {
                    _mm256_stream_pd(real.as_mut_ptr().add(idx), new_r);
                    _mm256_stream_pd(imag.as_mut_ptr().add(idx), new_im);
                } else {
                    _mm256_storeu_pd(real.as_mut_ptr().add(idx), new_r);
                    _mm256_storeu_pd(imag.as_mut_ptr().add(idx), new_im);
                }
                k += 4;
            }

            // Block 1: bit q is 1 -> multiply by exp(i theta/2)
            let p_base = base + block_size;
            let mut k = 0;
            while k < block_size {
                let idx = p_base + k;
                let r = _mm256_loadu_pd(real.as_ptr().add(idx));
                let im = _mm256_loadu_pd(imag.as_ptr().add(idx));

                // new_r = r * c_pos - im * s_pos
                let new_r = _mm256_fmsub_pd(r, c_pos_v, _mm256_mul_pd(im, s_pos_v));
                // new_im = r * s_pos + im * c_pos
                let new_im = _mm256_fmadd_pd(r, s_pos_v, _mm256_mul_pd(im, c_pos_v));

                if use_stream {
                    _mm256_stream_pd(real.as_mut_ptr().add(idx), new_r);
                    _mm256_stream_pd(imag.as_mut_ptr().add(idx), new_im);
                } else {
                    _mm256_storeu_pd(real.as_mut_ptr().add(idx), new_r);
                    _mm256_storeu_pd(imag.as_mut_ptr().add(idx), new_im);
                }
                k += 4;
            }

            base += stride;
        }

        if use_stream {
            _mm_sfence();
        }
    } else {
        apply_rz_scalar(real, imag, qubit, theta);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_cz_avx2(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let dim = real.len();
    let mask1 = 1 << q1;
    let mask2 = 1 << q2;
    let mask = mask1 | mask2;

    let sign_mask = _mm256_set1_pd(-0.0); // Sign-flip bitmask in IEEE 754

    let use_stream = dim >= (1 << 22)
        && ((real.as_ptr() as usize) & 31 == 0)
        && ((imag.as_ptr() as usize) & 31 == 0);

    let mut i = 0;
    while i < dim {
        // If all 4 indices in chunk have both bits set
        let in_cz = (i & mask) == mask
            && ((i + 1) & mask) == mask
            && ((i + 2) & mask) == mask
            && ((i + 3) & mask) == mask;

        if in_cz {
            let r = _mm256_loadu_pd(real.as_ptr().add(i));
            let im = _mm256_loadu_pd(imag.as_ptr().add(i));
            if use_stream {
                _mm256_stream_pd(real.as_mut_ptr().add(i), _mm256_xor_pd(r, sign_mask));
                _mm256_stream_pd(imag.as_mut_ptr().add(i), _mm256_xor_pd(im, sign_mask));
            } else {
                _mm256_storeu_pd(real.as_mut_ptr().add(i), _mm256_xor_pd(r, sign_mask));
                _mm256_storeu_pd(imag.as_mut_ptr().add(i), _mm256_xor_pd(im, sign_mask));
            }
        } else {
            // Scalar fallback for boundary elements
            for k in 0..4 {
                let idx = i + k;
                if (idx & mask) == mask {
                    real[idx] = -real[idx];
                    imag[idx] = -imag[idx];
                }
            }
        }
        i += 4;
    }

    if use_stream {
        _mm_sfence();
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn normalize_avx2(real: &mut [f64], imag: &mut [f64]) {
    let dim = real.len();
    let mut sum_v0 = _mm256_setzero_pd();
    let mut sum_v1 = _mm256_setzero_pd();

    let mut i = 0;
    while i + 8 <= dim {
        let r0 = _mm256_loadu_pd(real.as_ptr().add(i));
        let i0 = _mm256_loadu_pd(imag.as_ptr().add(i));
        let r1 = _mm256_loadu_pd(real.as_ptr().add(i + 4));
        let i1 = _mm256_loadu_pd(imag.as_ptr().add(i + 4));

        sum_v0 = _mm256_fmadd_pd(r0, r0, sum_v0);
        sum_v0 = _mm256_fmadd_pd(i0, i0, sum_v0);
        sum_v1 = _mm256_fmadd_pd(r1, r1, sum_v1);
        sum_v1 = _mm256_fmadd_pd(i1, i1, sum_v1);

        i += 8;
    }

    let sum_all = _mm256_add_pd(sum_v0, sum_v1);
    let mut buf = [0.0; 4];
    _mm256_storeu_pd(buf.as_mut_ptr(), sum_all);
    let mut norm_sq = buf[0] + buf[1] + buf[2] + buf[3];

    while i < dim {
        norm_sq += real[i] * real[i] + imag[i] * imag[i];
        i += 1;
    }

    let norm = norm_sq.sqrt();
    if norm > 1e-12 {
        let inv_norm = 1.0 / norm;
        let inv_vec = _mm256_set1_pd(inv_norm);

        let mut j = 0;
        while j + 4 <= dim {
            let r = _mm256_loadu_pd(real.as_ptr().add(j));
            let im = _mm256_loadu_pd(imag.as_ptr().add(j));
            _mm256_storeu_pd(real.as_mut_ptr().add(j), _mm256_mul_pd(r, inv_vec));
            _mm256_storeu_pd(imag.as_mut_ptr().add(j), _mm256_mul_pd(im, inv_vec));
            j += 4;
        }

        while j < dim {
            real[j] *= inv_norm;
            imag[j] *= inv_norm;
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn inner_product_avx2(
    r1: &[f64],
    i1: &[f64],
    r2: &[f64],
    i2: &[f64],
) -> (f64, f64) {
    let dim = r1.len();
    let mut re_sum = _mm256_setzero_pd();
    let mut im_sum = _mm256_setzero_pd();

    let mut i = 0;
    while i + 4 <= dim {
        let a_r = _mm256_loadu_pd(r1.as_ptr().add(i));
        let a_i = _mm256_loadu_pd(i1.as_ptr().add(i));
        let b_r = _mm256_loadu_pd(r2.as_ptr().add(i));
        let b_i = _mm256_loadu_pd(i2.as_ptr().add(i));

        // conj(A) * B = (a_r - i a_i) * (b_r + i b_i)
        // Re = a_r * b_r + a_i * b_i
        // Im = a_r * b_i - a_i * b_r
        re_sum = _mm256_fmadd_pd(a_r, b_r, re_sum);
        re_sum = _mm256_fmadd_pd(a_i, b_i, re_sum);

        im_sum = _mm256_fmadd_pd(a_r, b_i, im_sum);
        im_sum = _mm256_fnmadd_pd(a_i, b_r, im_sum);

        i += 4;
    }

    let mut buf_re = [0.0; 4];
    let mut buf_im = [0.0; 4];
    _mm256_storeu_pd(buf_re.as_mut_ptr(), re_sum);
    _mm256_storeu_pd(buf_im.as_mut_ptr(), im_sum);

    let mut final_re = buf_re[0] + buf_re[1] + buf_re[2] + buf_re[3];
    let mut final_im = buf_im[0] + buf_im[1] + buf_im[2] + buf_im[3];

    while i < dim {
        final_re += r1[i] * r2[i] + i1[i] * i2[i];
        final_im += r1[i] * i2[i] - i1[i] * r2[i];
        i += 1;
    }

    (final_re, final_im)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_rademacher_phase_avx2(real: &mut [f64], imag: &mut [f64], signs: &[f64]) {
    let n = signs.len();
    let dim = real.len();
    assert_eq!(dim, imag.len());

    let mut prod_s = 1.0;
    for &s in signs {
        if s < 0.0 {
            prod_s = -prod_s;
        }
    }
    let rem = n % 4;

    // Mask for [+1, -1, -1, +1] where p=0 (offsets 0, 1, 2, 3)
    let mask_pos = _mm256_set_pd(0.0, -0.0, -0.0, 0.0);
    // Mask for [-1, +1, +1, -1] where p=1 (offsets 0, 1, 2, 3)
    let mask_neg = _mm256_set_pd(-0.0, 0.0, 0.0, -0.0);

    let chunks = dim / 4;
    let use_stream = dim >= (1 << 22)
        && ((real.as_ptr() as usize) & 31 == 0)
        && ((imag.as_ptr() as usize) & 31 == 0);

    if rem == 0 || rem == 2 {
        let c_sign = if rem == 0 { prod_s } else { -prod_s };
        let c_is_neg = c_sign < 0.0;

        for c_idx in 0..chunks {
            let i = c_idx * 4;
            let p = (i.count_ones() & 1) != 0;
            let flip_0 = if c_is_neg { !p } else { p };
            let mask = if !flip_0 { mask_pos } else { mask_neg };

            let r_ptr = real.as_mut_ptr().add(i);
            let im_ptr = imag.as_mut_ptr().add(i);

            let r_vec = _mm256_loadu_pd(r_ptr);
            let im_vec = _mm256_loadu_pd(im_ptr);

            if use_stream {
                _mm256_stream_pd(r_ptr, _mm256_xor_pd(r_vec, mask));
                _mm256_stream_pd(im_ptr, _mm256_xor_pd(im_vec, mask));
            } else {
                _mm256_storeu_pd(r_ptr, _mm256_xor_pd(r_vec, mask));
                _mm256_storeu_pd(im_ptr, _mm256_xor_pd(im_vec, mask));
            }
        }
    } else {
        let c_im = if rem == 1 { -prod_s } else { prod_s };
        let c_is_pos_i = c_im > 0.0;

        for c_idx in 0..chunks {
            let i = c_idx * 4;
            let p = (i.count_ones() & 1) != 0;

            let flip_r = if c_is_pos_i { !p } else { p };
            let flip_im = if c_is_pos_i { p } else { !p };

            let mask_r = if !flip_r { mask_pos } else { mask_neg };
            let mask_im = if !flip_im { mask_pos } else { mask_neg };

            let r_ptr = real.as_mut_ptr().add(i);
            let im_ptr = imag.as_mut_ptr().add(i);

            let r_vec = _mm256_loadu_pd(r_ptr);
            let im_vec = _mm256_loadu_pd(im_ptr);

            if use_stream {
                _mm256_stream_pd(r_ptr, _mm256_xor_pd(im_vec, mask_r));
                _mm256_stream_pd(im_ptr, _mm256_xor_pd(r_vec, mask_im));
            } else {
                _mm256_storeu_pd(r_ptr, _mm256_xor_pd(im_vec, mask_r));
                _mm256_storeu_pd(im_ptr, _mm256_xor_pd(r_vec, mask_im));
            }
        }
    }

    if use_stream {
        _mm_sfence();
    }

    let rem_start = chunks * 4;
    for idx in rem_start..dim {
        let p = (idx.count_ones() & 1) != 0;
        let factor = if p { -1.0 } else { 1.0 };
        let r = real[idx];
        let im = imag[idx];
        if rem == 0 {
            real[idx] = r * prod_s * factor;
            imag[idx] = im * prod_s * factor;
        } else if rem == 2 {
            real[idx] = -r * prod_s * factor;
            imag[idx] = -im * prod_s * factor;
        } else if rem == 1 {
            real[idx] = im * prod_s * factor;
            imag[idx] = -r * prod_s * factor;
        } else {
            real[idx] = -im * prod_s * factor;
            imag[idx] = r * prod_s * factor;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn measure_dual_zx_avx2(real: &[f64], imag: &[f64], z_out: &mut [f64], x_out: &mut [f64]) {
    let dim = real.len();
    let n = z_out.len().min(x_out.len());

    for q in 0..n {
        if q == 0 {
            let mut z_acc0 = 0.0;
            let mut z_acc1 = 0.0;
            let mut x_acc0 = 0.0;
            let mut x_acc1 = 0.0;
            let mut i = 0;
            while i + 4 <= dim {
                let r0 = *real.get_unchecked(i);
                let i0 = *imag.get_unchecked(i);
                let r1 = *real.get_unchecked(i + 1);
                let i1 = *imag.get_unchecked(i + 1);

                let r2 = *real.get_unchecked(i + 2);
                let i2 = *imag.get_unchecked(i + 2);
                let r3 = *real.get_unchecked(i + 3);
                let i3 = *imag.get_unchecked(i + 3);

                let p0 = r0 * r0 + i0 * i0;
                let p1 = r1 * r1 + i1 * i1;
                z_acc0 += p0 - p1;
                x_acc0 += r0 * r1 + i0 * i1;

                let p2 = r2 * r2 + i2 * i2;
                let p3 = r3 * r3 + i3 * i3;
                z_acc1 += p2 - p3;
                x_acc1 += r2 * r3 + i2 * i3;

                i += 4;
            }
            while i + 2 <= dim {
                let r0 = *real.get_unchecked(i);
                let i0 = *imag.get_unchecked(i);
                let r1 = *real.get_unchecked(i + 1);
                let i1 = *imag.get_unchecked(i + 1);
                let p0 = r0 * r0 + i0 * i0;
                let p1 = r1 * r1 + i1 * i1;
                z_acc0 += p0 - p1;
                x_acc0 += r0 * r1 + i0 * i1;
                i += 2;
            }
            z_out[0] = z_acc0 + z_acc1;
            x_out[0] = 2.0 * (x_acc0 + x_acc1);
        } else if q == 1 {
            let mut z_acc0 = 0.0;
            let mut z_acc1 = 0.0;
            let mut x_acc0 = 0.0;
            let mut x_acc1 = 0.0;
            let mut i = 0;
            while i + 4 <= dim {
                let r0 = *real.get_unchecked(i);
                let i0 = *imag.get_unchecked(i);
                let r1 = *real.get_unchecked(i + 1);
                let i1 = *imag.get_unchecked(i + 1);
                let r2 = *real.get_unchecked(i + 2);
                let i2 = *imag.get_unchecked(i + 2);
                let r3 = *real.get_unchecked(i + 3);
                let i3 = *imag.get_unchecked(i + 3);

                let p0 = r0 * r0 + i0 * i0;
                let p1 = r1 * r1 + i1 * i1;
                let p2 = r2 * r2 + i2 * i2;
                let p3 = r3 * r3 + i3 * i3;

                z_acc0 += p0 - p2;
                z_acc1 += p1 - p3;

                x_acc0 += r0 * r2 + i0 * i2;
                x_acc1 += r1 * r3 + i1 * i3;

                i += 4;
            }
            z_out[1] = z_acc0 + z_acc1;
            x_out[1] = 2.0 * (x_acc0 + x_acc1);
        } else {
            let block_size = 1 << q;
            let stride = block_size << 1;

            let mut z_vec0 = _mm256_setzero_pd();
            let mut z_vec1 = _mm256_setzero_pd();
            let mut x_vec0 = _mm256_setzero_pd();
            let mut x_vec1 = _mm256_setzero_pd();

            let mut base = 0;
            while base < dim {
                let p_base = base + block_size;
                let mut k = 0;
                while k + 8 <= block_size {
                    let idx0 = base + k;
                    let idx1 = p_base + k;

                    let r0_0 = _mm256_loadu_pd(real.as_ptr().add(idx0));
                    let i0_0 = _mm256_loadu_pd(imag.as_ptr().add(idx0));
                    let r1_0 = _mm256_loadu_pd(real.as_ptr().add(idx1));
                    let i1_0 = _mm256_loadu_pd(imag.as_ptr().add(idx1));

                    let r0_1 = _mm256_loadu_pd(real.as_ptr().add(idx0 + 4));
                    let i0_1 = _mm256_loadu_pd(imag.as_ptr().add(idx0 + 4));
                    let r1_1 = _mm256_loadu_pd(real.as_ptr().add(idx1 + 4));
                    let i1_1 = _mm256_loadu_pd(imag.as_ptr().add(idx1 + 4));

                    // Z accumulation via 4 chained single-cycle FMAs
                    let s0_0 = _mm256_fmadd_pd(i0_0, i0_0, z_vec0);
                    let s0_1 = _mm256_fmadd_pd(r0_0, r0_0, s0_0);
                    let s0_2 = _mm256_fnmadd_pd(i1_0, i1_0, s0_1);
                    z_vec0 = _mm256_fnmadd_pd(r1_0, r1_0, s0_2);

                    let s1_0 = _mm256_fmadd_pd(i0_1, i0_1, z_vec1);
                    let s1_1 = _mm256_fmadd_pd(r0_1, r0_1, s1_0);
                    let s1_2 = _mm256_fnmadd_pd(i1_1, i1_1, s1_1);
                    z_vec1 = _mm256_fnmadd_pd(r1_1, r1_1, s1_2);

                    // X accumulation via 2 chained single-cycle FMAs
                    x_vec0 = _mm256_fmadd_pd(r0_0, r1_0, _mm256_fmadd_pd(i0_0, i1_0, x_vec0));
                    x_vec1 = _mm256_fmadd_pd(r0_1, r1_1, _mm256_fmadd_pd(i0_1, i1_1, x_vec1));

                    k += 8;
                }
                while k < block_size {
                    let idx0 = base + k;
                    let idx1 = p_base + k;

                    let r0 = _mm256_loadu_pd(real.as_ptr().add(idx0));
                    let i0 = _mm256_loadu_pd(imag.as_ptr().add(idx0));
                    let r1 = _mm256_loadu_pd(real.as_ptr().add(idx1));
                    let i1 = _mm256_loadu_pd(imag.as_ptr().add(idx1));

                    let s0 = _mm256_fmadd_pd(i0, i0, z_vec0);
                    let s1 = _mm256_fmadd_pd(r0, r0, s0);
                    let s2 = _mm256_fnmadd_pd(i1, i1, s1);
                    z_vec0 = _mm256_fnmadd_pd(r1, r1, s2);

                    x_vec0 = _mm256_fmadd_pd(r0, r1, _mm256_fmadd_pd(i0, i1, x_vec0));

                    k += 4;
                }
                base += stride;
            }

            let z_sum = _mm256_add_pd(z_vec0, z_vec1);
            let x_sum = _mm256_add_pd(x_vec0, x_vec1);
            let mut z_buf = [0.0; 4];
            let mut x_buf = [0.0; 4];
            _mm256_storeu_pd(z_buf.as_mut_ptr(), z_sum);
            _mm256_storeu_pd(x_buf.as_mut_ptr(), x_sum);

            z_out[q] = z_buf[0] + z_buf[1] + z_buf[2] + z_buf[3];
            x_out[q] = 2.0 * (x_buf[0] + x_buf[1] + x_buf[2] + x_buf[3]);
        }
    }
}

// =========================================================================
// SCALAR FALLBACK IMPLEMENTATIONS
// =========================================================================

#[allow(dead_code)]
fn apply_ry_scalar(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let dim = real.len();
    let mask = 1 << qubit;
    let (s, c) = (theta * 0.5).sin_cos();

    for i in 0..dim {
        if (i & mask) == 0 {
            let partner = i | mask;
            let r0 = real[i];
            let i0 = imag[i];
            let r1 = real[partner];
            let i1 = imag[partner];

            real[i] = c * r0 - s * r1;
            imag[i] = c * i0 - s * i1;
            real[partner] = s * r0 + c * r1;
            imag[partner] = s * i0 + c * i1;
        }
    }
}

#[allow(dead_code)]
fn apply_rz_scalar(real: &mut [f64], imag: &mut [f64], qubit: usize, theta: f64) {
    let dim = real.len();
    let mask = 1 << qubit;
    let (s_pos, c_pos) = (theta * 0.5).sin_cos();
    let (s_neg, c_neg) = (-theta * 0.5).sin_cos();

    for i in 0..dim {
        let r = real[i];
        let im = imag[i];
        if (i & mask) == 0 {
            real[i] = r * c_neg - im * s_neg;
            imag[i] = r * s_neg + im * c_neg;
        } else {
            real[i] = r * c_pos - im * s_pos;
            imag[i] = r * s_pos + im * c_pos;
        }
    }
}

#[allow(dead_code)]
fn apply_cz_scalar(real: &mut [f64], imag: &mut [f64], q1: usize, q2: usize) {
    let dim = real.len();
    let mask1 = 1 << q1;
    let mask2 = 1 << q2;
    for i in 0..dim {
        if (i & mask1) != 0 && (i & mask2) != 0 {
            real[i] = -real[i];
            imag[i] = -imag[i];
        }
    }
}

#[allow(dead_code)]
fn normalize_scalar(real: &mut [f64], imag: &mut [f64]) {
    let dim = real.len();
    let mut norm_sq = 0.0;
    for i in 0..dim {
        norm_sq += real[i] * real[i] + imag[i] * imag[i];
    }
    let norm = norm_sq.sqrt();
    if norm > 1e-12 {
        let inv = 1.0 / norm;
        for i in 0..dim {
            real[i] *= inv;
            imag[i] *= inv;
        }
    }
}

#[allow(dead_code)]
fn inner_product_scalar(
    r1: &[f64],
    i1: &[f64],
    r2: &[f64],
    i2: &[f64],
) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for idx in 0..r1.len() {
        re += r1[idx] * r2[idx] + i1[idx] * i2[idx];
        im += r1[idx] * i2[idx] - i1[idx] * r2[idx];
    }
    (re, im)
}

#[allow(dead_code)]
fn apply_rademacher_phase_scalar(real: &mut [f64], imag: &mut [f64], signs: &[f64]) {
    let n = signs.len();
    let dim = real.len();
    let mut prod_s = 1.0;
    for &s in signs {
        if s < 0.0 {
            prod_s = -prod_s;
        }
    }
    let rem = n % 4;

    for idx in 0..dim {
        let p = (idx.count_ones() & 1) != 0;
        let factor = if p { -1.0 } else { 1.0 };
        let r = real[idx];
        let im = imag[idx];
        match rem {
            0 => {
                real[idx] = r * prod_s * factor;
                imag[idx] = im * prod_s * factor;
            }
            2 => {
                real[idx] = -r * prod_s * factor;
                imag[idx] = -im * prod_s * factor;
            }
            1 => {
                real[idx] = im * prod_s * factor;
                imag[idx] = -r * prod_s * factor;
            }
            3 => {
                real[idx] = -im * prod_s * factor;
                imag[idx] = r * prod_s * factor;
            }
            _ => unreachable!(),
        }
    }
}

#[allow(dead_code)]
fn measure_dual_zx_scalar(real: &[f64], imag: &[f64], z_out: &mut [f64], x_out: &mut [f64]) {
    let dim = real.len();
    let n = z_out.len().min(x_out.len());
    for q in 0..n {
        let mask = 1 << q;
        let mut z_acc = 0.0;
        let mut x_acc = 0.0;
        for i in 0..dim {
            if (i & mask) == 0 {
                let partner = i | mask;
                let r0 = real[i];
                let i0 = imag[i];
                let r1 = real[partner];
                let i1 = imag[partner];

                let p0 = r0 * r0 + i0 * i0;
                let p1 = r1 * r1 + i1 * i1;
                z_acc += p0 - p1;
                x_acc += r0 * r1 + i0 * i1;
            }
        }
        z_out[q] = z_acc;
        x_out[q] = 2.0 * x_acc;
    }
}
