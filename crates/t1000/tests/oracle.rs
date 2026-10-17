//! Reference-oracle tests for the statevector core.
//!
//! Every SIMD / tiled kernel, observable, encoder, ansatz forward path and the circuit
//! compiler is checked amplitude-by-amplitude against a deliberately naive dense
//! simulator defined in this file. The oracle itself is validated against explicit
//! full-matrix (Kronecker-product) evolution at small qubit counts.

use std::collections::BTreeSet;
use std::f64::consts::PI;

use num_complex::Complex64 as C;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use t1000::compiler::Unitary2x2;
use t1000::*;

const TOL: f64 = 1e-10;

// =========================================================================
// Dense reference simulator (qubit 0 = least-significant bit of the index)
// =========================================================================

type Mat2 = [[C; 2]; 2];

fn c(re: f64) -> C {
    C::new(re, 0.0)
}

fn ry_mat(theta: f64) -> Mat2 {
    let (s, co) = (theta / 2.0).sin_cos();
    [[c(co), c(-s)], [c(s), c(co)]]
}

fn rz_mat(theta: f64) -> Mat2 {
    [
        [C::from_polar(1.0, -theta / 2.0), c(0.0)],
        [c(0.0), C::from_polar(1.0, theta / 2.0)],
    ]
}

fn x_mat() -> Mat2 {
    [[c(0.0), c(1.0)], [c(1.0), c(0.0)]]
}

fn z_mat() -> Mat2 {
    [[c(1.0), c(0.0)], [c(0.0), c(-1.0)]]
}

fn bit(i: usize, q: usize) -> usize {
    (i >> q) & 1
}

/// Applies a single-qubit gate by pairing amplitudes that differ only in bit `q`.
fn ref_1q(psi: &mut [C], q: usize, m: &Mat2) {
    for i in 0..psi.len() {
        if bit(i, q) == 0 {
            let j = i | (1 << q);
            let (a, b) = (psi[i], psi[j]);
            psi[i] = m[0][0] * a + m[0][1] * b;
            psi[j] = m[1][0] * a + m[1][1] * b;
        }
    }
}

fn ref_cz(psi: &mut [C], q1: usize, q2: usize) {
    assert_ne!(q1, q2);
    for (i, a) in psi.iter_mut().enumerate() {
        if bit(i, q1) == 1 && bit(i, q2) == 1 {
            *a = -*a;
        }
    }
}

fn ref_cnot(psi: &mut [C], control: usize, target: usize) {
    assert_ne!(control, target);
    let old = psi.to_vec();
    for i in 0..psi.len() {
        let src = if bit(i, control) == 1 {
            i ^ (1 << target)
        } else {
            i
        };
        psi[i] = old[src];
    }
}

/// <a|b>
fn ref_inner(a: &[C], b: &[C]) -> C {
    a.iter().zip(b).map(|(x, y)| x.conj() * y).sum()
}

/// <psi| P |psi> for a tensor product of single-qubit operators `ops = [(qubit, matrix)]`.
fn ref_expect(psi: &[C], ops: &[(usize, Mat2)]) -> f64 {
    let mut phi = psi.to_vec();
    for (q, m) in ops {
        ref_1q(&mut phi, *q, m);
    }
    let e = ref_inner(psi, &phi);
    assert!(
        e.im.abs() < 1e-9,
        "Hermitian expectation has imaginary part {}",
        e.im
    );
    e.re
}

fn ref_z(psi: &[C], q: usize) -> f64 {
    ref_expect(psi, &[(q, z_mat())])
}

fn ref_zz(psi: &[C], i: usize, j: usize) -> f64 {
    ref_expect(psi, &[(i, z_mat()), (j, z_mat())])
}

/// Intended entangling pattern of `CircuitAnsatz` layer `layer`: CZ on each distinct
/// unordered pair {q, (q + 2^(layer mod 4)) mod n}, applied exactly once.
fn ref_cz_pairs(n: usize, layer: usize) -> BTreeSet<(usize, usize)> {
    let stride = 1usize << (layer % 4);
    (0..n)
        .map(|q| (q, (q + stride) % n))
        .filter(|(q, t)| q != t)
        .map(|(q, t)| (q.min(t), q.max(t)))
        .collect()
}

fn ref_pi_layer(psi: &mut [C], n: usize) {
    for q in 0..n {
        let sign = if q % 2 == 0 { 1.0 } else { -1.0 };
        ref_1q(psi, q, &rz_mat(sign * PI));
    }
}

/// One full ansatz layer: Ry, Rz on every qubit, then the CZ pattern.
fn ref_layer(psi: &mut [C], n: usize, layer: usize, weights: &[f64]) {
    let base = layer * 2 * n;
    for q in 0..n {
        ref_1q(psi, q, &ry_mat(weights[base + 2 * q]));
        ref_1q(psi, q, &rz_mat(weights[base + 2 * q + 1]));
    }
    for (a, b) in ref_cz_pairs(n, layer) {
        ref_cz(psi, a, b);
    }
}

// ----- explicit full-matrix evolution used to validate the oracle itself -----

fn full_matrix_1q(n: usize, q: usize, m: &Mat2) -> Vec<Vec<C>> {
    let dim = 1 << n;
    let mut u = vec![vec![c(0.0); dim]; dim];
    for (r, row) in u.iter_mut().enumerate() {
        for (col, entry) in row.iter_mut().enumerate() {
            let mut v = c(1.0);
            for k in 0..n {
                let (rb, cb) = (bit(r, k), bit(col, k));
                v *= if k == q {
                    m[rb][cb]
                } else if rb == cb {
                    c(1.0)
                } else {
                    c(0.0)
                };
            }
            *entry = v;
        }
    }
    u
}

fn matvec(u: &[Vec<C>], psi: &[C]) -> Vec<C> {
    u.iter()
        .map(|row| row.iter().zip(psi).map(|(a, b)| a * b).sum())
        .collect()
}

// ----- helpers -----

fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

fn random_state(n: usize, r: &mut ChaCha8Rng) -> Vec<C> {
    let mut v: Vec<C> = (0..1usize << n)
        .map(|_| C::new(r.gen_range(-1.0..1.0), r.gen_range(-1.0..1.0)))
        .collect();
    let norm = v.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    v.iter_mut().for_each(|a| *a /= norm);
    v
}

fn random_angles(len: usize, r: &mut ChaCha8Rng) -> Vec<f64> {
    (0..len).map(|_| r.gen_range(-PI..PI)).collect()
}

fn to_qs(psi: &[C]) -> QuantumState {
    QuantumState {
        dim: psi.len(),
        real: psi.iter().map(|a| a.re).collect(),
        imag: psi.iter().map(|a| a.im).collect(),
    }
}

fn from_qs(s: &QuantumState) -> Vec<C> {
    s.real
        .iter()
        .zip(&s.imag)
        .map(|(&r, &i)| C::new(r, i))
        .collect()
}

fn assert_states_close(got: &[C], want: &[C], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length mismatch");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).norm() < TOL,
            "{ctx}: amplitude {k} differs: got {g}, want {w}"
        );
    }
}

fn assert_close(got: f64, want: f64, ctx: &str) {
    assert!((got - want).abs() < TOL, "{ctx}: got {got}, want {want}");
}

fn ansatz(n: usize, depth: usize, pi: bool) -> CircuitAnsatz {
    CircuitAnsatz::new(n, depth).with_pi_layer(pi)
}

fn pruned_len(n: usize, depth: usize) -> usize {
    2 * n * (depth - 1) + n
}

// =========================================================================
// 1. Oracle self-check
// =========================================================================

#[test]
fn oracle_single_qubit_gate_matches_full_matrix() {
    let mut r = rng(1);
    for n in 1..=4 {
        for q in 0..n {
            for m in [
                ry_mat(r.gen_range(-PI..PI)),
                rz_mat(r.gen_range(-PI..PI)),
                x_mat(),
            ] {
                let psi = random_state(n, &mut r);
                let mut got = psi.clone();
                ref_1q(&mut got, q, &m);
                let want = matvec(&full_matrix_1q(n, q, &m), &psi);
                assert_states_close(&got, &want, &format!("oracle n={n} q={q}"));
            }
        }
    }
}

#[test]
fn oracle_cz_and_cnot_match_definitions() {
    // CZ = (I ⊗ H) CNOT (I ⊗ H)
    let mut r = rng(2);
    let h = [
        [c(1.0 / 2f64.sqrt()), c(1.0 / 2f64.sqrt())],
        [c(1.0 / 2f64.sqrt()), c(-1.0 / 2f64.sqrt())],
    ];
    for n in 2..=4 {
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let psi = random_state(n, &mut r);
                let mut via_cz = psi.clone();
                ref_cz(&mut via_cz, a, b);
                let mut via_cnot = psi.clone();
                ref_1q(&mut via_cnot, b, &h);
                ref_cnot(&mut via_cnot, a, b);
                ref_1q(&mut via_cnot, b, &h);
                assert_states_close(&via_cz, &via_cnot, &format!("CZ/CNOT n={n} ({a},{b})"));
            }
        }
    }
}

// =========================================================================
// 2. SIMD gate kernels (every qubit index exercises scalar q<2 and AVX2 q>=2 paths)
// =========================================================================

#[test]
fn simd_single_qubit_kernels_match_oracle() {
    let mut r = rng(10);
    for n in 1..=10 {
        for q in 0..n {
            let theta = r.gen_range(-PI..PI);
            let psi = random_state(n, &mut r);

            let mut s = to_qs(&psi);
            simd::apply_ry_kernel(&mut s.real, &mut s.imag, q, theta);
            let mut want = psi.clone();
            ref_1q(&mut want, q, &ry_mat(theta));
            assert_states_close(&from_qs(&s), &want, &format!("Ry n={n} q={q}"));

            let mut s = to_qs(&psi);
            simd::apply_rz_kernel(&mut s.real, &mut s.imag, q, theta);
            let mut want = psi.clone();
            ref_1q(&mut want, q, &rz_mat(theta));
            assert_states_close(&from_qs(&s), &want, &format!("Rz n={n} q={q}"));
        }
    }
}

#[test]
fn simd_cz_kernel_matches_oracle() {
    let mut r = rng(11);
    for n in 2..=8 {
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let psi = random_state(n, &mut r);
                let mut s = to_qs(&psi);
                simd::apply_cz_kernel(&mut s.real, &mut s.imag, a, b);
                let mut want = psi.clone();
                ref_cz(&mut want, a, b);
                assert_states_close(&from_qs(&s), &want, &format!("CZ n={n} ({a},{b})"));
            }
        }
    }
}

#[test]
fn rademacher_phase_kernel_matches_oracle_for_all_phase_cases() {
    // n mod 4 selects the global phase branch (1, -i, -1, +i); random signs flip it further.
    let mut r = rng(12);
    for n in 1..=9 {
        for _ in 0..3 {
            let signs: Vec<f64> = (0..n)
                .map(|_| if r.gen_bool(0.5) { 1.0 } else { -1.0 })
                .collect();
            let psi = random_state(n, &mut r);
            let mut s = to_qs(&psi);
            simd::apply_rademacher_phase_kernel(&mut s.real, &mut s.imag, &signs);
            let mut want = psi.clone();
            for (q, sign) in signs.iter().enumerate() {
                ref_1q(&mut want, q, &rz_mat(sign * PI));
            }
            assert_states_close(
                &from_qs(&s),
                &want,
                &format!("Rademacher n={n} signs={signs:?}"),
            );
        }
    }
}

#[test]
fn normalize_inner_product_fidelity_match_oracle() {
    let mut r = rng(13);
    for n in 1..=8 {
        let a = random_state(n, &mut r);
        let b = random_state(n, &mut r);
        let (sa, sb) = (to_qs(&a), to_qs(&b));
        let ip = ref_inner(&a, &b);
        let got = sa.inner_product(&sb);
        assert!(
            (got - ip).norm() < TOL,
            "inner product n={n}: {got} vs {ip}"
        );
        assert_close(sa.fidelity(&sb), ip.norm_sqr(), &format!("fidelity n={n}"));

        let scaled: Vec<C> = a.iter().map(|x| x * 3.7).collect();
        let mut s = to_qs(&scaled);
        s.normalize();
        assert_states_close(&from_qs(&s), &a, &format!("normalize n={n}"));
    }
}

// =========================================================================
// 3. Tiled high-qubit kernels (n = 17 crosses TILED_LAYER_THRESHOLD and the
//    interleaved q >= 16 path)
// =========================================================================

#[test]
fn tiled_kernels_match_oracle_on_17_qubits() {
    let n = 17;
    let mut r = rng(20);
    let psi = random_state(n, &mut r);
    for q in [0, 1, 8, 9, 15, 16] {
        let theta = r.gen_range(-PI..PI);

        let mut s = to_qs(&psi);
        tiled_apply_ry(&mut s.real, &mut s.imag, q, theta);
        let mut want = psi.clone();
        ref_1q(&mut want, q, &ry_mat(theta));
        assert_states_close(&from_qs(&s), &want, &format!("tiled Ry q={q}"));

        let mut s = to_qs(&psi);
        tiled_apply_rz(&mut s.real, &mut s.imag, q, theta);
        let mut want = psi.clone();
        ref_1q(&mut want, q, &rz_mat(theta));
        assert_states_close(&from_qs(&s), &want, &format!("tiled Rz q={q}"));
    }
    for (a, b) in [(0, 16), (16, 3), (15, 16), (9, 12)] {
        let mut s = to_qs(&psi);
        tiled_apply_cz(&mut s.real, &mut s.imag, a, b);
        let mut want = psi.clone();
        ref_cz(&mut want, a, b);
        assert_states_close(&from_qs(&s), &want, &format!("tiled CZ ({a},{b})"));
    }
}

#[test]
fn parallel_ry_matches_oracle_on_every_qubit() {
    // n >= 14 crosses PARALLEL_MIN_DIM; low qubits use the chunked path, high qubits the
    // paired-halves path (when more than one thread is available).
    let mut r = rng(23);
    for n in [15, 17] {
        let psi = random_state(n, &mut r);
        for q in 0..n {
            let theta = r.gen_range(-PI..PI);
            let mut s = to_qs(&psi);
            t1000::parallel::apply_ry(&mut s.real, &mut s.imag, q, theta);
            let mut want = psi.clone();
            ref_1q(&mut want, q, &ry_mat(theta));
            assert_states_close(&from_qs(&s), &want, &format!("parallel Ry n={n} q={q}"));
        }
    }
}

#[test]
fn fused_diagonal_layer_matches_oracle() {
    let mut r = rng(22);
    for n in (1..=11).chain([17]) {
        for trial in 0..4 {
            // Fewer angles than qubits on some trials; untouched qubits must be left alone.
            let n_angles = if trial % 2 == 0 {
                n
            } else {
                r.gen_range(0..=n)
            };
            let rz = random_angles(n_angles, &mut r);
            // Random pairs, possibly repeated (a repeated CZ pair must cancel).
            let n_pairs = if n < 2 { 0 } else { r.gen_range(0..=2 * n) };
            let pairs: Vec<(usize, usize)> = (0..n_pairs)
                .map(|_| {
                    let a = r.gen_range(0..n);
                    let b = (a + r.gen_range(1..n)) % n;
                    (a, b)
                })
                .collect();
            let psi = random_state(n, &mut r);
            let mut s = to_qs(&psi);
            t1000::diagonal::apply_rz_cz_layer(&mut s.real, &mut s.imag, &rz, &pairs);
            let mut want = psi.clone();
            for (q, &t) in rz.iter().enumerate() {
                ref_1q(&mut want, q, &rz_mat(t));
            }
            for &(a, b) in &pairs {
                ref_cz(&mut want, a, b);
            }
            assert_states_close(
                &from_qs(&s),
                &want,
                &format!("diagonal layer n={n} trial={trial}"),
            );
        }
    }
}

#[test]
fn apply_layer_fused_matches_oracle() {
    let mut r = rng(21);
    for n in [1, 4, 9, 10, 17] {
        let psi = random_state(n, &mut r);
        let ry = random_angles(n, &mut r);
        let rz = random_angles(n, &mut r);
        let mut s = to_qs(&psi);
        apply_layer_fused(&mut s.real, &mut s.imag, &ry, &rz);
        let mut want = psi.clone();
        for q in 0..n {
            ref_1q(&mut want, q, &ry_mat(ry[q]));
            ref_1q(&mut want, q, &rz_mat(rz[q]));
        }
        assert_states_close(&from_qs(&s), &want, &format!("fused layer n={n}"));
    }
}

// =========================================================================
// 4. Measurement kernels and observables
// =========================================================================

#[test]
fn measurement_kernels_match_oracle() {
    let mut r = rng(30);
    for n in 1..=8 {
        let psi = random_state(n, &mut r);
        let s = to_qs(&psi);

        let mut z = vec![0.0; n];
        let mut x = vec![0.0; n];
        simd::measure_dual_zx_expectations(&s.real, &s.imag, &mut z, &mut x);
        for q in 0..n {
            assert_close(z[q], ref_z(&psi, q), &format!("dual Z n={n} q={q}"));
            assert_close(
                x[q],
                ref_expect(&psi, &[(q, x_mat())]),
                &format!("dual X n={n} q={q}"),
            );
            assert_close(
                simd::measure_single_qubit_z(&s.real, &s.imag, q),
                ref_z(&psi, q),
                &format!("single Z n={n} q={q}"),
            );
        }

        let n_pairs = n * (n - 1) / 2;
        let mut zz = vec![0.0; n_pairs];
        simd::measure_zz_correlators(&s.real, &s.imag, n, &mut zz);
        let mut idx = 0;
        for i in 1..n {
            for j in 0..i {
                assert_close(zz[idx], ref_zz(&psi, i, j), &format!("ZZ n={n} ({i},{j})"));
                let (pzz, pzx, pxz, pxx) =
                    simd::measure_2qubit_pauli_correlators(&s.real, &s.imag, i, j);
                assert_close(pzz, ref_zz(&psi, i, j), "pair ZZ");
                assert_close(
                    pzx,
                    ref_expect(&psi, &[(i, z_mat()), (j, x_mat())]),
                    "pair ZX",
                );
                assert_close(
                    pxz,
                    ref_expect(&psi, &[(i, x_mat()), (j, z_mat())]),
                    "pair XZ",
                );
                assert_close(
                    pxx,
                    ref_expect(&psi, &[(i, x_mat()), (j, x_mat())]),
                    "pair XX",
                );
                idx += 1;
            }
        }
    }
}

#[test]
fn fused_terminal_ry_readout_matches_oracle() {
    let mut r = rng(31);
    for n in 1..=8 {
        let psi = random_state(n, &mut r);
        let s = to_qs(&psi);
        let theta = random_angles(n, &mut r);
        let mut rotated = psi.clone();
        for q in 0..n {
            ref_1q(&mut rotated, q, &ry_mat(theta[q]));
        }

        let mut z = vec![0.0; n];
        let mut zz = vec![0.0; n * (n - 1) / 2];
        simd::measure_fused_ry_full_readout(&s.real, &s.imag, &theta, &mut z, &mut zz);
        let mut z_only = vec![0.0; n];
        simd::measure_fused_ry_z_readout(&s.real, &s.imag, &theta, &mut z_only);

        let mut idx = 0;
        for i in 0..n {
            assert_close(z[i], ref_z(&rotated, i), &format!("fused Z n={n} q={i}"));
            assert_close(
                z_only[i],
                ref_z(&rotated, i),
                &format!("fused Z-only n={n} q={i}"),
            );
            for j in 0..i {
                assert_close(
                    zz[idx],
                    ref_zz(&rotated, i, j),
                    &format!("fused ZZ n={n} ({i},{j})"),
                );
                idx += 1;
            }
        }
    }
}

#[test]
fn pauli_observables_match_oracle() {
    let mut r = rng(32);
    for n in 1..=7 {
        let psi = random_state(n, &mut r);
        let s = to_qs(&psi);
        for q in 0..n {
            assert_close(
                PauliObservable::Z(q).expectation_value(&s),
                ref_z(&psi, q),
                "obs Z",
            );
            assert_close(
                PauliObservable::X(q).expectation_value(&s),
                ref_expect(&psi, &[(q, x_mat())]),
                "obs X",
            );
            for j in 0..n {
                if j != q {
                    assert_close(
                        PauliObservable::ZZ(q, j).expectation_value(&s),
                        ref_zz(&psi, q, j),
                        "obs ZZ",
                    );
                }
            }
        }
        let weights: Vec<(usize, f64)> = (0..n).map(|q| (q, r.gen_range(-2.0..2.0))).collect();
        let want_lin: f64 = weights.iter().map(|&(q, w)| w * ref_z(&psi, q)).sum();
        assert_close(
            PauliObservable::LinearZ(weights).expectation_value(&s),
            want_lin,
            "obs LinearZ",
        );
        let want_mean = (0..n).map(|q| ref_z(&psi, q)).sum::<f64>() / n as f64;
        assert_close(
            PauliObservable::GlobalMeanZ.expectation_value(&s),
            want_mean,
            "obs GlobalMeanZ",
        );
    }
}

// =========================================================================
// 5. Encoders
// =========================================================================

#[test]
fn angle_and_amplitude_encoding_match_oracle() {
    let mut r = rng(40);
    for n in 1usize..=7 {
        // Fewer, equal and more features than qubits.
        for n_feat in [n.saturating_sub(1), n, n + 3] {
            let x: Vec<f64> = (0..n_feat).map(|_| r.gen_range(-3.0..3.0)).collect();

            let got = DataEncodingLayer::new(n, EncodingKind::Angle).encode(&x);
            let mut want = vec![c(0.0); 1 << n];
            want[0] = c(1.0);
            for q in 0..n {
                ref_1q(&mut want, q, &ry_mat(x.get(q).copied().unwrap_or(0.0)));
            }
            assert_states_close(
                &from_qs(&got),
                &want,
                &format!("angle n={n} feats={n_feat}"),
            );

            if n_feat == 0 {
                continue;
            }
            let enc = DataEncodingLayer::new(n, EncodingKind::Amplitude);
            let kept: Vec<f64> = x.iter().copied().take(1 << n).collect();
            let norm = kept.iter().map(|v| v * v).sum::<f64>().sqrt();
            let mut want = vec![c(0.0); 1 << n];
            for (k, v) in kept.iter().enumerate() {
                want[k] = c(v / norm);
            }
            assert_states_close(
                &from_qs(&enc.encode(&x)),
                &want,
                &format!("amplitude n={n} feats={n_feat}"),
            );
            let mut into = QuantumState::zero(1 << n);
            enc.encode_amplitude_into(&x, &mut into);
            assert_states_close(
                &from_qs(&into),
                &want,
                &format!("amplitude_into n={n} feats={n_feat}"),
            );
        }
    }
}

// =========================================================================
// 6. CircuitAnsatz forward paths
// =========================================================================

/// pi layer (optional) followed by full layers `0..layers`.
fn ref_forward_layers(
    psi: &[C],
    n: usize,
    layers: std::ops::Range<usize>,
    w: &[f64],
    pi: bool,
) -> Vec<C> {
    let mut s = psi.to_vec();
    if pi && layers.start == 0 {
        ref_pi_layer(&mut s, n);
    }
    for l in layers {
        ref_layer(&mut s, n, l, w);
    }
    s
}

#[test]
fn ansatz_entangling_pattern_applies_each_pair_once() {
    for n in 1..=20 {
        let depth = 4;
        let a = CircuitAnsatz::new(n, depth);
        let dag = a.to_dag(&a.weights);
        // Each layer opens with 2n rotations; CZs belong to the layer of the preceding rotations.
        let mut pairs = vec![BTreeSet::new(); depth];
        let mut rotations = 0usize;
        for node in &dag.nodes {
            match node.op {
                CircuitOp::Ry { .. } | CircuitOp::Rz { .. } => rotations += 1,
                CircuitOp::CZ { q1, q2 } => {
                    let layer = (rotations - 1) / (2 * n);
                    let p = (q1.min(q2), q1.max(q2));
                    assert!(
                        pairs[layer].insert(p),
                        "n={n} layer={layer}: CZ pair {p:?} applied twice (cancels)"
                    );
                }
                _ => {}
            }
        }
        for (layer, got) in pairs.iter().enumerate() {
            assert_eq!(
                *got,
                ref_cz_pairs(n, layer),
                "n={n} layer={layer}: wrong CZ pattern"
            );
        }
    }
}

#[test]
fn ansatz_full_forward_matches_oracle() {
    let mut r = rng(50);
    for n in 1..=8 {
        for depth in 1..=5 {
            for pi in [false, true] {
                let a = ansatz(n, depth, pi);
                let w = random_angles(a.param_count(), &mut r);
                let psi = random_state(n, &mut r);
                let want = ref_forward_layers(&psi, n, 0..depth, &w, pi);
                let ctx = format!("forward n={n} depth={depth} pi={pi}");

                assert_states_close(&from_qs(&a.forward(&to_qs(&psi), &w)), &want, &ctx);
                let mut s = to_qs(&psi);
                a.forward_in_place(&mut s, &w);
                assert_states_close(&from_qs(&s), &want, &format!("{ctx} in_place"));
                let mut out = QuantumState::zero(1 << n);
                a.forward_into(&to_qs(&psi), &w, &mut out);
                assert_states_close(&from_qs(&out), &want, &format!("{ctx} into"));
            }
        }
    }
}

#[test]
fn ansatz_tiled_layer_path_matches_oracle() {
    // n >= 17 routes forward_layer_into through apply_layer_fused + tiled CZ.
    let n = 17;
    let mut r = rng(51);
    let a = ansatz(n, 2, true);
    let w = random_angles(a.param_count(), &mut r);
    let psi = random_state(n, &mut r);
    let want = ref_forward_layers(&psi, n, 0..2, &w, true);
    assert_states_close(
        &from_qs(&a.forward(&to_qs(&psi), &w)),
        &want,
        "tiled ansatz n=17",
    );
}

#[test]
fn ansatz_pruned_and_fused_readouts_match_oracle() {
    let mut r = rng(52);
    for n in 1..=7 {
        for depth in 1..=4 {
            for pi in [false, true] {
                let a = ansatz(n, depth, pi);
                let w = random_angles(pruned_len(n, depth), &mut r);
                let psi = random_state(n, &mut r);
                let ctx = format!("pruned n={n} depth={depth} pi={pi}");

                let pre = ref_forward_layers(&psi, n, 0..depth - 1, &w, pi);
                let mut post = pre.clone();
                let t0 = (depth - 1) * 2 * n;
                for q in 0..n {
                    ref_1q(&mut post, q, &ry_mat(w[t0 + q]));
                }

                let mut out = QuantumState::zero(1 << n);
                a.forward_l2_into(&to_qs(&psi), &w, &mut out);
                assert_states_close(&from_qs(&out), &pre, &format!("{ctx} l2_into"));
                let mut s = to_qs(&psi);
                a.forward_l2_in_place(&mut s, &w);
                assert_states_close(&from_qs(&s), &pre, &format!("{ctx} l2_in_place"));

                a.forward_pruned_into(&to_qs(&psi), &w, &mut out);
                assert_states_close(&from_qs(&out), &post, &format!("{ctx} pruned_into"));
                let mut s = to_qs(&psi);
                a.forward_pruned_in_place(&mut s, &w);
                assert_states_close(&from_qs(&s), &post, &format!("{ctx} pruned_in_place"));

                // Terminal Rz/CZ are diagonal, so Z / ZZ of the pruned state equal the full circuit's.
                let mut z = vec![0.0; n];
                let mut zz = vec![0.0; n * (n - 1) / 2];
                a.forward_fused_full_readout_into(&to_qs(&psi), &w, &mut out, &mut z, &mut zz);
                let mut z_only = vec![0.0; n];
                a.forward_fused_readout_into(&to_qs(&psi), &w, &mut out, &mut z_only);
                let mut idx = 0;
                for i in 0..n {
                    assert_close(z[i], ref_z(&post, i), &format!("{ctx} fused Z q={i}"));
                    assert_close(
                        z_only[i],
                        ref_z(&post, i),
                        &format!("{ctx} fused Z-only q={i}"),
                    );
                    for j in 0..i {
                        assert_close(
                            zz[idx],
                            ref_zz(&post, i, j),
                            &format!("{ctx} fused ZZ ({i},{j})"),
                        );
                        idx += 1;
                    }
                }
            }
        }
    }
}

/// Terminal readout: Ry(theta_q) on every qubit, then Z and ZZ.
fn ref_terminal_readout(state: &[C], n: usize, thetas: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut s = state.to_vec();
    for q in 0..n {
        ref_1q(&mut s, q, &ry_mat(thetas[q]));
    }
    let z = (0..n).map(|q| ref_z(&s, q)).collect();
    let mut zz = Vec::new();
    for i in 1..n {
        for j in 0..i {
            zz.push(ref_zz(&s, i, j));
        }
    }
    (z, zz)
}

fn assert_readout(z: &[f64], zz: Option<&[f64]>, want: &(Vec<f64>, Vec<f64>), ctx: &str) {
    for (q, (g, w)) in z.iter().zip(&want.0).enumerate() {
        assert_close(*g, *w, &format!("{ctx} Z q={q}"));
    }
    if let Some(zz) = zz {
        for (k, (g, w)) in zz.iter().zip(&want.1).enumerate() {
            assert_close(*g, *w, &format!("{ctx} ZZ pair {k}"));
        }
    }
}

#[test]
fn ansatz_midcircuit_feedforward_matches_oracle() {
    let mut r = rng(53);
    for n in 2..=6 {
        for depth in 2..=4 {
            for pi in [false, true] {
                let a = ansatz(n, depth, pi);
                let w = random_angles(pruned_len(n, depth), &mut r);
                let alphas: Vec<f64> = (0..n).map(|_| r.gen_range(-1.5..1.5)).collect();
                let src = r.gen_range(0..n);
                let psi = random_state(n, &mut r);
                let ctx = format!("midcircuit n={n} depth={depth} pi={pi}");
                let t0 = (depth - 1) * 2 * n;

                // Feed-forward into the terminal angles after all full layers.
                let pre = ref_forward_layers(&psi, n, 0..depth - 1, &w, pi);
                let m = ref_z(&pre, src);
                let thetas: Vec<f64> = (0..n).map(|q| w[t0 + q] + alphas[q] * m).collect();
                let want = ref_terminal_readout(&pre, n, &thetas);

                let mut out = QuantumState::zero(1 << n);
                let (mut z, mut zz) = (vec![0.0; n], vec![0.0; n * (n - 1) / 2]);
                let got_m = a.forward_midcircuit_feedforward_into(
                    &to_qs(&psi),
                    &w,
                    src,
                    &alphas,
                    &mut out,
                    &mut z,
                );
                assert_close(got_m, m, &format!("{ctx} m_src"));
                assert_readout(&z, None, &want, &format!("{ctx} ff_into"));

                a.forward_midcircuit_full_readout_into(
                    &to_qs(&psi),
                    &w,
                    src,
                    &alphas,
                    &mut out,
                    &mut z,
                    &mut zz,
                );
                assert_readout(&z, Some(&zz), &want, &format!("{ctx} ff_full_into"));

                let mut s = to_qs(&psi);
                a.forward_midcircuit_feedforward_in_place(&mut s, &w, src, &alphas, &mut z);
                assert_readout(&z, None, &want, &format!("{ctx} ff_in_place"));

                let mut s = to_qs(&psi);
                a.forward_midcircuit_full_readout_in_place(
                    &mut s, &w, src, &alphas, &mut z, &mut zz,
                );
                assert_readout(&z, Some(&zz), &want, &format!("{ctx} ff_full_in_place"));

                // L0 variant: steer with Ry(alpha * m) right after layer 0, then the remaining layers.
                let mut s0 = ref_forward_layers(&psi, n, 0..1, &w, pi);
                let m0 = ref_z(&s0, src);
                for q in 0..n {
                    ref_1q(&mut s0, q, &ry_mat(alphas[q] * m0));
                }
                for l in 1..depth - 1 {
                    ref_layer(&mut s0, n, l, &w);
                }
                let want0 = ref_terminal_readout(&s0, n, &w[t0..t0 + n]);
                let got_m0 = a.forward_midcircuit_l0_feedforward_into(
                    &to_qs(&psi),
                    &w,
                    src,
                    &alphas,
                    &mut out,
                    &mut z,
                );
                assert_close(got_m0, m0, &format!("{ctx} l0 m_src"));
                assert_readout(&z, None, &want0, &format!("{ctx} l0_ff"));
            }
        }
    }
}

#[test]
fn ansatz_data_reuploading_matches_oracle() {
    let mut r = rng(54);
    for n in 2..=6 {
        for depth in 2..=4 {
            for pi in [false, true] {
                let a = ansatz(n, depth, pi);
                let w = random_angles(pruned_len(n, depth), &mut r);
                let feats: Vec<f64> = (0..n + 2).map(|_| r.gen_range(-5.0..5.0)).collect();
                let scales: Vec<f64> = (0..n).map(|_| r.gen_range(-2.0..2.0)).collect();
                let alphas: Vec<f64> = (0..n).map(|_| r.gen_range(-1.5..1.5)).collect();
                let src = r.gen_range(0..n);
                let psi = random_state(n, &mut r);
                let ctx = format!("reupload n={n} depth={depth} pi={pi}");
                let t0 = (depth - 1) * 2 * n;

                let fnorm = feats.iter().map(|v| v * v).sum::<f64>().sqrt();
                let mut s = ref_forward_layers(&psi, n, 0..1, &w, pi);
                for q in 0..n {
                    ref_1q(&mut s, q, &ry_mat(scales[q] * feats[q] / fnorm));
                }
                for l in 1..depth - 1 {
                    ref_layer(&mut s, n, l, &w);
                }
                let m = ref_z(&s, src);
                let thetas: Vec<f64> = (0..n).map(|q| w[t0 + q] + alphas[q] * m).collect();
                let want = ref_terminal_readout(&s, n, &thetas);

                let (mut z, mut zz) = (vec![0.0; n], vec![0.0; n * (n - 1) / 2]);
                let mut st = to_qs(&psi);
                let got_m = a.forward_reuploading_in_place(
                    &mut st, &w, &feats, &scales, src, &alphas, &mut z,
                );
                assert_close(got_m, m, &format!("{ctx} m_src"));
                assert_readout(&z, None, &want, &ctx);

                let mut st = to_qs(&psi);
                a.forward_reuploading_full_readout_in_place(
                    &mut st, &w, &feats, &scales, src, &alphas, &mut z, &mut zz,
                );
                assert_readout(&z, Some(&zz), &want, &format!("{ctx} full"));
            }
        }
    }
}

#[test]
fn compiled_ansatz_matches_direct_forward() {
    let mut r = rng(55);
    for n in 2..=7 {
        for depth in 1..=5 {
            let a = ansatz(n, depth, depth % 2 == 0);
            let w = random_angles(a.param_count(), &mut r);
            let psi = random_state(n, &mut r);
            let want = from_qs(&a.forward(&to_qs(&psi), &w));
            let mut s = to_qs(&psi);
            a.compile(&w).execute(&mut s);
            assert_states_close(
                &from_qs(&s),
                &want,
                &format!("compiled ansatz n={n} depth={depth}"),
            );
        }
    }
}

// =========================================================================
// 7. Circuit compiler on random circuits
// =========================================================================

fn random_unitary(r: &mut ChaCha8Rng) -> (Unitary2x2, Mat2) {
    let (a, b, g) = (
        r.gen_range(-PI..PI),
        r.gen_range(-PI..PI),
        r.gen_range(-PI..PI),
    );
    let u = Unitary2x2::rz(g)
        .matmul(&Unitary2x2::ry(b))
        .matmul(&Unitary2x2::rz(a));
    let m = [[u.u00(), u.u01()], [u.u10(), u.u11()]];
    (u, m)
}

#[test]
fn compiler_random_circuits_match_oracle() {
    let mut r = rng(60);
    for trial in 0..60 {
        let n = 1 + trial % 6;
        let mut builder = CircuitBuilder::new(n);
        let psi = random_state(n, &mut r);
        let mut want = psi.clone();
        for _ in 0..40 {
            let q = r.gen_range(0..n);
            let kind = if n == 1 {
                r.gen_range(0..3)
            } else {
                r.gen_range(0..6)
            };
            match kind {
                0 => {
                    let t = r.gen_range(-PI..PI);
                    builder.ry(q, t);
                    ref_1q(&mut want, q, &ry_mat(t));
                }
                1 => {
                    let t = r.gen_range(-PI..PI);
                    builder.rz(q, t);
                    ref_1q(&mut want, q, &rz_mat(t));
                }
                2 => {
                    let (u, m) = random_unitary(&mut r);
                    builder.unitary_1q(q, u);
                    ref_1q(&mut want, q, &m);
                }
                3 | 4 => {
                    let t = (q + r.gen_range(1..n)) % n;
                    if kind == 3 {
                        builder.cz(q, t);
                        ref_cz(&mut want, q, t);
                    } else {
                        builder.cnot(q, t);
                        ref_cnot(&mut want, q, t);
                    }
                }
                _ => {
                    let signs: Vec<f64> = (0..n)
                        .map(|_| if r.gen_bool(0.5) { 1.0 } else { -1.0 })
                        .collect();
                    for (k, sgn) in signs.iter().enumerate() {
                        ref_1q(&mut want, k, &rz_mat(sgn * PI));
                    }
                    builder.rademacher(signs);
                }
            }
        }
        let mut s = to_qs(&psi);
        builder.compile().execute(&mut s);
        assert_states_close(
            &from_qs(&s),
            &want,
            &format!("compiler trial {trial} n={n}"),
        );
    }
}

// =========================================================================
// 8. Parameter-shift gradients vs. finite differences
// =========================================================================

#[test]
fn parameter_shift_gradient_matches_finite_difference() {
    let mut r = rng(70);
    for n in 2..=4 {
        let enc = DataEncodingLayer::new(n, EncodingKind::Angle);
        let a = ansatz(n, 2, false);
        let engine = ParameterShiftEngine::new(enc, a, PauliObservable::Z(0));
        let x = random_angles(n, &mut r);
        let w = random_angles(engine.ansatz.param_count(), &mut r);
        let grads = engine.compute_circuit_gradients(&x, &w);
        let h = 1e-6;
        for j in 0..w.len() {
            let mut wp = w.clone();
            wp[j] += h;
            let mut wm = w.clone();
            wm[j] -= h;
            let fd = (engine.predict(&x, &wp) - engine.predict(&x, &wm)) / (2.0 * h);
            assert!(
                (grads[j] - fd).abs() < 1e-6,
                "n={n} param {j}: shift {} vs FD {}",
                grads[j],
                fd
            );
        }
    }
}

fn observable_zoo(n: usize) -> Vec<PauliObservable> {
    let mut v = vec![
        PauliObservable::Z(0),
        PauliObservable::Z(n - 1),
        PauliObservable::X(n / 2),
        PauliObservable::GlobalMeanZ,
        PauliObservable::LinearZ((0..n).map(|q| (q, 0.3 * q as f64 - 0.5)).collect()),
    ];
    if n >= 2 {
        v.push(PauliObservable::ZZ(0, n - 1));
    }
    v
}

#[test]
fn adjoint_gradient_matches_parameter_shift_and_finite_differences() {
    let mut r = rng(71);
    for n in 2..=6 {
        for depth in 1..=4 {
            for pi in [false, true] {
                let a = ansatz(n, depth, pi);
                let x = random_angles(n, &mut r);
                let w = random_angles(a.param_count(), &mut r);
                for obs in observable_zoo(n) {
                    let engine = ParameterShiftEngine::new(
                        DataEncodingLayer::new(n, EncodingKind::Angle),
                        a.clone(),
                        obs.clone(),
                    );
                    let ctx = format!("n={n} depth={depth} pi={pi} obs={obs:?}");
                    let adj = engine.compute_circuit_gradients(&x, &w);
                    let ps = engine.compute_circuit_gradients_parameter_shift(&x, &w);
                    for (j, (ga, gp)) in adj.iter().zip(&ps).enumerate() {
                        assert!(
                            (ga - gp).abs() < 1e-10,
                            "{ctx} param {j}: adjoint {ga} vs shift {gp}"
                        );
                    }
                    // Spot-check against central finite differences.
                    for j in [0, w.len() / 2, w.len() - 1] {
                        let h = 1e-6;
                        let mut wp = w.clone();
                        wp[j] += h;
                        let mut wm = w.clone();
                        wm[j] -= h;
                        let fd = (engine.predict(&x, &wp) - engine.predict(&x, &wm)) / (2.0 * h);
                        assert!(
                            (adj[j] - fd).abs() < 1e-6,
                            "{ctx} param {j}: adjoint {} vs FD {fd}",
                            adj[j]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn adjoint_loss_gradients_match_parameter_shift() {
    let mut r = rng(72);
    for n in [3, 5] {
        let a = ansatz(n, 3, true);
        let w = random_angles(a.param_count(), &mut r);
        let psi = to_qs(&random_state(n, &mut r));
        let obs = observable_zoo(n);
        let targets: Vec<f64> = obs.iter().map(|_| r.gen_range(-1.0..1.0)).collect();
        let engine = ParameterShiftEngine::new(
            DataEncodingLayer::new(n, EncodingKind::Angle),
            a.clone(),
            PauliObservable::Z(0),
        );

        let (p_adj, l_adj, g_adj) =
            engine.compute_multi_observable_loss_gradients(&psi, &obs, &targets, &w);
        let (p_ps, l_ps, g_ps) = engine
            .compute_multi_observable_loss_gradients_parameter_shift(&psi, &obs, &targets, &w);
        for (a1, b1) in p_adj.iter().zip(&p_ps) {
            assert_close(*a1, *b1, "multi-observable prediction");
        }
        assert_close(l_adj, l_ps, "multi-observable loss");
        for (j, (ga, gp)) in g_adj.iter().zip(&g_ps).enumerate() {
            assert!(
                (ga - gp).abs() < 1e-10,
                "n={n} multi-observable param {j}: {ga} vs {gp}"
            );
        }

        let y = 0.25;
        let (pred, loss, g) = engine.compute_loss_gradients_for_state(&psi, y, &w);
        let base = engine.compute_circuit_gradients_for_state(&psi, &w);
        assert_close(pred, engine.predict_state(&psi, &w), "scalar prediction");
        assert_close(loss, 0.5 * (pred - y) * (pred - y), "scalar loss");
        for (gj, bj) in g.iter().zip(&base) {
            assert!((gj - (pred - y) * bj).abs() < 1e-12, "scalar loss gradient");
        }
    }
}

#[test]
fn adjoint_matches_parameter_shift_for_zinc_readout_setup() {
    // Mirrors the ZINC trainers: amplitude-encoded 19 features, pi layer, depth 3, and a loss
    // that weights every <Z_q> by a coefficient derived from the measured values.
    let n = 12;
    let mut r = rng(74);
    let a = ansatz(n, 3, true);
    let w = random_angles(a.param_count(), &mut r);
    let feats: Vec<f64> = (0..19).map(|_| r.gen_range(0.0..50.0)).collect();
    let input = DataEncodingLayer::new(n, EncodingKind::Amplitude).encode(&feats);
    let z_obs: Vec<PauliObservable> = (0..n).map(PauliObservable::Z).collect();
    let readout: Vec<f64> = (0..n).map(|_| r.gen_range(-1.0..1.0)).collect();

    let res = t1000::adjoint_gradient(&a, &input, &w, &z_obs, |z| {
        // Nonlinear in z, like an MSE head: c_q = readout_q * (sum_k readout_k z_k - 0.3)
        let pred: f64 = readout.iter().zip(z).map(|(rw, zq)| rw * zq).sum();
        readout.iter().map(|rw| rw * (pred - 0.3)).collect()
    });

    // Parameter-shift reference for d/dθ of sum_q c_q <Z_q> with c fixed at the base point.
    let pred: f64 = readout
        .iter()
        .zip(&res.expectations)
        .map(|(rw, zq)| rw * zq)
        .sum();
    let coeffs: Vec<f64> = readout.iter().map(|rw| rw * (pred - 0.3)).collect();
    let objective = |weights: &[f64]| {
        let s = a.forward(&input, weights);
        z_obs
            .iter()
            .zip(&coeffs)
            .map(|(o, c)| c * o.expectation_value(&s))
            .sum::<f64>()
    };
    for j in 0..w.len() {
        let mut wp = w.clone();
        wp[j] += PI / 2.0;
        let mut wm = w.clone();
        wm[j] -= PI / 2.0;
        let ps = 0.5 * (objective(&wp) - objective(&wm));
        assert!(
            (res.gradient[j] - ps).abs() < 1e-10,
            "param {j}: adjoint {} vs shift {ps}",
            res.gradient[j]
        );
    }
}

/// The ansatz expressed as a gate program (weights as parameters, pi layer as fixed Rz).
fn ansatz_program(n: usize, depth: usize, pi: bool) -> Vec<t1000::adjoint::GateOp> {
    use t1000::adjoint::{Angle, GateOp};
    let mut ops = Vec::new();
    if pi {
        for q in 0..n {
            let sign = if q % 2 == 0 { 1.0 } else { -1.0 };
            ops.push(GateOp::Rz {
                qubit: q,
                angle: Angle::fixed(sign * PI),
            });
        }
    }
    for layer in 0..depth {
        for q in 0..n {
            ops.push(GateOp::Ry {
                qubit: q,
                angle: Angle::param(layer * 2 * n + 2 * q),
            });
            ops.push(GateOp::Rz {
                qubit: q,
                angle: Angle::param(layer * 2 * n + 2 * q + 1),
            });
        }
        for (a, b) in ref_cz_pairs(n, layer) {
            ops.push(GateOp::Cz(a, b));
        }
    }
    ops
}

#[test]
fn program_adjoint_matches_ansatz_adjoint() {
    let mut r = rng(75);
    for (n, depth) in [(3, 2), (5, 3), (6, 4)] {
        let a = ansatz(n, depth, true);
        let w = random_angles(a.param_count(), &mut r);
        let input = to_qs(&random_state(n, &mut r));
        let obs: Vec<PauliObservable> = (0..n)
            .map(PauliObservable::Z)
            .chain([PauliObservable::ZZ(0, n - 1)])
            .collect();
        let coeffs: Vec<f64> = obs.iter().map(|_| r.gen_range(-1.0..1.0)).collect();
        let reference = t1000::adjoint_gradient(&a, &input, &w, &obs, |_| coeffs.clone());
        let ops = ansatz_program(n, depth, true);
        let got = t1000::adjoint::adjoint_program(&input, &ops, &w, &obs, |_| coeffs.clone());
        let state = t1000::adjoint::run_program(&input, &ops, &w);
        assert_states_close(
            &from_qs(&state),
            &from_qs(&a.forward(&input, &w)),
            "program forward",
        );
        for (k, (x, y)) in got
            .expectations
            .iter()
            .zip(&reference.expectations)
            .enumerate()
        {
            assert_close(*x, *y, &format!("expectation {k}"));
        }
        for (j, (x, y)) in got.gradient.iter().zip(&reference.gradient).enumerate() {
            assert!(
                (x - y).abs() < 1e-10,
                "n={n} depth={depth} param {j}: program {x} vs ansatz {y}"
            );
        }
    }
}

#[test]
fn program_adjoint_handles_scaled_shared_and_fixed_angles() {
    use t1000::adjoint::{adjoint_program, run_program, Angle, GateOp};
    let mut r = rng(76);
    for n in [2, 4, 15] {
        let n_params = 5;
        let mut ops = Vec::new();
        for _ in 0..30 {
            let q = r.gen_range(0..n);
            let angle = match r.gen_range(0..3) {
                0 => Angle::fixed(r.gen_range(-PI..PI)),
                1 => Angle::param(r.gen_range(0..n_params)),
                _ => Angle {
                    param: Some(r.gen_range(0..n_params)),
                    scale: r.gen_range(-2.0..2.0),
                    offset: r.gen_range(-1.0..1.0),
                },
            };
            ops.push(if r.gen_bool(0.5) {
                GateOp::Ry { qubit: q, angle }
            } else {
                GateOp::Rz { qubit: q, angle }
            });
            if n >= 2 && r.gen_bool(0.3) {
                let a = r.gen_range(0..n);
                ops.push(GateOp::Cz(a, (a + 1) % n));
            }
        }
        let params = random_angles(n_params, &mut r);
        let input = to_qs(&random_state(n, &mut r));
        let obs: Vec<PauliObservable> = vec![
            PauliObservable::Z(0),
            PauliObservable::X(n - 1),
            PauliObservable::GlobalMeanZ,
        ];
        let coeffs = vec![0.7, -1.3, 0.4];
        let res = adjoint_program(&input, &ops, &params, &obs, |_| coeffs.clone());
        let objective = |p: &[f64]| {
            let s = run_program(&input, &ops, p);
            obs.iter()
                .zip(&coeffs)
                .map(|(o, c)| c * o.expectation_value(&s))
                .sum::<f64>()
        };
        for j in 0..n_params {
            let h = 1e-6;
            let mut pp = params.clone();
            pp[j] += h;
            let mut pm = params.clone();
            pm[j] -= h;
            let fd = (objective(&pp) - objective(&pm)) / (2.0 * h);
            assert!(
                (res.gradient[j] - fd).abs() < 1e-6,
                "n={n} param {j}: adjoint {} vs FD {fd}",
                res.gradient[j]
            );
        }
    }
}

#[test]
fn adjoint_gradient_on_multithreaded_path() {
    // 15 qubits crosses PARALLEL_MIN_DIM, so the backward sweep uses the parallel kernels.
    let n = 15;
    let mut r = rng(73);
    let a = ansatz(n, 2, true);
    let w = random_angles(a.param_count(), &mut r);
    let x = random_angles(n, &mut r);
    let engine = ParameterShiftEngine::new(
        DataEncodingLayer::new(n, EncodingKind::Angle),
        a,
        PauliObservable::GlobalMeanZ,
    );
    let adj = engine.compute_circuit_gradients(&x, &w);
    let ps = engine.compute_circuit_gradients_parameter_shift(&x, &w);
    for (j, (ga, gp)) in adj.iter().zip(&ps).enumerate() {
        assert!(
            (ga - gp).abs() < 1e-10,
            "param {j}: adjoint {ga} vs shift {gp}"
        );
    }
}

// =========================================================================
// 9. Precondition guards: invalid input must panic, never corrupt memory
// =========================================================================

#[test]
#[should_panic(expected = "out of range")]
fn ry_rejects_out_of_range_qubit() {
    let mut s = QuantumState::new(8);
    simd::apply_ry_kernel(&mut s.real, &mut s.imag, 3, 0.1);
}

#[test]
#[should_panic(expected = "power of two")]
fn kernels_reject_non_power_of_two_buffers() {
    let (mut re, mut im) = (vec![0.0; 1500], vec![0.0; 1500]);
    simd::apply_ry_kernel(&mut re, &mut im, 10, 0.1);
}

#[test]
#[should_panic(expected = "length mismatch")]
fn kernels_reject_mismatched_real_imag() {
    let (mut re, mut im) = (vec![0.0; 16], vec![0.0; 8]);
    simd::apply_rz_kernel(&mut re, &mut im, 0, 0.1);
}

#[test]
#[should_panic(expected = "distinct qubits")]
fn cz_rejects_repeated_qubit() {
    let mut s = QuantumState::new(16);
    simd::apply_cz_kernel(&mut s.real, &mut s.imag, 2, 2);
}

#[test]
#[should_panic(expected = "out of range")]
fn tiled_cz_rejects_out_of_range_qubit() {
    let mut s = QuantumState::new(16);
    tiled_apply_cz(&mut s.real, &mut s.imag, 1, 20);
}

#[test]
#[should_panic(expected = "inner product operands")]
fn inner_product_rejects_mismatched_lengths() {
    let a = vec![0.0; 16];
    let b = vec![0.0; 8];
    simd::inner_product_kernel(&a, &a, &b, &b);
}

#[test]
#[should_panic(expected = "power of two")]
fn quantum_state_rejects_non_power_of_two_dim() {
    let _ = QuantumState::new(1500);
}

#[test]
#[should_panic(expected = "qubit expectations")]
fn dual_zx_rejects_too_many_outputs() {
    let s = QuantumState::new(8);
    let (mut z, mut x) = (vec![0.0; 4], vec![0.0; 4]);
    simd::measure_dual_zx_expectations(&s.real, &s.imag, &mut z, &mut x);
}

#[test]
#[should_panic(expected = "fused readout supports at most")]
fn fused_readout_rejects_more_than_16_qubits() {
    let s = QuantumState::new(1 << 17);
    let theta = vec![0.0; 17];
    let mut z = vec![0.0; 17];
    simd::measure_fused_ry_z_readout(&s.real, &s.imag, &theta, &mut z);
}

#[test]
#[should_panic(expected = "Pruned weights length mismatch")]
fn midcircuit_rejects_wrong_weight_count() {
    let a = ansatz(4, 3, false);
    let mut s = QuantumState::new(16);
    let mut z = vec![0.0; 4];
    a.forward_midcircuit_feedforward_in_place(&mut s, &[0.0; 5], 0, &[0.0; 4], &mut z);
}

#[test]
#[should_panic(expected = "depth >= 2")]
fn reuploading_rejects_depth_one() {
    let a = ansatz(4, 1, false);
    let mut s = QuantumState::new(16);
    let mut z = vec![0.0; 4];
    a.forward_reuploading_in_place(
        &mut s, &[0.0; 4], &[1.0; 4], &[1.0; 4], 0, &[0.0; 4], &mut z,
    );
}
