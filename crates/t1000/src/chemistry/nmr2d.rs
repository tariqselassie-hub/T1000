// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Pure Rust 2D Molecular Correlation Spectroscopy (2D COSY / HSQC) Engine.
//!
//! Simulates physical Nuclear Magnetic Resonance (NMR) Homonuclear Correlation Spectroscopy
//! from OpenSMILES molecular graphs. Generates 32x32 (1,024-amplitude) 2D frequency spectra
//! matrices with diagonal chemical shifts and off-diagonal vicinal scalar J-coupling cross-peaks,
//! normalized to unit L2 Euclidean length for direct 10-Qubit Hilbert state injection.

use crate::smiles::{parse, BondKind, ParsedSmiles, SmilesError};

pub const NMR_GRID_DIM: usize = 32;
pub const NMR_STATE_DIM: usize = NMR_GRID_DIM * NMR_GRID_DIM; // 1,024
pub const PPM_MIN: f64 = 0.0;
pub const PPM_MAX: f64 = 12.0;

/// Proton chemical environment with isotropic chemical shift (ppm) and proton count.
#[derive(Debug, Clone, PartialEq)]
pub struct ProtonEnvironment {
    pub atom_idx: usize,
    pub shift_ppm: f64,
    pub h_count: u32,
    pub is_aromatic: bool,
    pub is_heteroatom_bound: bool,
}

/// Computes the empirical 1H chemical shifts (in ppm) for all proton-bearing atoms in a molecule.
pub fn predict_proton_environments(parsed: &ParsedSmiles) -> Vec<ProtonEnvironment> {
    let mut envs = Vec::new();

    // Map each atom to its connected neighbors: (neighbor_idx, bond_kind)
    let mut adj: Vec<Vec<(usize, BondKind)>> = vec![Vec::new(); parsed.atoms.len()];
    for &(a, b, kind) in &parsed.bonds {
        adj[a].push((b, kind));
        adj[b].push((a, kind));
    }

    for (i, atom) in parsed.atoms.iter().enumerate() {
        let h_count = parsed.hydrogens[i];
        if h_count == 0 {
            continue;
        }
        let mut has_c_double_o = false;
        let mut has_c_double_c = false;
        let mut has_o_single = false;
        let mut has_n_single = false;
        let mut has_halogen = false;

        // Check immediate bonded neighbors
        for &(nbr_idx, kind) in &adj[i] {
            let nbr = &parsed.atoms[nbr_idx];

            if nbr.element == "O" {
                if kind == BondKind::Double {
                    has_c_double_o = true;
                } else {
                    has_o_single = true;
                }
            } else if nbr.element == "N" {
                has_n_single = true;
            } else if nbr.element == "F" || nbr.element == "Cl" || nbr.element == "Br" {
                has_halogen = true;
            } else if nbr.element == "C" {
                if kind == BondKind::Double {
                    has_c_double_c = true;
                }
                // Check if neighbor carbon is a carbonyl (C=O)
                for &(n2_idx, k2) in &adj[nbr_idx] {
                    if parsed.atoms[n2_idx].element == "O" && k2 == BondKind::Double {
                        has_c_double_o = true;
                    }
                }
            }
        }

        let shift = if atom.element == "O" {
            // Hydroxyl proton (-OH)
            2.50
        } else if atom.element == "N" {
            // Amine/Amide proton (-NH-)
            if has_c_double_o { 6.80 } else { 2.10 }
        } else if atom.aromatic {
            // Aromatic ring protons (Benzene / Pyridine / Thiophene)
            if atom.element == "N" {
                8.50 // Pyridine alpha-proton
            } else if has_o_single || has_n_single {
                6.85
            } else {
                7.40
            }
        } else if has_c_double_o && h_count == 1 && adj[i].len() == 1 {
            // Aldehyde proton (-CH=O)
            9.75
        } else if has_c_double_c {
            // Vinylic alkene proton (=CH-)
            5.25
        } else if has_o_single {
            // Alpha to Oxygen (Ether / Alcohol / Ester)
            3.65
        } else if has_halogen {
            // Alpha to Halogen (-CH2-Cl)
            3.45
        } else if has_n_single {
            // Alpha to Nitrogen (-CH2-NH2)
            2.65
        } else if has_c_double_o {
            // Alpha to Carbonyl (-CH2-C=O)
            2.25
        } else if h_count == 3 {
            // Methyl group (-CH3)
            0.90
        } else if h_count == 2 {
            // Methylene group (-CH2-)
            1.30
        } else {
            // Methine group (-CH-)
            1.55
        };

        envs.push(ProtonEnvironment {
            atom_idx: i,
            shift_ppm: shift,
            h_count,
            is_aromatic: atom.aromatic,
            is_heteroatom_bound: has_o_single || has_n_single,
        });
    }

    envs
}

/// Generates a 32x32 2D COSY spectrum matrix (1,024 bins, exact L2 normalized) from a SMILES string.
pub fn generate_cosy_spectrum(smiles: &str) -> Result<[f64; NMR_STATE_DIM], SmilesError> {
    let parsed = parse(smiles)?;
    Ok(render_cosy_matrix(&parsed))
}

/// Renders the 32x32 2D COSY NMR frequency matrix from parsed molecular graph.
pub fn render_cosy_matrix(parsed: &ParsedSmiles) -> [f64; NMR_STATE_DIM] {
    let envs = predict_proton_environments(parsed);
    let mut matrix = [0.0; NMR_STATE_DIM];

    if envs.is_empty() {
        // Fallback for hydrogen-free molecules (e.g. CCl4): uniform ground state
        let inv = 1.0 / (NMR_STATE_DIM as f64).sqrt();
        matrix.fill(inv);
        return matrix;
    }

    let bin_width = (PPM_MAX - PPM_MIN) / (NMR_GRID_DIM as f64);
    let sigma_bin = 0.45; // Gaussian line-broadening width in bins
    let two_sigma_sq = 2.0 * sigma_bin * sigma_bin;

    // Helper to map ppm to continuous grid bin [0..31]
    let ppm_to_bin = |ppm: f64| -> f64 {
        ((ppm - PPM_MIN) / bin_width).clamp(0.0, (NMR_GRID_DIM - 1) as f64)
    };

    // 1. Diagonal Peaks: S(w1, w2) where w1 == w2
    for env in &envs {
        let b = ppm_to_bin(env.shift_ppm);
        let weight = env.h_count as f64;

        let r_min = ((b - 3.0).floor() as isize).max(0) as usize;
        let r_max = ((b + 3.0).ceil() as isize).min(NMR_GRID_DIM as isize - 1) as usize;

        for r in r_min..=r_max {
            let dr = r as f64 - b;
            for c in r_min..=r_max {
                let dc = c as f64 - b;
                let dist_sq = dr * dr + dc * dc;
                let intensity = weight * (-dist_sq / two_sigma_sq).exp();
                matrix[r * NMR_GRID_DIM + c] += intensity;
            }
        }
    }

    // 2. Off-Diagonal Cross-Peaks: For 3-bond vicinal scalar coupling (atom_a bonded to atom_b)
    for &(a, b, _) in &parsed.bonds {
        let env_a = envs.iter().find(|e| e.atom_idx == a);
        let env_b = envs.iter().find(|e| e.atom_idx == b);

        if let (Some(ea), Some(eb)) = (env_a, env_b) {
            let b_a = ppm_to_bin(ea.shift_ppm);
            let b_b = ppm_to_bin(eb.shift_ppm);
            let cross_weight = 0.75 * (ea.h_count as f64 * eb.h_count as f64).sqrt();

            // Symmetric cross-peaks at (b_a, b_b) and (b_b, b_a)
            for &(center_r, center_c) in &[(b_a, b_b), (b_b, b_a)] {
                let r_min = ((center_r - 2.5).floor() as isize).max(0) as usize;
                let r_max = ((center_r + 2.5).ceil() as isize).min(NMR_GRID_DIM as isize - 1) as usize;
                let c_min = ((center_c - 2.5).floor() as isize).max(0) as usize;
                let c_max = ((center_c + 2.5).ceil() as isize).min(NMR_GRID_DIM as isize - 1) as usize;

                for r in r_min..=r_max {
                    let dr = r as f64 - center_r;
                    for c in c_min..=c_max {
                        let dc = c as f64 - center_c;
                        let dist_sq = dr * dr + dc * dc;
                        let intensity = cross_weight * (-dist_sq / two_sigma_sq).exp();
                        matrix[r * NMR_GRID_DIM + c] += intensity;
                    }
                }
            }
        }
    }

    // 3. Instrument Baseline Floor: simulates finite RF sensitivity and receiver noise
    for val in matrix.iter_mut() {
        *val += 0.005;
    }

    // 4. Exact L2 Normalization into 10-Qubit Statevector
    let mut norm_sq = 0.0;
    for &val in &matrix {
        norm_sq += val * val;
    }

    if norm_sq > 1e-12 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for val in matrix.iter_mut() {
            *val *= inv_norm;
        }
    }

    matrix
}

/// Extracts 65 2D spatial, projection, and cross-peak features from the 32x32 NMR matrix.
pub const N_NMR_SPATIAL: usize = 65;

pub fn extract_nmr_features(matrix: &[f64; NMR_STATE_DIM]) -> [f64; N_NMR_SPATIAL] {
    let mut feats = [0.0; N_NMR_SPATIAL];

    // 1. Amplitudes 0..16: 4x4 Coarse block averages (diagonal & off-diagonal cross zones)
    for r4 in 0..4 {
        for c4 in 0..4 {
            let mut sum = 0.0;
            for dr in 0..8 {
                for dc in 0..8 {
                    sum += matrix[(r4 * 8 + dr) * NMR_GRID_DIM + (c4 * 8 + dc)];
                }
            }
            feats[r4 * 4 + c4] = sum / 64.0;
        }
    }

    // 2. Amplitudes 16..32: 1D Chemical Shift Projection (Marginal along diagonal axis)
    for i in 0..16 {
        let mut row_sum = 0.0;
        for c in 0..NMR_GRID_DIM {
            row_sum += matrix[(i * 2) * NMR_GRID_DIM + c];
            row_sum += matrix[(i * 2 + 1) * NMR_GRID_DIM + c];
        }
        feats[16 + i] = row_sum / (2.0 * NMR_GRID_DIM as f64);
    }

    // 3. Amplitudes 32..48: Diagonal Trace vs Off-Diagonal Cross-Peak Ratio in 16 frequency bands
    for i in 0..16 {
        let mut diag_sum = 0.0;
        let mut off_sum = 0.0;
        for step in 0..2 {
            let idx = i * 2 + step;
            diag_sum += matrix[idx * NMR_GRID_DIM + idx];
            for c in 0..NMR_GRID_DIM {
                if (c as isize - idx as isize).abs() > 3 {
                    off_sum += matrix[idx * NMR_GRID_DIM + c];
                }
            }
        }
        feats[32 + i] = if diag_sum > 1e-6 { off_sum / diag_sum } else { 0.0 };
    }

    // 4. Amplitudes 48..65 (17 values): Total Spectral Moments & Cross-Coupling Entropy
    let mut total_cross_mass = 0.0;
    let mut total_diag_mass = 0.0;
    for r in 0..NMR_GRID_DIM {
        for c in 0..NMR_GRID_DIM {
            let v = matrix[r * NMR_GRID_DIM + c];
            if r == c {
                total_diag_mass += v;
            } else if (r as isize - c as isize).abs() > 2 {
                total_cross_mass += v;
            }
        }
    }
    feats[48] = total_cross_mass;
    feats[49] = total_diag_mass;
    feats[50] = if total_diag_mass > 1e-6 { total_cross_mass / total_diag_mass } else { 0.0 };

    for k in 0..14 {
        feats[51 + k] = matrix[k * 2 * NMR_GRID_DIM + (31 - k * 2)];
    }

    feats
}

/// Number of targeted bipartite Pauli observables for 2D COSY NMR.
pub const N_NMR_OBSERVABLES: usize = 80;

/// Computes the 80-observable Pauli expectation vector for a normalized 10-qubit statevector.
/// Treats H_1024 as bipartite tensor product H_w1 (x) H_w2 (32x32 = 1,024).
///
/// Layout:
///   [0..10]  : 10 Single-qubit Z_i (marginal chemical shift projections along w1 and w2)
///   [10..20] : 10 Single-qubit X_i (local transverse phase across adjacent dyadic bins)
///   [20..65] : 45 All-pairs Z_i Z_j (0 <= i < j < 10) (intensity correlations)
///   [65..70] :  5 Dyadic Mirror X_{s+5} X_s (s = 0..5) (real symmetric cross-peak bit-flip exchange)
///   [70..75] :  5 Dyadic Mirror Y_{s+5} Y_s (s = 0..5) (phase-inverted cross-peak interference)
///   [75..79] :  4 Adjacent-Scale Cross X_{s+6} X_s (s = 0..4) (cross-scale J-couplings)
///   [79]     :  1 Global 10-qubit parity Z^{\otimes 10} (global parity anchor)
#[inline]
pub fn measure_nmr2d_pauli_80(
    re: &[f64; NMR_STATE_DIM],
    im: &[f64; NMR_STATE_DIM],
) -> [f64; N_NMR_OBSERVABLES] {
    let mut obs = [0.0f64; N_NMR_OBSERVABLES];

    // Precompute bitmasks for the 5 dyadic mirror pairs (q_{s+5}, q_s)
    const MIRROR_MASKS: [usize; 5] = [
        (1 << 5) | (1 << 0), // s=0: 0x021 (fine J-coupling scale, 0.3125 ppm)
        (1 << 6) | (1 << 1), // s=1: 0x042 (0.625 ppm)
        (1 << 7) | (1 << 2), // s=2: 0x084 (1.250 ppm)
        (1 << 8) | (1 << 3), // s=3: 0x108 (2.500 ppm)
        (1 << 9) | (1 << 4), // s=4: 0x210 (coarse quadrant, 5.000 ppm)
    ];

    // Precompute bitmasks for the 4 adjacent-scale cross pairs (q_{s+6}, q_s)
    const ADJ_MASKS: [usize; 4] = [
        (1 << 6) | (1 << 0), // 0x041
        (1 << 7) | (1 << 1), // 0x082
        (1 << 8) | (1 << 2), // 0x104
        (1 << 9) | (1 << 3), // 0x208
    ];

    for k in 0..NMR_STATE_DIM {
        let rk = re[k];
        let ik = im[k];
        let prob = rk * rk + ik * ik;

        // 1. Single-qubit Z_i [0..10] and X_i [10..20]
        for i in 0..10usize {
            let mask_i = 1usize << i;
            let sign_i = if (k & mask_i) == 0 { 1.0 } else { -1.0 };
            obs[i] += sign_i * prob;

            // Accumulate X_i only when bit i is 0 to count unordered pair {k, k ^ mask_i} cleanly
            if (k & mask_i) == 0 {
                let kp = k ^ mask_i;
                obs[10 + i] += 2.0 * (rk * re[kp] + ik * im[kp]);
            }
        }

        // 2. All-pairs Z_i Z_j [20..65]
        let mut zz_idx = 20usize;
        for i in 0..10usize {
            let bit_i = (k >> i) & 1;
            for j in (i + 1)..10usize {
                let bit_j = (k >> j) & 1;
                let sign_ij = if (bit_i ^ bit_j) == 0 { 1.0 } else { -1.0 };
                obs[zz_idx] += sign_ij * prob;
                zz_idx += 1;
            }
        }

        // 3. Dyadic Mirror XX [65..70] and YY [70..75]
        for (s, &mask_ab) in MIRROR_MASKS.iter().enumerate() {
            let kp = k ^ mask_ab;
            if k < kp {
                let real_corr = 2.0 * (rk * re[kp] + ik * im[kp]);
                obs[65 + s] += real_corr; // <X_{s+5} X_s>

                let bit_r = (k >> (s + 5)) & 1;
                let bit_c = (k >> s) & 1;
                let yy_sign = if (bit_r ^ bit_c) == 0 { -1.0 } else { 1.0 };
                obs[70 + s] += yy_sign * real_corr; // <Y_{s+5} Y_s>
            }
        }

        // 4. Adjacent-Scale Cross XX [75..79]
        for (s, &mask_adj) in ADJ_MASKS.iter().enumerate() {
            let kp = k ^ mask_adj;
            if k < kp {
                obs[75 + s] += 2.0 * (rk * re[kp] + ik * im[kp]);
            }
        }

        // 5. Global 10-qubit parity Z^{\otimes 10} [79]
        let parity_sign = if (k.count_ones() & 1) == 0 { 1.0 } else { -1.0 };
        obs[79] += parity_sign * prob;
    }

    obs
}

/// Constructs the list of 80 PauliObservable descriptors matching measure_nmr2d_pauli_80.
pub fn build_nmr2d_80_observables() -> Vec<crate::measurements::PauliObservable> {
    use crate::measurements::PauliObservable;

    let mut observables: Vec<PauliObservable> = (0..10).map(PauliObservable::Z).collect();
    for q in 0..10 {
        observables.push(PauliObservable::X(q));
    }
    for i in 0..10 {
        for j in (i + 1)..10 {
            observables.push(PauliObservable::ZZ(i, j));
        }
    }
    for s in 0..5 {
        observables.push(PauliObservable::XX(s + 5, s));
    }
    for s in 0..5 {
        observables.push(PauliObservable::YY(s + 5, s));
    }
    for s in 0..4 {
        observables.push(PauliObservable::XX(s + 6, s));
    }
    observables.push(PauliObservable::AllZ);

    assert_eq!(observables.len(), N_NMR_OBSERVABLES);
    observables
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::QuantumState;

    #[test]
    fn test_ethanol_cosy_generation() {
        // Ethanol: CCO (H3C-CH2-OH)
        let matrix = generate_cosy_spectrum("CCO").expect("Valid SMILES");
        assert_eq!(matrix.len(), 1024);

        // Verify exact L2 normalization
        let norm_sq: f64 = matrix.iter().map(|&x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5, "Matrix must be L2 normalized");
    }

    #[test]
    fn test_benzene_cosy_diagonal_dominance() {
        // Benzene: c1ccccc1 (all aromatic protons ~7.4 ppm)
        let matrix = generate_cosy_spectrum("c1ccccc1").expect("Valid SMILES");
        let bin_7_4 = ((7.4 - PPM_MIN) / ((PPM_MAX - PPM_MIN) / 32.0)) as usize;
        let peak_val = matrix[bin_7_4 * NMR_GRID_DIM + bin_7_4];
        assert!(peak_val > 0.05, "Aromatic peak should be prominent on diagonal");
    }

    #[test]
    fn test_measure_nmr2d_pauli_80_exact_match() {
        let matrix = generate_cosy_spectrum("CCO").expect("Valid SMILES");
        let mut state = QuantumState::new(NMR_STATE_DIM);
        state.real.copy_from_slice(&matrix);
        // Inject small imaginary components to test complex interference
        for (i, im) in state.imag.iter_mut().enumerate() {
            *im = 0.05 * (i as f64).sin();
        }
        let norm_sq: f64 = state.real.iter().zip(&state.imag).map(|(&r, &im)| r * r + im * im).sum();
        let inv_n = 1.0 / norm_sq.sqrt();
        for (r, im) in state.real.iter_mut().zip(&mut state.imag) {
            *r *= inv_n;
            *im *= inv_n;
        }

        let mut re_arr = [0.0; NMR_STATE_DIM];
        let mut im_arr = [0.0; NMR_STATE_DIM];
        re_arr.copy_from_slice(&state.real);
        im_arr.copy_from_slice(&state.imag);

        let fast_obs = measure_nmr2d_pauli_80(&re_arr, &im_arr);
        let individual_obs = build_nmr2d_80_observables();

        for (idx, obs_op) in individual_obs.iter().enumerate() {
            let exp_val = obs_op.expectation_value(&state);
            let fast_val = fast_obs[idx];
            assert!(
                (exp_val - fast_val).abs() < 1e-10,
                "Observable [{}] mismatch: fast={:.8}, exact={:.8}",
                idx,
                fast_val,
                exp_val
            );
        }
    }
}
