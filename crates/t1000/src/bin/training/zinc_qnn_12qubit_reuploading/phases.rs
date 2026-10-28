// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use super::report;
use super::*;
use std::time::Instant;
use t1000::zinc_stage_common::{
    central_difference, composite_loss, fit_and_score, linear_predict, mols_per_sec, phase_header,
    print_angle_row, Adam, Fit, FEATS_10640, RIDGE,
};
use t1000::zinc_training_common::solve_ridge_regression;

/// Angles (re-upload scales, feed-forward alphas) of a circuit evaluation.
pub type Angles<'a> = (&'a [f64], &'a [f64]);

/// Predicted logP of molecule #10640 at each architecture stage.
#[derive(Debug)]
pub struct Audit {
    pub base: f64,
    pub ff: f64,
    pub dru: f64,
    pub uni: f64,
}

/// Molecules per second of each readout.
#[derive(Debug)]
pub struct Throughput {
    pub baseline: f64,
    pub reupload: f64,
    pub unified: f64,
}

fn z12_rows(
    circuit: &Circuit,
    samples: &[ZincMolecule],
    angles: Angles,
    state: &mut QuantumState,
) -> Vec<Vec<f64>> {
    samples
        .iter()
        .map(|s| circuit.z12(&s.features, angles, state).to_vec())
        .collect()
}

fn obs78_rows(
    circuit: &Circuit,
    samples: &[ZincMolecule],
    angles: Angles,
    state: &mut QuantumState,
) -> Vec<Vec<f64>> {
    samples
        .iter()
        .map(|s| circuit.obs78(&s.features, angles, state))
        .collect()
}

/// Ridge fit of the 12 marginals at `angles`.
fn fit_z12(circuit: &Circuit, data: &Data, angles: Angles, state: &mut QuantumState) -> Fit {
    let x_train = z12_rows(circuit, &data.train, angles, state);
    let x_test = z12_rows(circuit, &data.test, angles, state);
    fit_and_score(data, &x_train, &x_test)
}

/// Phase 1: the no-mechanism baseline and Mechanism #2 alone (feed-forward on, upload off).
pub fn baselines(
    circuit: &Circuit,
    data: &Data,
    alphas: &[f64],
    state: &mut QuantumState,
) -> (Fit, Fit) {
    println!();
    phase_header(" [PHASE 1] BASELINE & MECHANISM #2 VERIFICATION                          ");
    let zeros = vec![0.0; N_QUBITS];
    let base = fit_z12(circuit, data, (&zeros, &zeros), state);
    let ff = fit_z12(circuit, data, (&zeros, alphas), state);
    report::print_baselines(&base, &ff);
    (base, ff)
}

/// Composite MSE loss of the ridge head refit for the candidate re-upload `scales`.
fn eval_dru_loss(
    circuit: &Circuit,
    scales: &[f64],
    alphas: &[f64],
    samples: &[ZincMolecule],
) -> (f64, Vec<Vec<f64>>, Vec<f64>) {
    let mut state = QuantumState::new(DIM);
    let x_mat = z12_rows(circuit, samples, (scales, alphas), &mut state);
    let y_mat: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| vec![s.log_p, s.qed, s.sas])
        .collect();

    let (w, b) = solve_ridge_regression(&x_mat, &y_mat, RIDGE);
    (composite_loss(&w, &b, &x_mat, &y_mat), w, b)
}

const N_EPOCHS: usize = 12;
const LR: f64 = 0.10;

/// Phase 2: optimise the 12 inter-layer re-upload scales with Adam over a nested ridge fit.
pub fn train_scales(circuit: &Circuit, train: &[ZincMolecule], alphas: &[f64]) -> Vec<f64> {
    println!();
    report::print_phase2_intro();
    let mut scales = vec![0.0; N_QUBITS];
    // Seed with feature-proportional initialization
    scales[0] = 0.5; // Heavy atoms
    scales[1] = 0.8; // MW est
    scales[8] = 1.0; // Halogens / Sulfur / Electronegatives
    scales[9] = 0.5;

    let opt_subset: Vec<ZincMolecule> = train.iter().take(2_000).cloned().collect();
    let mut adam = Adam::new(N_QUBITS, LR);
    let mut state = QuantumState::new(DIM);
    let t_opt = Instant::now();

    report::print_scale_table_header();
    for epoch in 1..=N_EPOCHS {
        let (loss_curr, w_curr, b_curr) = eval_dru_loss(circuit, &scales, alphas, &opt_subset);
        let z = circuit.z12(&FEATS_10640, (&scales, alphas), &mut state);
        let pred_lp_10640 = linear_predict(&w_curr, &b_curr, &z)[0];

        let grad = central_difference(&scales, |s| {
            eval_dru_loss(circuit, s, alphas, &opt_subset).0
        });
        adam.step(&mut scales, &grad, epoch);
        print_angle_row(epoch, loss_curr, pred_lp_10640, &scales);
    }
    report::print_scale_footer(t_opt.elapsed(), &scales);
    scales
}

/// Phase 3: 12-observable readout with re-uploading and feed-forward.
pub fn reuploading(
    circuit: &Circuit,
    data: &Data,
    angles: Angles,
    base: &Fit,
    state: &mut QuantumState,
) -> Fit {
    phase_header(" [PHASE 3] EVALUATING DATA RE-UPLOADING (12 Observables: DRU + FF)       ");
    let dru = fit_z12(circuit, data, angles, state);
    report::print_reuploading(base, &dru);
    dru
}

/// Phase 4: the unified stack, all three mechanisms with 78 observables.
pub fn unified(
    circuit: &Circuit,
    data: &Data,
    angles: Angles,
    (base, ff, dru): (&Fit, &Fit, &Fit),
    state: &mut QuantumState,
) -> Fit {
    report::print_phase4_intro();
    let x_train = obs78_rows(circuit, &data.train, angles, state);
    let x_test = obs78_rows(circuit, &data.test, angles, state);
    let uni = fit_and_score(data, &x_train, &x_test);
    report::print_ledger([base, ff, dru, &uni]);
    uni
}

/// Phase 5: where molecule #10640 lands at each architecture stage.
pub fn audit(
    circuit: &Circuit,
    (scales, alphas): Angles,
    [base, ff, dru, uni]: [&Fit; 4],
    state: &mut QuantumState,
) -> Audit {
    phase_header(" [PHASE 5] MOLECULE #10640 BORN SOLVATION RECOVERY AUDIT                 ");
    let zeros = vec![0.0; N_QUBITS];
    let lp12 = |fit: &Fit, angles: Angles, state: &mut QuantumState| {
        let z = circuit.z12(&FEATS_10640, angles, state);
        linear_predict(&fit.w, &fit.b, &z)[0]
    };
    let b = lp12(base, (&zeros, &zeros), state);
    let ff_lp = lp12(ff, (&zeros, alphas), state);
    let dru_lp = lp12(dru, (scales, alphas), state);
    let obs = circuit.obs78(&FEATS_10640, (scales, alphas), state);
    let uni_lp = linear_predict(&uni.w, &uni.b, &obs)[0];

    let audit = Audit {
        base: b,
        ff: ff_lp,
        dru: dru_lp,
        uni: uni_lp,
    };
    report::print_audit(&audit);
    audit
}

/// Phase 7: baseline, re-uploading, and unified readout throughput.
pub fn throughput(
    circuit: &Circuit,
    test: &[ZincMolecule],
    angles: Angles,
    state: &mut QuantumState,
) -> Throughput {
    phase_header(" [PHASE 7] THROUGHPUT BENCHMARK                                          ");
    let repeats = 3;
    let baseline = mols_per_sec(test, repeats, |s| {
        circuit.encoding.encode_amplitude_into(&s.features, state);
        circuit
            .ansatz
            .forward_l2_in_place(state, &circuit.params[..48]);
        let mut z = [0.0; N_QUBITS];
        t1000::simd::measure_fused_ry_z_readout(
            &state.real,
            &state.imag,
            &circuit.params[48..60],
            &mut z,
        );
    });
    let reupload = mols_per_sec(test, repeats, |s| {
        circuit.z12(&s.features, angles, state);
    });
    let unified = mols_per_sec(test, repeats, |s| {
        let (mut z, mut zz) = ([0.0; N_QUBITS], [0.0; N_PAIRS]);
        circuit.readout78(&s.features, angles, state, (&mut z, &mut zz));
    });
    let tp = Throughput {
        baseline,
        reupload,
        unified,
    };
    report::print_throughput(&tp);
    tp
}
