use super::report;
use super::*;
use std::time::Instant;
use t1000::zinc_training_common::{compute_mae_and_r2, solve_ridge_regression};

/// Descriptor of the anomalous molecule #10640.
pub const FEATS_10640: [f64; 19] = [
    24.0, 408.279, 15.0, 5.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 12.0, 0.5, 9.0, 0.375, 5.0, 0.0, 4.0,
    3.0, 4.0,
];
pub const TRUE_LP_10640: f64 = -6.7716;
const RIDGE: f64 = 1e-3;

/// A ridge readout head and its holdout predictions and scores.
pub struct Fit {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub preds: Vec<Vec<f64>>,
    pub mae: [f64; 3],
    pub r2: [f64; 3],
}

/// The no-feed-forward fit plus the anomaly's baseline prediction and mid-circuit signal.
pub struct Baseline {
    pub fit: Fit,
    pub pred_lp_10640: f64,
    pub m8_10640: f64,
}

/// Predicted logP of molecule #10640 at the feed-forward and combined stages.
pub struct Audit {
    pub ff_lp: f64,
    pub comb_lp: f64,
}

/// Molecules per second of each readout.
pub struct Throughput {
    pub baseline: f64,
    pub feedforward: f64,
    pub combined: f64,
}

fn z12_rows(
    circuit: &Circuit,
    samples: &[ZincMolecule],
    alphas: &[f64],
    state: &mut QuantumState,
) -> Vec<Vec<f64>> {
    samples
        .iter()
        .map(|s| circuit.z12(&s.features, alphas, state).0.to_vec())
        .collect()
}

fn obs78_rows(
    circuit: &Circuit,
    samples: &[ZincMolecule],
    alphas: &[f64],
    state: &mut QuantumState,
) -> Vec<Vec<f64>> {
    samples
        .iter()
        .map(|s| circuit.obs78(&s.features, alphas, state))
        .collect()
}

/// Ridge fit on the train rows, scored on the holdout rows.
fn fit_and_score(data: &Data, x_train: &[Vec<f64>], x_test: &[Vec<f64>]) -> Fit {
    let (w, b) = solve_ridge_regression(x_train, &data.y_train, RIDGE);
    let preds: Vec<Vec<f64>> = x_test
        .iter()
        .map(|x| linear_predict(&w, &b, x).to_vec())
        .collect();
    let (mae, r2) = compute_mae_and_r2(&preds, &data.y_test);
    Fit {
        w,
        b,
        preds,
        mae,
        r2,
    }
}

/// Phase 1: readout fit with zero feed-forward (α = 0), plus the anomaly check.
pub fn baseline(circuit: &Circuit, data: &Data, state: &mut QuantumState) -> Baseline {
    report::phase_header(
        " [PHASE 1] BASELINE QUANTUM EVALUATION (Zero Feed-Forward: α = 0)        ",
    );
    let alpha_zero = vec![0.0; N_QUBITS];
    let x_train = z12_rows(circuit, &data.train, &alpha_zero, state);
    let x_test = z12_rows(circuit, &data.test, &alpha_zero, state);
    let fit = fit_and_score(data, &x_train, &x_test);
    report::print_baseline(&fit);

    let (z, m8_10640) = circuit.z12(&FEATS_10640, &alpha_zero, state);
    let pred_lp_10640 = linear_predict(&fit.w, &fit.b, &z)[0];
    report::print_anomaly_baseline(pred_lp_10640, m8_10640);
    Baseline {
        fit,
        pred_lp_10640,
        m8_10640,
    }
}

/// Composite MSE loss (logP 1.0, QED 5.0, SAS 1.0) of the ridge head refit for `alphas`.
fn eval_alpha_loss(
    circuit: &Circuit,
    alphas: &[f64],
    samples: &[ZincMolecule],
    lambda: f64,
) -> (f64, Vec<Vec<f64>>, Vec<f64>) {
    let mut state = QuantumState::new(DIM);
    let x_mat = z12_rows(circuit, samples, alphas, &mut state);
    let y_mat: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| vec![s.log_p, s.qed, s.sas])
        .collect();

    let (w, b) = solve_ridge_regression(&x_mat, &y_mat, lambda);

    let mut total_loss = 0.0;
    for (x_row, y_row) in x_mat.iter().zip(&y_mat) {
        let p = linear_predict(&w, &b, x_row);
        let e_lp = p[0] - y_row[0];
        let e_qed = p[1] - y_row[1];
        let e_sas = p[2] - y_row[2];
        // Calibrated weights: 1.0 for logP, 5.0 for QED, 1.0 for SAS
        total_loss += 1.0 * (e_lp * e_lp) + 5.0 * (e_qed * e_qed) + 1.0 * (e_sas * e_sas);
    }
    total_loss /= samples.len() as f64;

    (total_loss, w, b)
}

const N_EPOCHS: usize = 12;
const LR: f64 = 0.08;
const FD_EPS: f64 = 0.01; // Finite difference step
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;

/// Adam moment estimates for the feed-forward angles.
struct Adam {
    m: Vec<f64>,
    v: Vec<f64>,
}

impl Adam {
    fn step(&mut self, alphas: &mut [f64], grad: &[f64], epoch: usize) {
        for q in 0..N_QUBITS {
            self.m[q] = BETA1 * self.m[q] + (1.0 - BETA1) * grad[q];
            self.v[q] = BETA2 * self.v[q] + (1.0 - BETA2) * grad[q] * grad[q];
            let m_hat = self.m[q] / (1.0 - BETA1.powi(epoch as i32));
            let v_hat = self.v[q] / (1.0 - BETA2.powi(epoch as i32));
            alphas[q] -= LR * m_hat / (v_hat.sqrt() + 1e-8);
            // Clamp to avoid angle wrap-around
            alphas[q] = alphas[q].clamp(-std::f64::consts::PI, std::f64::consts::PI);
        }
    }
}

/// Central finite-difference gradient of the composite loss with respect to each α_q.
fn central_gradient(circuit: &Circuit, alphas: &[f64], subset: &[ZincMolecule]) -> Vec<f64> {
    let mut grad = vec![0.0; N_QUBITS];
    for q in 0..N_QUBITS {
        let mut alphas_plus = alphas.to_vec();
        alphas_plus[q] += FD_EPS;
        let (l_plus, _, _) = eval_alpha_loss(circuit, &alphas_plus, subset, RIDGE);

        let mut alphas_minus = alphas.to_vec();
        alphas_minus[q] -= FD_EPS;
        let (l_minus, _, _) = eval_alpha_loss(circuit, &alphas_minus, subset, RIDGE);

        grad[q] = (l_plus - l_minus) / (2.0 * FD_EPS);
    }
    grad
}

/// Phase 2: optimise the 12 feed-forward angles with Adam over a nested ridge fit.
pub fn train_alphas(circuit: &Circuit, train: &[ZincMolecule]) -> Vec<f64> {
    report::print_phase2_intro();
    let mut alphas = vec![0.0; N_QUBITS];
    // Seed with physics-motivated initialization: counter-rotate neighbors of Qubit 8
    alphas[7] = 0.25;
    alphas[8] = -0.40;
    alphas[9] = 0.25;

    // Representative subset for fast gradient calculation
    let opt_subset: Vec<ZincMolecule> = train.iter().take(2_000).cloned().collect();
    let mut adam = Adam {
        m: vec![0.0; N_QUBITS],
        v: vec![0.0; N_QUBITS],
    };
    let mut state = QuantumState::new(DIM);
    let t_opt = Instant::now();

    report::print_alpha_table_header();
    for epoch in 1..=N_EPOCHS {
        let (loss_curr, w_curr, b_curr) = eval_alpha_loss(circuit, &alphas, &opt_subset, RIDGE);
        // Molecule #10640 prediction at the current alpha
        let (z, _) = circuit.z12(&FEATS_10640, &alphas, &mut state);
        let pred_lp_10640 = linear_predict(&w_curr, &b_curr, &z)[0];

        let grad = central_gradient(circuit, &alphas, &opt_subset);
        adam.step(&mut alphas, &grad, epoch);
        report::print_alpha_row(epoch, loss_curr, pred_lp_10640, &alphas);
    }
    report::print_alpha_footer(t_opt.elapsed(), &alphas);
    alphas
}

/// Phase 3: 12-observable readout under the trained feed-forward angles.
pub fn feedforward(
    circuit: &Circuit,
    data: &Data,
    alphas: &[f64],
    base: &Fit,
    state: &mut QuantumState,
) -> Fit {
    report::phase_header(
        " [PHASE 3] EVALUATING MID-CIRCUIT FEED-FORWARD (12 Observables)          ",
    );
    // Fit final readout weights on the full 10,000 training set
    let x_train = z12_rows(circuit, &data.train, alphas, state);
    let x_test = z12_rows(circuit, &data.test, alphas, state);
    let ff = fit_and_score(data, &x_train, &x_test);
    report::print_feedforward(base, &ff);
    ff
}

/// Phase 4: feed-forward combined with the 66 two-body ZZ correlators (78 observables).
pub fn combined(
    circuit: &Circuit,
    data: &Data,
    alphas: &[f64],
    base: &Fit,
    ff: &Fit,
    state: &mut QuantumState,
) -> Fit {
    report::print_phase4_intro();
    let x_train = obs78_rows(circuit, &data.train, alphas, state);
    let x_test = obs78_rows(circuit, &data.test, alphas, state);
    let comb = fit_and_score(data, &x_train, &x_test);
    report::print_ledger(base, ff, &comb);
    comb
}

/// Phase 5: where molecule #10640 lands at each architecture stage.
pub fn audit(
    circuit: &Circuit,
    alphas: &[f64],
    (base, ff, comb): (&Baseline, &Fit, &Fit),
    state: &mut QuantumState,
) -> Audit {
    report::phase_header(
        " [PHASE 5] MOLECULE #10640 RESOLUTION AUDIT (MAX QUANTUM ANOMALY)        ",
    );
    let (z_ff, _) = circuit.z12(&FEATS_10640, alphas, state);
    let ff_lp = linear_predict(&ff.w, &ff.b, &z_ff)[0];
    let obs = circuit.obs78(&FEATS_10640, alphas, state);
    let comb_lp = linear_predict(&comb.w, &comb.b, &obs)[0];
    report::print_audit(base, ff_lp, comb_lp);
    Audit { ff_lp, comb_lp }
}

/// Molecules per second over `repeats` passes of the holdout set.
fn mols_per_sec(test: &[ZincMolecule], repeats: usize, mut run: impl FnMut(&ZincMolecule)) -> f64 {
    let t = Instant::now();
    for _ in 0..repeats {
        test.iter().for_each(&mut run);
    }
    (test.len() * repeats) as f64 / t.elapsed().as_secs_f64()
}

/// Phase 7: baseline, feed-forward, and combined readout throughput.
pub fn throughput(
    circuit: &Circuit,
    test: &[ZincMolecule],
    alphas: &[f64],
    state: &mut QuantumState,
) -> Throughput {
    report::phase_header(
        " [PHASE 7] THROUGHPUT BENCHMARK                                          ",
    );
    let repeats = 3;
    let baseline = mols_per_sec(test, repeats, |sample| {
        circuit
            .encoding
            .encode_amplitude_into(&sample.features, state);
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
    let feedforward = mols_per_sec(test, repeats, |sample| {
        circuit.z12(&sample.features, alphas, state);
    });
    let combined = mols_per_sec(test, repeats, |sample| {
        let (mut z, mut zz) = ([0.0; N_QUBITS], [0.0; N_PAIRS]);
        circuit.readout78(&sample.features, alphas, state, &mut z, &mut zz);
    });
    let tp = Throughput {
        baseline,
        feedforward,
        combined,
    };
    report::print_throughput(&tp);
    tp
}
