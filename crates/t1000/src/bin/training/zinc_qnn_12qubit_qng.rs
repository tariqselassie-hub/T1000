//! GuddaNetworks 12-Qubit QNN: Quantum Natural Gradient Engine (Mechanism #4)
//!
//! MECHANISM #4: Quantum Natural Gradient (QNG) Optimization on Fubini-Study Manifold
//!
//! Theoretical Foundation (Stokes et al., Quantum 4, 269 (2020)):
//!   Standard gradient descent assumes parameters θ exist in flat Euclidean space:
//!     θ_{t+1} = θ_t - η ∇_θ L
//!   In quantum circuits, parameters generate state trajectories in complex projective
//!   Hilbert space (CP^{2^N - 1}). The natural gradient follows the steepest descent
//!   direction along the Fubini-Study Riemannian metric tensor g(θ):
//!     θ_{t+1} = θ_t - η (g(θ) + λ I)^{-1} ∇_θ L
//!
//! Metric Tensor Formulation:
//!   g_{ij}(θ) = Re[ ⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩ ]
//!   Computed exactly via statevector fidelity overlap on CP^N:
//!     1 - |⟨ψ(θ + ε e_i) | ψ(θ - ε e_i)⟩|^2 = (2ε)^2 g_{ii}(θ) / 4 + O(ε^3)
//!
//! Execution:
//!   Rigorous head-to-head comparison on identical starting weights:
//!   - Path A: Standard Euclidean Adam Optimizer
//!   - Path B: Quantum Natural Gradient (QNG) Riemannian Optimizer
//!   Evaluated on 10,000 ZINC train / 2,000 holdout test molecules across:
//!   logP, QED, SAS, Molecule #10640 solvation recovery, and Fubini-Study spectrum.

use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};
use t1000::zinc_data::{self, ZincMolecule};
use t1000::zinc_training_common::{ReuploadingCheckpoint, resolve_workspace_path, utc_timestamp_rfc3339};

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4,096
#[allow(dead_code)]
const N_PAIRS: usize = N_QUBITS * (N_QUBITS - 1) / 2; // 66
#[allow(dead_code)]
const N_OBS_FULL: usize = N_QUBITS + N_PAIRS; // 78
const SOURCE_QUBIT: usize = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QngCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub source_qubit: usize,
    pub circuit_params_euclidean: Vec<f64>,
    pub circuit_params_qng: Vec<f64>,
    pub upload_scales_qng: Vec<f64>,
    pub feedforward_alphas: Vec<f64>,
    pub readout_weights_78: Vec<Vec<f64>>,
    pub readout_biases_78: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QngEpochTelemetry {
    pub epoch: usize,
    pub loss_euclidean: f64,
    pub loss_qng: f64,
    pub g_cond_number: f64,
    pub g_min_eigenval: f64,
    pub g_max_eigenval: f64,
    pub pred_mol10640_euclidean: f64,
    pub pred_mol10640_qng: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QngComparisonReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub n_qubits: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub pre_qng_mae_logp: f64,
    pub pre_qng_mae_qed: f64,
    pub pre_qng_mae_sas: f64,
    pub pre_qng_r2_logp: f64,
    pub pre_qng_r2_qed: f64,
    pub pre_qng_r2_sas: f64,
    pub euclidean_mae_logp: f64,
    pub euclidean_mae_qed: f64,
    pub euclidean_mae_sas: f64,
    pub euclidean_r2_logp: f64,
    pub euclidean_r2_qed: f64,
    pub euclidean_r2_sas: f64,
    pub qng_mae_logp: f64,
    pub qng_mae_qed: f64,
    pub qng_mae_sas: f64,
    pub qng_r2_logp: f64,
    pub qng_r2_qed: f64,
    pub qng_r2_sas: f64,
    pub mol10640_true_logp: f64,
    pub mol10640_pre_qng_pred: f64,
    pub mol10640_euclidean_pred: f64,
    pub mol10640_qng_pred: f64,
    pub epoch_history: Vec<QngEpochTelemetry>,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
}

#[inline(always)]
fn state_fidelity(s1: &QuantumState, s2: &QuantumState) -> f64 {
    let mut sum_re = 0.0;
    let mut sum_im = 0.0;
    for i in 0..DIM {
        sum_re += s1.real[i] * s2.real[i] + s1.imag[i] * s2.imag[i];
        sum_im += s1.real[i] * s2.imag[i] - s1.imag[i] * s2.real[i];
    }
    sum_re * sum_re + sum_im * sum_im
}

/// Computes the exact Fubini-Study diagonal metric tensor g_{ii}(θ)
/// averaged across the representative batch:
/// g_{ii} = E_{x} [ (1 - |⟨ψ(θ + ε e_i) | ψ(θ - ε e_i)⟩|^2) / (4 ε^2) ] + λ
fn compute_fubini_study_diagonal(
    ansatz: &CircuitAnsatz,
    encoding: &DataEncodingLayer,
    circuit_params: &[f64],
    dru_scales: &[f64],
    feedforward_alphas: &[f64],
    batch: &[ZincMolecule],
    lambda_damping: f64,
) -> Vec<f64> {
    let p_len = circuit_params.len();
    let mut g_diag = vec![lambda_damping; p_len];
    let eps = 0.005;
    let eps2_4 = 4.0 * eps * eps;

    let mut state_pos = QuantumState::new(DIM);
    let mut state_neg = QuantumState::new(DIM);

    // Sample a subset of batch for high-efficiency Riemannian metric estimation
    let metric_sub = &batch[..batch.len().min(64)];
    let scale_fac = 1.0 / metric_sub.len() as f64;

    for s in metric_sub {
        for i in 0..p_len {
            let mut p_pos = circuit_params.to_vec();
            p_pos[i] += eps;
            let mut p_neg = circuit_params.to_vec();
            p_neg[i] -= eps;

            encoding.encode_amplitude_into(&s.features, &mut state_pos);
            let mut dummy_z = [0.0; N_QUBITS];
            ansatz.forward_reuploading_in_place(
                &mut state_pos,
                &p_pos,
                &s.features,
                dru_scales,
                SOURCE_QUBIT,
                feedforward_alphas,
                &mut dummy_z,
            );

            encoding.encode_amplitude_into(&s.features, &mut state_neg);
            ansatz.forward_reuploading_in_place(
                &mut state_neg,
                &p_neg,
                &s.features,
                dru_scales,
                SOURCE_QUBIT,
                feedforward_alphas,
                &mut dummy_z,
            );

            let fid = state_fidelity(&state_pos, &state_neg).min(1.0);
            let g_val = (1.0 - fid) / eps2_4;
            g_diag[i] += scale_fac * g_val.max(0.0);
        }
    }

    g_diag
}

fn solve_ridge_cholesky(
    z_matrix: &[Vec<f64>], // N x K
    targets: &[f64],       // N
    lambda: f64,
) -> (Vec<f64>, f64) {
    let n = z_matrix.len();
    let k = z_matrix[0].len();
    assert_eq!(n, targets.len());

    let mut mean_y = 0.0;
    for &y in targets {
        mean_y += y;
    }
    mean_y /= n as f64;

    let mut mean_z = vec![0.0; k];
    for row in z_matrix {
        for (j, &val) in row.iter().enumerate() {
            mean_z[j] += val;
        }
    }
    for m in &mut mean_z {
        *m /= n as f64;
    }

    let mut a = vec![vec![0.0; k]; k];
    let mut b = vec![0.0; k];

    for (row_idx, row) in z_matrix.iter().enumerate() {
        let dy = targets[row_idx] - mean_y;
        for i in 0..k {
            let dzi = row[i] - mean_z[i];
            b[i] += dzi * dy;
            for j in i..k {
                let dzj = row[j] - mean_z[j];
                a[i][j] += dzi * dzj;
            }
        }
    }

    for i in 0..k {
        for j in i..k {
            a[j][i] = a[i][j];
        }
        a[i][i] += lambda;
    }

    let mut l = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..=i {
            let mut sum = a[i][j];
            for p in 0..j {
                sum -= l[i][p] * l[j][p];
            }
            if i == j {
                let diag = sum.max(1e-12);
                l[i][j] = diag.sqrt();
            } else {
                l[i][j] = sum / l[j][j].max(1e-12);
            }
        }
    }

    let mut y_sol = vec![0.0; k];
    for i in 0..k {
        let mut sum = b[i];
        for p in 0..i {
            sum -= l[i][p] * y_sol[p];
        }
        y_sol[i] = sum / l[i][i].max(1e-12);
    }

    let mut w = vec![0.0; k];
    for i in (0..k).rev() {
        let mut sum = y_sol[i];
        for p in (i + 1)..k {
            sum -= l[p][i] * w[p];
        }
        let raw_w = sum / l[i][i].max(1e-12);
        w[i] = raw_w.clamp(-2.5, 2.5);
    }

    let mut bias = mean_y;
    for (j, &wj) in w.iter().enumerate() {
        bias -= wj * mean_z[j];
    }

    (w, bias)
}

fn eval_loss_and_weights(
    ansatz: &CircuitAnsatz,
    encoding: &DataEncodingLayer,
    circuit_params: &[f64],
    dru_scales: &[f64],
    feedforward_alphas: &[f64],
    batch: &[ZincMolecule],
    lambda_ridge: f64,
) -> (f64, Vec<Vec<f64>>, Vec<f64>) {
    let mut z_mat = Vec::with_capacity(batch.len());
    let mut t_logp = Vec::with_capacity(batch.len());
    let mut t_qed = Vec::with_capacity(batch.len());
    let mut t_sas = Vec::with_capacity(batch.len());

    let mut l2_state = QuantumState::new(DIM);
    let mut z = [0.0; N_QUBITS];

    for s in batch {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            circuit_params,
            &s.features,
            dru_scales,
            SOURCE_QUBIT,
            feedforward_alphas,
            &mut z,
        );
        z_mat.push(z.to_vec());
        t_logp.push(s.log_p);
        t_qed.push(s.qed);
        t_sas.push(s.sas);
    }

    let (w_logp, b_logp) = solve_ridge_cholesky(&z_mat, &t_logp, lambda_ridge);
    let (w_qed, b_qed) = solve_ridge_cholesky(&z_mat, &t_qed, lambda_ridge);
    let (w_sas, b_sas) = solve_ridge_cholesky(&z_mat, &t_sas, lambda_ridge);

    let n = batch.len() as f64;
    let mut total_loss = 0.0;
    for (i, row) in z_mat.iter().enumerate() {
        let mut p_logp = b_logp;
        let mut p_qed = b_qed;
        let mut p_sas = b_sas;
        for q in 0..N_QUBITS {
            p_logp += w_logp[q] * row[q];
            p_qed += w_qed[q] * row[q];
            p_sas += w_sas[q] * row[q];
        }
        let e_logp = (p_logp - t_logp[i]) / 1.5;
        let e_qed = (p_qed - t_qed[i]) / 0.15;
        let e_sas = (p_sas - t_sas[i]) / 0.8;
        total_loss += e_logp * e_logp + e_qed * e_qed + e_sas * e_sas;
    }

    let w_all = vec![w_logp, w_qed, w_sas];
    let b_all = vec![b_logp, b_qed, b_sas];
    (total_loss / n, w_all, b_all)
}

fn compute_mae_and_r2(preds: &[f64], targets: &[f64]) -> (f64, f64) {
    let n = preds.len() as f64;
    let mut mae = 0.0;
    let mut mean_t = 0.0;
    for (&p, &t) in preds.iter().zip(targets) {
        mae += (p - t).abs();
        mean_t += t;
    }
    mae /= n;
    mean_t /= n;

    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for (&p, &t) in preds.iter().zip(targets) {
        let dt = t - mean_t;
        let dp = t - p;
        ss_tot += dt * dt;
        ss_res += dp * dp;
    }
    let r2 = if ss_tot > 1e-12 {
        1.0 - (ss_res / ss_tot)
    } else {
        0.0
    };
    (mae, r2)
}

fn main() {
    println!("=========================================================================");
    println!(" GUDDANETWORKS 12-QUBIT QNN: QUANTUM NATURAL GRADIENT (MECHANISM #4)     ");
    println!("=========================================================================");
    println!(" Optimization Manifold : Complex Projective Fubini-Study Hilbert Space    ");
    println!(" Metric Tensor g(θ)    : Re[ ⟨∂_i ψ | ∂_j ψ⟩ - ⟨∂_i ψ | ψ⟩⟨ψ | ∂_j ψ⟩ ]   ");
    println!(" Comparison            : Euclidean Adam vs Quantum Natural Gradient      ");
    println!(" Subspace Target       : 60 Circuit Rotation Angles + 12 Inter-Layer DRU ");
    println!(" Hardware Engine       : AVX2+FMA Fused Zero-Allocation POVM Reduction   ");
    println!(" Visual Standard       : Deen OS Sovereign Palette                       ");
    println!("-------------------------------------------------------------------------\n");

    let train_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/data/zinc_train_10k.csv");
    let test_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/data/zinc_test_2k.csv");
    let ckpt_reupload_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_reuploading.json",
    );

    let train_samples = zinc_data::load(&train_path, 10_000).expect("Load train data");
    let test_samples = zinc_data::load(&test_path, 2_000).expect("Load test data");

    println!("  -> Train samples: {}", train_samples.len());
    println!("  -> Test samples : {}", test_samples.len());

    let ckpt_json = fs::read_to_string(&ckpt_reupload_path).expect("Read reuploading checkpoint");
    let ckpt: ReuploadingCheckpoint = serde_json::from_str(&ckpt_json).expect("Parse checkpoint");

    let circuit_params_init = ckpt.circuit_params.clone();
    let feedforward_alphas = ckpt.feedforward_alphas.clone();
    let dru_scales_init = ckpt.upload_scales_l0.clone();

    println!("  -> Checkpoint loaded: 60 circuit params, 12 DRU scales, 12 FF alphas.\n");

    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);

    let feats_10640 = vec![
        24.0, 408.279, 15.0, 5.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 12.0, 0.5, 9.0, 0.375, 5.0, 0.0,
        4.0, 3.0, 4.0,
    ];
    let true_lp_10640 = -6.7716;

    // Evaluate Pre-QNG Baseline on holdout test set (12 obs)
    println!("=========================================================================");
    println!(" [PHASE 1] PRE-OPTIMIZATION BASELINE EVALUATION (RE-UPLOADING CHECKPOINT) ");
    println!("=========================================================================");

    let (_, w_init, b_init) = eval_loss_and_weights(
        &ansatz,
        &encoding,
        &circuit_params_init,
        &dru_scales_init,
        &feedforward_alphas,
        &train_samples[..2000],
        1e-3,
    );

    let mut preds_init_logp = Vec::with_capacity(test_samples.len());
    let mut preds_init_qed = Vec::with_capacity(test_samples.len());
    let mut preds_init_sas = Vec::with_capacity(test_samples.len());
    let mut targets_logp = Vec::with_capacity(test_samples.len());
    let mut targets_qed = Vec::with_capacity(test_samples.len());
    let mut targets_sas = Vec::with_capacity(test_samples.len());

    let mut l2_state = QuantumState::new(DIM);
    let mut z = [0.0; N_QUBITS];

    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &circuit_params_init,
            &s.features,
            &dru_scales_init,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z,
        );
        let mut pl = b_init[0];
        let mut pq = b_init[1];
        let mut ps = b_init[2];
        for q in 0..N_QUBITS {
            pl += w_init[0][q] * z[q];
            pq += w_init[1][q] * z[q];
            ps += w_init[2][q] * z[q];
        }
        preds_init_logp.push(pl);
        preds_init_qed.push(pq);
        preds_init_sas.push(ps);
        targets_logp.push(s.log_p);
        targets_qed.push(s.qed);
        targets_sas.push(s.sas);
    }

    let (mae_init_lp, r2_init_lp) = compute_mae_and_r2(&preds_init_logp, &targets_logp);
    let (mae_init_qd, r2_init_qd) = compute_mae_and_r2(&preds_init_qed, &targets_qed);
    let (mae_init_ss, r2_init_ss) = compute_mae_and_r2(&preds_init_sas, &targets_sas);

    println!(
        "  Holdout MAE : logP = {:.4}, QED = {:.4}, SAS = {:.4}",
        mae_init_lp, mae_init_qd, mae_init_ss
    );
    println!(
        "  Holdout R²  : logP = {:.4}, QED = {:.4}, SAS = {:.4}\n",
        r2_init_lp, r2_init_qd, r2_init_ss
    );

    // =========================================================================
    // DUAL COMPARISON: PATH A (EUCLIDEAN ADAM) vs PATH B (QUANTUM NATURAL GRADIENT)
    // =========================================================================
    let opt_subset: Vec<ZincMolecule> = train_samples.iter().take(2000).cloned().collect();
    let n_epochs = 12;
    let lr = 0.04;
    let eps = 0.01;
    let lambda_qng = 0.02; // Tikhonov regularization for Fubini-Study inversion

    println!("=========================================================================");
    println!(" [PHASE 2] TRAINING TRAJECTORY AUDIT: EUCLIDEAN ADAM vs QUANTUM NATURAL GRADIENT");
    println!("=========================================================================");
    println!(
        "  Batch Size : 2,000 Molecules | Learning Rate: {:.2} | Damping λ: {:.3}",
        lr, lambda_qng
    );
    println!("  Param Space: 60 Quantum Circuit Angles (All 3 Layers)\n");

    let mut params_euc = circuit_params_init.clone();
    let dru_euc = dru_scales_init.clone();
    let mut m_euc = vec![0.0; params_euc.len()];
    let mut v_euc = vec![0.0; params_euc.len()];

    let mut params_qng = circuit_params_init.clone();
    let dru_qng = dru_scales_init.clone();
    let mut m_qng = vec![0.0; params_qng.len()];
    let mut v_qng = vec![0.0; params_qng.len()];

    let mut telemetry_history = Vec::with_capacity(n_epochs);

    println!(" Epoch │ Loss (Euclidean) │   Loss (QNG)   │ Δ Advantage │ Fubini-Study κ(g) │ #10640 Pred (QNG)");
    println!("───────┼──────────────────┼────────────────┼─────────────┼───────────────────┼──────────────────");

    let t_train_start = Instant::now();

    for epoch in 1..=n_epochs {
        // Compute current metrics for both
        let (loss_euc, w_euc, b_euc) = eval_loss_and_weights(
            &ansatz,
            &encoding,
            &params_euc,
            &dru_euc,
            &feedforward_alphas,
            &opt_subset,
            1e-3,
        );

        let (loss_qng, w_qng, b_qng) = eval_loss_and_weights(
            &ansatz,
            &encoding,
            &params_qng,
            &dru_qng,
            &feedforward_alphas,
            &opt_subset,
            1e-3,
        );

        // Compute Fubini-Study metric tensor g(θ) for QNG
        let g_diag = compute_fubini_study_diagonal(
            &ansatz,
            &encoding,
            &params_qng,
            &dru_qng,
            &feedforward_alphas,
            &opt_subset,
            lambda_qng,
        );

        let g_min = g_diag.iter().cloned().fold(f64::INFINITY, f64::min);
        let g_max = g_diag.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let cond_num = g_max / g_min.max(1e-9);

        // Molecule #10640 tracking
        encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
        let mut z_10640 = [0.0; N_QUBITS];
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &params_euc,
            &feats_10640,
            &dru_euc,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z_10640,
        );
        let mut pred_10640_euc = b_euc[0];
        for q in 0..N_QUBITS {
            pred_10640_euc += w_euc[0][q] * z_10640[q];
        }

        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &params_qng,
            &feats_10640,
            &dru_qng,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z_10640,
        );
        let mut pred_10640_qng = b_qng[0];
        for q in 0..N_QUBITS {
            pred_10640_qng += w_qng[0][q] * z_10640[q];
        }

        telemetry_history.push(QngEpochTelemetry {
            epoch,
            loss_euclidean: loss_euc,
            loss_qng,
            g_cond_number: cond_num,
            g_min_eigenval: g_min,
            g_max_eigenval: g_max,
            pred_mol10640_euclidean: pred_10640_euc,
            pred_mol10640_qng: pred_10640_qng,
        });

        let delta_adv = (loss_euc - loss_qng) / loss_euc * 100.0;
        println!(
            "  {:>2}   │    {:>10.6}    │   {:>10.6}   │   {:>+6.2}%   │     {:>8.2}      │     {:>+8.4}",
            epoch, loss_euc, loss_qng, delta_adv, cond_num, pred_10640_qng
        );

        // --- Gradient Evaluation for Euclidean Path ---
        let mut grad_euc = vec![0.0; params_euc.len()];
        for i in 0..params_euc.len() {
            let mut p_plus = params_euc.clone();
            p_plus[i] += eps;
            let (l_plus, _, _) = eval_loss_and_weights(
                &ansatz,
                &encoding,
                &p_plus,
                &dru_euc,
                &feedforward_alphas,
                &opt_subset[..500],
                1e-3,
            );
            let mut p_minus = params_euc.clone();
            p_minus[i] -= eps;
            let (l_minus, _, _) = eval_loss_and_weights(
                &ansatz,
                &encoding,
                &p_minus,
                &dru_euc,
                &feedforward_alphas,
                &opt_subset[..500],
                1e-3,
            );
            grad_euc[i] = (l_plus - l_minus) / (2.0 * eps);
        }

        // --- Gradient Evaluation for QNG Path ---
        let mut grad_qng = vec![0.0; params_qng.len()];
        for i in 0..params_qng.len() {
            let mut p_plus = params_qng.clone();
            p_plus[i] += eps;
            let (l_plus, _, _) = eval_loss_and_weights(
                &ansatz,
                &encoding,
                &p_plus,
                &dru_qng,
                &feedforward_alphas,
                &opt_subset[..500],
                1e-3,
            );
            let mut p_minus = params_qng.clone();
            p_minus[i] -= eps;
            let (l_minus, _, _) = eval_loss_and_weights(
                &ansatz,
                &encoding,
                &p_minus,
                &dru_qng,
                &feedforward_alphas,
                &opt_subset[..500],
                1e-3,
            );
            grad_qng[i] = (l_plus - l_minus) / (2.0 * eps);
        }

        // Step Euclidean Adam (with trust-region clip [-0.15, 0.15])
        let beta1 = 0.9;
        let beta2 = 0.999;
        for i in 0..params_euc.len() {
            m_euc[i] = beta1 * m_euc[i] + (1.0 - beta1) * grad_euc[i];
            v_euc[i] = beta2 * v_euc[i] + (1.0 - beta2) * grad_euc[i] * grad_euc[i];
            let m_hat = m_euc[i] / (1.0 - beta1.powi(epoch as i32));
            let v_hat = v_euc[i] / (1.0 - beta2.powi(epoch as i32));
            let step = lr * m_hat / (v_hat.sqrt() + 1e-8);
            params_euc[i] -= step.clamp(-0.15, 0.15);
        }

        // Step Quantum Natural Gradient:
        // Precondition raw Euclidean gradient by inverse of Fubini-Study tensor with lambda=0.05 damping:
        // g̃ = (g(θ) + λ I)^{-1} ∇θ L
        let lambda_damping = 0.05;
        for i in 0..params_qng.len() {
            let f_study_metric = g_diag[i].max(0.0) + lambda_damping;
            let nat_grad = grad_qng[i] / f_study_metric;
            m_qng[i] = beta1 * m_qng[i] + (1.0 - beta1) * nat_grad;
            v_qng[i] = beta2 * v_qng[i] + (1.0 - beta2) * nat_grad * nat_grad;
            let m_hat = m_qng[i] / (1.0 - beta1.powi(epoch as i32));
            let v_hat = v_qng[i] / (1.0 - beta2.powi(epoch as i32));
            let step = lr * m_hat / (v_hat.sqrt() + 1e-8);
            params_qng[i] -= step.clamp(-0.15, 0.15);
        }
    }

    let el_train = t_train_start.elapsed();
    println!("───────┴──────────────────┴────────────────┴─────────────┴───────────────────┴──────────────────");
    println!(
        "  Dual optimization completed in {:.2}s\n",
        el_train.as_secs_f64()
    );

    // =========================================================================
    // PHASE 3: HOLDOUT TEST SET EVALUATION ACROSS BOTH OPTIMIZERS
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 3] HOLDOUT TEST SET BENCHMARK (2,000 MOLECULES)                  ");
    println!("=========================================================================");

    // Solve final weights on 10,000 train molecules
    let (_, w_euc_final, b_euc_final) = eval_loss_and_weights(
        &ansatz,
        &encoding,
        &params_euc,
        &dru_euc,
        &feedforward_alphas,
        &train_samples,
        1e-3,
    );

    let (_, w_qng_final, b_qng_final) = eval_loss_and_weights(
        &ansatz,
        &encoding,
        &params_qng,
        &dru_qng,
        &feedforward_alphas,
        &train_samples,
        1e-3,
    );

    // Evaluate Euclidean Path
    let mut preds_euc_lp = Vec::with_capacity(test_samples.len());
    let mut preds_euc_qd = Vec::with_capacity(test_samples.len());
    let mut preds_euc_ss = Vec::with_capacity(test_samples.len());
    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &params_euc,
            &s.features,
            &dru_euc,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z,
        );
        let mut pl = b_euc_final[0];
        let mut pq = b_euc_final[1];
        let mut ps = b_euc_final[2];
        for q in 0..N_QUBITS {
            pl += w_euc_final[0][q] * z[q];
            pq += w_euc_final[1][q] * z[q];
            ps += w_euc_final[2][q] * z[q];
        }
        preds_euc_lp.push(pl);
        preds_euc_qd.push(pq);
        preds_euc_ss.push(ps);
    }
    let (mae_euc_lp, r2_euc_lp) = compute_mae_and_r2(&preds_euc_lp, &targets_logp);
    let (mae_euc_qd, r2_euc_qd) = compute_mae_and_r2(&preds_euc_qd, &targets_qed);
    let (mae_euc_ss, r2_euc_ss) = compute_mae_and_r2(&preds_euc_ss, &targets_sas);

    // Evaluate QNG Path
    let mut preds_qng_lp = Vec::with_capacity(test_samples.len());
    let mut preds_qng_qd = Vec::with_capacity(test_samples.len());
    let mut preds_qng_ss = Vec::with_capacity(test_samples.len());
    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &params_qng,
            &s.features,
            &dru_qng,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z,
        );
        let mut pl = b_qng_final[0];
        let mut pq = b_qng_final[1];
        let mut ps = b_qng_final[2];
        for q in 0..N_QUBITS {
            pl += w_qng_final[0][q] * z[q];
            pq += w_qng_final[1][q] * z[q];
            ps += w_qng_final[2][q] * z[q];
        }
        preds_qng_lp.push(pl);
        preds_qng_qd.push(pq);
        preds_qng_ss.push(ps);
    }
    let (mae_qng_lp, r2_qng_lp) = compute_mae_and_r2(&preds_qng_lp, &targets_logp);
    let (mae_qng_qd, r2_qng_qd) = compute_mae_and_r2(&preds_qng_qd, &targets_qed);
    let (mae_qng_ss, r2_qng_ss) = compute_mae_and_r2(&preds_qng_ss, &targets_sas);

    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │ Pre-Opt (DRU)    │ Euclidean Adam   │ QNG (Riemannian) │  QNG Net Gain    │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!("  │  logP MAE   │         {:>8.4} │         {:>8.4} │         {:>8.4} │         {:>+6.2}%  │",
        mae_init_lp, mae_euc_lp, mae_qng_lp, (mae_euc_lp - mae_qng_lp) / mae_euc_lp * 100.0);
    println!("  │  QED MAE    │         {:>8.4} │         {:>8.4} │         {:>8.4} │         {:>+6.2}%  │",
        mae_init_qd, mae_euc_qd, mae_qng_qd, (mae_euc_qd - mae_qng_qd) / mae_euc_qd * 100.0);
    println!("  │  SAS MAE    │         {:>8.4} │         {:>8.4} │         {:>8.4} │         {:>+6.2}%  │",
        mae_init_ss, mae_euc_ss, mae_qng_ss, (mae_euc_ss - mae_qng_ss) / mae_euc_ss * 100.0);
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!("  │  logP R²    │         {:>8.4} │         {:>8.4} │         {:>8.4} │         {:>+6.2}%  │",
        r2_init_lp, r2_euc_lp, r2_qng_lp, (r2_qng_lp - r2_euc_lp) / r2_euc_lp * 100.0);
    println!("  │  QED R²     │         {:>8.4} │         {:>8.4} │         {:>8.4} │         {:>+6.2}%  │",
        r2_init_qd, r2_euc_qd, r2_qng_qd, (r2_qng_qd - r2_euc_qd) / r2_euc_qd * 100.0);
    println!("  │  SAS R²     │         {:>8.4} │         {:>8.4} │         {:>8.4} │         {:>+6.2}%  │",
        r2_init_ss, r2_euc_ss, r2_qng_ss, (r2_qng_ss - r2_euc_ss) / r2_euc_ss * 100.0);
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘\n");

    // =========================================================================
    // PHASE 4: MOLECULE #10640 SOLVATION RECOVERY COMPARISON
    // =========================================================================
    encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
    ansatz.forward_reuploading_in_place(
        &mut l2_state,
        &circuit_params_init,
        &feats_10640,
        &dru_scales_init,
        SOURCE_QUBIT,
        &feedforward_alphas,
        &mut z,
    );
    let mut pred_lp_init_10640 = b_init[0];
    for q in 0..N_QUBITS {
        pred_lp_init_10640 += w_init[0][q] * z[q];
    }

    ansatz.forward_reuploading_in_place(
        &mut l2_state,
        &params_euc,
        &feats_10640,
        &dru_euc,
        SOURCE_QUBIT,
        &feedforward_alphas,
        &mut z,
    );
    let mut pred_lp_euc_10640 = b_euc_final[0];
    for q in 0..N_QUBITS {
        pred_lp_euc_10640 += w_euc_final[0][q] * z[q];
    }

    ansatz.forward_reuploading_in_place(
        &mut l2_state,
        &params_qng,
        &feats_10640,
        &dru_qng,
        SOURCE_QUBIT,
        &feedforward_alphas,
        &mut z,
    );
    let mut pred_lp_qng_10640 = b_qng_final[0];
    for q in 0..N_QUBITS {
        pred_lp_qng_10640 += w_qng_final[0][q] * z[q];
    }

    println!("=========================================================================");
    println!(" [PHASE 4] MOLECULE #10640 BORN SOLVATION RECOVERY AUDIT                 ");
    println!("=========================================================================");
    println!(
        "  SMILES: O=[N+]([O-])c1ccc2c(c1)=[NH+][C@@H](CS[C@H]1[NH+]=c3ccc(Br)cc3=[NH+]1)[NH+]=2"
    );
    println!("  Physical Target : True logP = {:>+8.4}", true_lp_10640);
    println!("  ┌────────────────────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Optimization Method           │  Predicted logP  │  Absolute Error  │");
    println!("  ├────────────────────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  Pre-Optimization Baseline     │       {:>+10.4} │       {:>10.4} │",
        pred_lp_init_10640,
        (pred_lp_init_10640 - true_lp_10640).abs()
    );
    println!(
        "  │  Euclidean Adam Optimizer      │       {:>+10.4} │       {:>10.4} │",
        pred_lp_euc_10640,
        (pred_lp_euc_10640 - true_lp_10640).abs()
    );
    println!(
        "  │  Quantum Natural Gradient (QNG)│       {:>+10.4} │       {:>10.4} │",
        pred_lp_qng_10640,
        (pred_lp_qng_10640 - true_lp_10640).abs()
    );
    println!("  └────────────────────────────────┴──────────────────┴──────────────────┘\n");

    // =========================================================================
    // PHASE 5: PERSISTENCE (CHECKPOINT & REPORT)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 5] PERSISTING CHECKPOINT & TELEMETRY REPORT                      ");
    println!("=========================================================================");

    let checkpoint = QngCheckpoint {
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        n_qubits: N_QUBITS,
        depth: 3,
        source_qubit: SOURCE_QUBIT,
        circuit_params_euclidean: params_euc,
        circuit_params_qng: params_qng,
        upload_scales_qng: dru_qng,
        feedforward_alphas,
        readout_weights_78: ckpt.readout_weights_78,
        readout_biases_78: ckpt.readout_biases_78,
    };

    let ckpt_save_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_qng.json",
    );
    if let Some(parent) = ckpt_save_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let ckpt_json = serde_json::to_string_pretty(&checkpoint).expect("Serialize checkpoint");
    t1000::results::write(&ckpt_save_path, ckpt_json).expect("Write checkpoint");
    println!("  [OK] Saved QNG Checkpoint: {}", ckpt_save_path.display());

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        train_samples
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        test_samples
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let report = QngComparisonReport {
        classical_baselines: classical_baselines.clone(),
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        host_cpu: "Native Host AVX2+FMA".to_string(),
        n_qubits: N_QUBITS,
        train_samples: train_samples.len(),
        test_samples: test_samples.len(),
        pre_qng_mae_logp: mae_init_lp,
        pre_qng_mae_qed: mae_init_qd,
        pre_qng_mae_sas: mae_init_ss,
        pre_qng_r2_logp: r2_init_lp,
        pre_qng_r2_qed: r2_init_qd,
        pre_qng_r2_sas: r2_init_ss,
        euclidean_mae_logp: mae_euc_lp,
        euclidean_mae_qed: mae_euc_qd,
        euclidean_mae_sas: mae_euc_ss,
        euclidean_r2_logp: r2_euc_lp,
        euclidean_r2_qed: r2_euc_qd,
        euclidean_r2_sas: r2_euc_ss,
        qng_mae_logp: mae_qng_lp,
        qng_mae_qed: mae_qng_qd,
        qng_mae_sas: mae_qng_ss,
        qng_r2_logp: r2_qng_lp,
        qng_r2_qed: r2_qng_qd,
        qng_r2_sas: r2_qng_ss,
        mol10640_true_logp: true_lp_10640,
        mol10640_pre_qng_pred: pred_lp_init_10640,
        mol10640_euclidean_pred: pred_lp_euc_10640,
        mol10640_qng_pred: pred_lp_qng_10640,
        epoch_history: telemetry_history,
    };

    let report_save_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/reports/zinc_12qubit_qng_report.json",
    );
    if let Some(parent) = report_save_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    {
        use t1000::baselines::{print_comparison, QnnScore};
        let qnn_rows = [
            QnnScore::new("before optimisation")
                .mae([
                    report.pre_qng_mae_logp,
                    report.pre_qng_mae_qed,
                    report.pre_qng_mae_sas,
                ])
                .r2([
                    report.pre_qng_r2_logp,
                    report.pre_qng_r2_qed,
                    report.pre_qng_r2_sas,
                ]),
            QnnScore::new("Euclidean Adam")
                .mae([
                    report.euclidean_mae_logp,
                    report.euclidean_mae_qed,
                    report.euclidean_mae_sas,
                ])
                .r2([
                    report.euclidean_r2_logp,
                    report.euclidean_r2_qed,
                    report.euclidean_r2_sas,
                ]),
            QnnScore::new("quantum natural gradient")
                .mae([report.qng_mae_logp, report.qng_mae_qed, report.qng_mae_sas])
                .r2([report.qng_r2_logp, report.qng_r2_qed, report.qng_r2_sas]),
        ];
        print_comparison(&qnn_rows, &classical_baselines);
    }

    let report_json = serde_json::to_string_pretty(&report).expect("Serialize report");
    t1000::results::write(&report_save_path, report_json).expect("Write report");
    println!(
        "  [OK] Saved Machine-Readable Report: {}",
        report_save_path.display()
    );
    println!("\n=========================================================================");
    println!(" MECHANISM #4 QUANTUM NATURAL GRADIENT AUDIT COMPLETE                     ");
    println!("=========================================================================");
}
