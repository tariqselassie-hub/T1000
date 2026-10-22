//! GuddaNetworks 12-Qubit QNN: Data Re-Uploading Engine (Mechanism #3)
//!
//! MECHANISM #3: Inter-layer classical feature re-encoding between Layer 0 and Layer 1:
//!   |ψ_1⟩ = U_CZ · [ ⊗_q R_y(θ_{1y, q}) R_z(θ_{1z, q}) R_y(s_q · x̃_q) ] · |ψ_0⟩
//!
//! Theoretical Foundation (Pérez-Salinas et al. 2020, Schuld et al. 2021):
//!   Single-upload unitary circuits are restricted to degree-2 quadratic forms / degree-1
//!   Fourier harmonics. Interleaving input descriptor re-uploading between parameterized
//!   layers transforms the QNN into a Universal Function Approximator by multiplying the
//!   accessible Fourier frequency spectrum: ω ∈ {-L, ..., +L}.
//!
//! Combined Unified Stack:
//!   Mechanism #1 : 66 Two-Body Ising Correlators ⟨Z_i Z_j⟩
//!   Mechanism #2 : Mid-Circuit Feed-Forward from Qubit 8 (θ̃_{y,q} = θ_{y,q} + α_q · ⟨Z_8⟩_L1)
//!   Mechanism #3 : Inter-Layer Data Re-Uploading (s_q · x̃_q before Layer 1 CZ block)

use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use t1000::zinc_data::{self, ZincMolecule};
use t1000::zinc_training_common::{
    compute_mae_and_r2, resolve_workspace_path, solve_ridge_regression, utc_timestamp_rfc3339,
    MidCircuitCheckpoint, PrunedCheckpoint, ReuploadingCheckpoint,
};
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4,096
const N_PAIRS: usize = N_QUBITS * (N_QUBITS - 1) / 2; // 66
const N_OBS_FULL: usize = N_QUBITS + N_PAIRS; // 78
const SOURCE_QUBIT: usize = 8; // Qubit 8 exhibiting severe spin freeze on anomalous molecules

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReuploadingReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub n_qubits: usize,
    pub source_qubit: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub feedforward_alphas: Vec<f64>,
    pub upload_scales_l0: Vec<f64>,
    pub baseline_mae_logp: f64,
    pub baseline_mae_qed: f64,
    pub baseline_mae_sas: f64,
    pub baseline_r2_logp: f64,
    pub baseline_r2_qed: f64,
    pub baseline_r2_sas: f64,
    pub feedforward_mae_logp: f64,
    pub feedforward_mae_qed: f64,
    pub feedforward_mae_sas: f64,
    pub feedforward_r2_logp: f64,
    pub feedforward_r2_qed: f64,
    pub feedforward_r2_sas: f64,
    pub reuploading_mae_logp: f64,
    pub reuploading_mae_qed: f64,
    pub reuploading_mae_sas: f64,
    pub reuploading_r2_logp: f64,
    pub reuploading_r2_qed: f64,
    pub reuploading_r2_sas: f64,
    pub unified_mae_logp: f64,
    pub unified_mae_qed: f64,
    pub unified_mae_sas: f64,
    pub unified_r2_logp: f64,
    pub unified_r2_qed: f64,
    pub unified_r2_sas: f64,
    pub mol10640_true_logp: f64,
    pub mol10640_baseline_pred_logp: f64,
    pub mol10640_feedforward_pred_logp: f64,
    pub mol10640_reuploading_pred_logp: f64,
    pub mol10640_unified_pred_logp: f64,
    pub baseline_throughput_mols_per_sec: f64,
    pub reuploading_throughput_mols_per_sec: f64,
    pub unified_throughput_mols_per_sec: f64,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
}

/// Evaluates composite MSE loss for a candidate DRU scales vector:
fn eval_dru_loss(
    ansatz: &CircuitAnsatz,
    encoding: &DataEncodingLayer,
    circuit_params: &[f64],
    dru_scales: &[f64],
    alphas: &[f64],
    samples: &[ZincMolecule],
    lambda: f64,
) -> (f64, Vec<Vec<f64>>, Vec<f64>) {
    let mut x_mat = Vec::with_capacity(samples.len());
    let mut y_mat = Vec::with_capacity(samples.len());
    let mut l2_state = QuantumState::new(DIM);

    for s in samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z_obs = [0.0; N_QUBITS];
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            circuit_params,
            &s.features,
            dru_scales,
            SOURCE_QUBIT,
            alphas,
            &mut z_obs,
        );
        x_mat.push(z_obs.to_vec());
        y_mat.push(vec![s.log_p, s.qed, s.sas]);
    }

    let (w, b) = solve_ridge_regression(&x_mat, &y_mat, lambda);

    let mut total_loss = 0.0;
    for (x_row, y_row) in x_mat.iter().zip(&y_mat) {
        let mut p_lp = b[0];
        let mut p_qed = b[1];
        let mut p_sas = b[2];
        for q in 0..N_QUBITS {
            p_lp += w[q][0] * x_row[q];
            p_qed += w[q][1] * x_row[q];
            p_sas += w[q][2] * x_row[q];
        }
        let e_lp = p_lp - y_row[0];
        let e_qed = p_qed - y_row[1];
        let e_sas = p_sas - y_row[2];
        total_loss += 1.0 * (e_lp * e_lp) + 5.0 * (e_qed * e_qed) + 1.0 * (e_sas * e_sas);
    }
    total_loss /= samples.len() as f64;

    (total_loss, w, b)
}

fn main() {
    println!("=========================================================================");
    println!(" GUDDANETWORKS 12-QUBIT QNN: DATA RE-UPLOADING ENGINE (MECHANISM #3)    ");
    println!("=========================================================================");
    println!(" Mode               : Universal Multi-Frequency Harmonic Expansion");
    println!(" Inter-Layer Upload : Stage 1 DRU between Layer 0 and Layer 1 CZ Block");
    println!(" Re-upload Kernel   : R_y(s_q · x̃_q) per qubit q ∈ 0..11");
    println!(
        " Feed-Forward Qubit : Qubit {} (⟨Z_{}⟩ Dynamic Terminal Modulation)",
        SOURCE_QUBIT, SOURCE_QUBIT
    );
    println!(
        " Hilbert Space Dim  : {} (12 Qubits, 64 KB L2 Cache-Resident)",
        DIM
    );
    println!(" Hardware Target    : Bare-Metal AVX2 + FMA Single-Pass Fused POVM");
    println!(" Palette            : Deen OS Sovereign (Quantum Green / Hardware Gold / Alert Red)");
    println!("-------------------------------------------------------------------------\n");

    let pruned_ckpt_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_pruned99.json",
    );
    let midcircuit_ckpt_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_midcircuit.json",
    );
    let train_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/data/zinc_train_10k.csv");
    let test_path = resolve_workspace_path("tests/benchmarks/zinc_molecular/data/zinc_test_2k.csv");

    // Load datasets
    println!("[INIT] Loading training set (10,000 molecules)...");
    let train_samples = match zinc_data::load(&train_path, 10_000) {
        Ok(s) => s,
        Err(e) => {
            let fallback = resolve_workspace_path(
                "tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv",
            );
            match zinc_data::load(&fallback, 10_000) {
                Ok(s) => s,
                Err(_) => {
                    eprintln!("[-] Error: {}", e);
                    return;
                }
            }
        }
    };
    println!("  -> Loaded {} training molecules.\n", train_samples.len());

    println!("[INIT] Loading holdout test set (2,000 molecules)...");
    let test_samples = match zinc_data::load(&test_path, 2_000) {
        Ok(s) => s,
        Err(e) => {
            let fallback =
                resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv");
            match zinc_data::load(&fallback, 2_000) {
                Ok(s) => s,
                Err(_) => {
                    eprintln!("[-] Error: {}", e);
                    return;
                }
            }
        }
    };
    println!("  -> Loaded {} test molecules.\n", test_samples.len());

    // Load circuit parameters
    println!("[INIT] Loading quantum circuit parameters...");
    let circuit_params: Vec<f64>;
    if pruned_ckpt_path.exists() {
        let content = fs::read_to_string(&pruned_ckpt_path).expect("Read pruned checkpoint");
        let ckpt: PrunedCheckpoint =
            serde_json::from_str(&content).expect("Parse pruned checkpoint");
        circuit_params = ckpt.pruned_hybrid_params[..60].to_vec();
        println!("  -> Loaded 60 pruned circuit parameters.");
    } else {
        eprintln!(
            "[-] Pruned checkpoint not found at: {}",
            pruned_ckpt_path.display()
        );
        return;
    }

    // Load feedforward alphas from Mechanism #2
    let feedforward_alphas: Vec<f64>;
    if midcircuit_ckpt_path.exists() {
        let content =
            fs::read_to_string(&midcircuit_ckpt_path).expect("Read midcircuit checkpoint");
        let ckpt: MidCircuitCheckpoint =
            serde_json::from_str(&content).expect("Parse midcircuit checkpoint");
        feedforward_alphas = ckpt.feedforward_alphas;
        println!("  -> Loaded 12 optimized feed-forward alphas from Mechanism #2.");
    } else {
        feedforward_alphas = vec![0.0; N_QUBITS];
        println!("  -> Using default zero feed-forward alphas.");
    }

    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let mut l2_state = QuantumState::new(DIM);

    let feats_10640 = [
        24.0, 408.279, 15.0, 5.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 12.0, 0.5, 9.0, 0.375, 5.0, 0.0,
        4.0, 3.0, 4.0,
    ];
    let true_lp_10640 = -6.7716;

    // =========================================================================
    // PHASE 1: BASELINE & MECHANISM #2 EVALUATION
    // =========================================================================
    println!("\n=========================================================================");
    println!(" [PHASE 1] BASELINE & MECHANISM #2 VERIFICATION                          ");
    println!("=========================================================================");

    let zero_scales = vec![0.0; N_QUBITS];
    let zero_alphas = vec![0.0; N_QUBITS];

    // Baseline (No DRU, No FF)
    let mut x_train_b = Vec::with_capacity(train_samples.len());
    let mut y_train = Vec::with_capacity(train_samples.len());
    for s in &train_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z = [0.0; N_QUBITS];
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &circuit_params,
            &s.features,
            &zero_scales,
            SOURCE_QUBIT,
            &zero_alphas,
            &mut z,
        );
        x_train_b.push(z.to_vec());
        y_train.push(vec![s.log_p, s.qed, s.sas]);
    }
    let (w_base, b_base) = solve_ridge_regression(&x_train_b, &y_train, 1e-3);

    let mut y_test = Vec::with_capacity(test_samples.len());
    let mut preds_base = Vec::with_capacity(test_samples.len());
    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z = [0.0; N_QUBITS];
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &circuit_params,
            &s.features,
            &zero_scales,
            SOURCE_QUBIT,
            &zero_alphas,
            &mut z,
        );
        y_test.push(vec![s.log_p, s.qed, s.sas]);
        let mut p = [b_base[0], b_base[1], b_base[2]];
        for q in 0..N_QUBITS {
            for k in 0..3 {
                p[k] += w_base[q][k] * z[q];
            }
        }
        preds_base.push(p.to_vec());
    }
    let (mae_b, r2_b) = compute_mae_and_r2(&preds_base, &y_test);

    // Mechanism #2 alone (FF active, DRU = 0)
    let mut x_train_ff = Vec::with_capacity(train_samples.len());
    for s in &train_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z = [0.0; N_QUBITS];
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &circuit_params,
            &s.features,
            &zero_scales,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z,
        );
        x_train_ff.push(z.to_vec());
    }
    let (w_ff, b_ff) = solve_ridge_regression(&x_train_ff, &y_train, 1e-3);

    let mut preds_ff = Vec::with_capacity(test_samples.len());
    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z = [0.0; N_QUBITS];
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &circuit_params,
            &s.features,
            &zero_scales,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z,
        );
        let mut p = [b_ff[0], b_ff[1], b_ff[2]];
        for q in 0..N_QUBITS {
            for k in 0..3 {
                p[k] += w_ff[q][k] * z[q];
            }
        }
        preds_ff.push(p.to_vec());
    }
    let (mae_ff, r2_ff) = compute_mae_and_r2(&preds_ff, &y_test);

    println!(
        "  Baseline MAE: logP = {:.4}, QED = {:.4}, SAS = {:.4} | logP R² = {:.4}",
        mae_b[0], mae_b[1], mae_b[2], r2_b[0]
    );
    println!(
        "  Mech #2  MAE: logP = {:.4}, QED = {:.4}, SAS = {:.4} | logP R² = {:.4}",
        mae_ff[0], mae_ff[1], mae_ff[2], r2_ff[0]
    );

    // =========================================================================
    // PHASE 2: OPTIMIZING DATA RE-UPLOADING SCALES (s_q)
    // =========================================================================
    println!("\n=========================================================================");
    println!(" [PHASE 2] TRAINING DATA RE-UPLOADING SCALING COEFFICIENTS (s ∈ R¹²)    ");
    println!("=========================================================================");
    println!(" Optimizing 12 inter-layer descriptor re-upload scales via Adam...");
    println!(" Re-uploading kernel: R_y(s_q · x̃_q) before Layer 1 CZ entanglement ring\n");

    let mut dru_scales = vec![0.0; N_QUBITS];
    // Seed with feature-proportional initialization
    dru_scales[0] = 0.5; // Heavy atoms
    dru_scales[1] = 0.8; // MW est
    dru_scales[8] = 1.0; // Halogens / Sulfur / Electronegatives
    dru_scales[9] = 0.5;

    let n_train_opt = 2_000;
    let opt_subset: Vec<ZincMolecule> = train_samples.iter().take(n_train_opt).cloned().collect();

    let n_epochs = 12;
    let lr = 0.10;
    let eps = 0.01;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let mut m_adam = vec![0.0; N_QUBITS];
    let mut v_adam = vec![0.0; N_QUBITS];

    let t_opt = Instant::now();

    println!(" Epoch │ Composite Loss │  logP Pred (#10640) │  ||s||_2  │ Top Scales (q: s_q)");
    println!("───────┼────────────────┼─────────────────────┼───────────┼────────────────────────────────────");

    for epoch in 1..=n_epochs {
        let (loss_curr, w_curr, b_curr) = eval_dru_loss(
            &ansatz,
            &encoding,
            &circuit_params,
            &dru_scales,
            &feedforward_alphas,
            &opt_subset,
            1e-3,
        );

        encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
        let mut z_10640_curr = [0.0; N_QUBITS];
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &circuit_params,
            &feats_10640,
            &dru_scales,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z_10640_curr,
        );
        let mut pred_lp_10640_epoch = b_curr[0];
        for q in 0..N_QUBITS {
            pred_lp_10640_epoch += w_curr[q][0] * z_10640_curr[q];
        }

        let mut grad = vec![0.0; N_QUBITS];
        for q in 0..N_QUBITS {
            let mut s_plus = dru_scales.clone();
            s_plus[q] += eps;
            let (l_plus, _, _) = eval_dru_loss(
                &ansatz,
                &encoding,
                &circuit_params,
                &s_plus,
                &feedforward_alphas,
                &opt_subset,
                1e-3,
            );

            let mut s_minus = dru_scales.clone();
            s_minus[q] -= eps;
            let (l_minus, _, _) = eval_dru_loss(
                &ansatz,
                &encoding,
                &circuit_params,
                &s_minus,
                &feedforward_alphas,
                &opt_subset,
                1e-3,
            );

            grad[q] = (l_plus - l_minus) / (2.0 * eps);
        }

        for q in 0..N_QUBITS {
            m_adam[q] = beta1 * m_adam[q] + (1.0 - beta1) * grad[q];
            v_adam[q] = beta2 * v_adam[q] + (1.0 - beta2) * grad[q] * grad[q];
            let m_hat = m_adam[q] / (1.0 - beta1.powi(epoch as i32));
            let v_hat = v_adam[q] / (1.0 - beta2.powi(epoch as i32));
            dru_scales[q] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            dru_scales[q] = dru_scales[q].clamp(-std::f64::consts::PI, std::f64::consts::PI);
        }

        let norm_s = dru_scales.iter().map(|&s| s * s).sum::<f64>().sqrt();
        let mut s_pairs: Vec<(usize, f64)> = dru_scales.iter().cloned().enumerate().collect();
        s_pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        let top_str = format!(
            "q{}: {:>+5.2}, q{}: {:>+5.2}, q{}: {:>+5.2}",
            s_pairs[0].0, s_pairs[0].1, s_pairs[1].0, s_pairs[1].1, s_pairs[2].0, s_pairs[2].1
        );

        println!(
            "  {:>3}  │   {:>10.6}   │      {:>+10.4}     │  {:>7.4}  │ {}",
            epoch, loss_curr, pred_lp_10640_epoch, norm_s, top_str
        );
    }
    let el_opt = t_opt.elapsed();
    println!("───────┴────────────────┴─────────────────────┴───────────┴────────────────────────────────────");
    println!(
        "  DRU Optimization completed in {:.2}s\n",
        el_opt.as_secs_f64()
    );

    println!(" Final Data Re-Uploading Scales (s_q for q ∈ 0..11):");
    print!("   ");
    for q in 0..N_QUBITS {
        print!("q{:02}: {:>+6.3}  ", q, dru_scales[q]);
        if (q + 1) % 6 == 0 {
            print!("\n   ");
        }
    }
    println!();

    // =========================================================================
    // PHASE 3: EVALUATING DATA RE-UPLOADING (12 Observables)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 3] EVALUATING DATA RE-UPLOADING (12 Observables: DRU + FF)       ");
    println!("=========================================================================");

    let mut x_train_dru = Vec::with_capacity(train_samples.len());
    for s in &train_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z = [0.0; N_QUBITS];
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &circuit_params,
            &s.features,
            &dru_scales,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z,
        );
        x_train_dru.push(z.to_vec());
    }
    let (w_dru, b_dru) = solve_ridge_regression(&x_train_dru, &y_train, 1e-3);

    let mut preds_dru = Vec::with_capacity(test_samples.len());
    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z = [0.0; N_QUBITS];
        ansatz.forward_reuploading_in_place(
            &mut l2_state,
            &circuit_params,
            &s.features,
            &dru_scales,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z,
        );
        let mut p = [b_dru[0], b_dru[1], b_dru[2]];
        for q in 0..N_QUBITS {
            for k in 0..3 {
                p[k] += w_dru[q][k] * z[q];
            }
        }
        preds_dru.push(p.to_vec());
    }
    let (mae_dru, r2_dru) = compute_mae_and_r2(&preds_dru, &y_test);

    let imp_dru_lp = (1.0 - mae_dru[0] / mae_b[0]) * 100.0;
    let imp_dru_qed = (1.0 - mae_dru[1] / mae_b[1]) * 100.0;
    let imp_dru_sas = (1.0 - mae_dru[2] / mae_b[2]) * 100.0;

    println!("  Holdout Comparison (12 Observables):");
    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline MAE    │  Re-Upload MAE   │  Improvement     │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  logP       │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[0], mae_dru[0], imp_dru_lp
    );
    println!(
        "  │  QED        │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[1], mae_dru[1], imp_dru_qed
    );
    println!(
        "  │  SAS        │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[2], mae_dru[2], imp_dru_sas
    );
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┘\n");
    println!("  ┌─────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline R²     │  Re-Upload R²    │");
    println!("  ├─────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  logP       │ {:>16.6} │ {:>16.6} │",
        r2_b[0], r2_dru[0]
    );
    println!(
        "  │  QED        │ {:>16.6} │ {:>16.6} │",
        r2_b[1], r2_dru[1]
    );
    println!(
        "  │  SAS        │ {:>16.6} │ {:>16.6} │",
        r2_b[2], r2_dru[2]
    );
    println!("  └─────────────┴──────────────────┴──────────────────┘\n");

    // =========================================================================
    // PHASE 4: UNIFIED FRONTIER STACK (M1: ZZ + M2: FF + M3: DRU = 78 Observables)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 4] UNIFIED FRONTIER STACK (78 Observables: ZZ + FF + DRU)       ");
    println!("=========================================================================");
    println!(" Full integration of all 3 frontier mechanisms into a single quantum pass:\n");

    let mut x_train_uni = Vec::with_capacity(train_samples.len());
    for s in &train_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z = [0.0; N_QUBITS];
        let mut zz = [0.0; N_PAIRS];
        ansatz.forward_reuploading_full_readout_in_place(
            &mut l2_state,
            &circuit_params,
            &s.features,
            &dru_scales,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z,
            &mut zz,
        );
        let mut obs_78 = Vec::with_capacity(N_OBS_FULL);
        obs_78.extend_from_slice(&z);
        obs_78.extend_from_slice(&zz);
        x_train_uni.push(obs_78);
    }
    let (w_uni, b_uni) = solve_ridge_regression(&x_train_uni, &y_train, 1e-3);

    let mut preds_uni = Vec::with_capacity(test_samples.len());
    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z = [0.0; N_QUBITS];
        let mut zz = [0.0; N_PAIRS];
        ansatz.forward_reuploading_full_readout_in_place(
            &mut l2_state,
            &circuit_params,
            &s.features,
            &dru_scales,
            SOURCE_QUBIT,
            &feedforward_alphas,
            &mut z,
            &mut zz,
        );
        let mut obs_78 = Vec::with_capacity(N_OBS_FULL);
        obs_78.extend_from_slice(&z);
        obs_78.extend_from_slice(&zz);
        let mut p = [b_uni[0], b_uni[1], b_uni[2]];
        for q in 0..N_OBS_FULL {
            for k in 0..3 {
                p[k] += w_uni[q][k] * obs_78[q];
            }
        }
        preds_uni.push(p.to_vec());
    }
    let (mae_uni, r2_uni) = compute_mae_and_r2(&preds_uni, &y_test);

    let imp_uni_lp = (1.0 - mae_uni[0] / mae_b[0]) * 100.0;
    let imp_uni_qed = (1.0 - mae_uni[1] / mae_b[1]) * 100.0;
    let imp_uni_sas = (1.0 - mae_uni[2] / mae_b[2]) * 100.0;

    println!("  Quad-Stage Architectural Evolution Ledger:");
    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline (12)   │  Mech #2 (FF)    │  Mech #3 (DRU)   │  Unified (78 Obs)│  Net Gain (%)    │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  logP MAE   │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[0], mae_ff[0], mae_dru[0], mae_uni[0], imp_uni_lp
    );
    println!(
        "  │  QED MAE    │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[1], mae_ff[1], mae_dru[1], mae_uni[1], imp_uni_qed
    );
    println!(
        "  │  SAS MAE    │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[2], mae_ff[2], mae_dru[2], mae_uni[2], imp_uni_sas
    );
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  logP R²    │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        r2_b[0],
        r2_ff[0],
        r2_dru[0],
        r2_uni[0],
        (r2_uni[0] - r2_b[0]) / r2_b[0].abs() * 100.0
    );
    println!(
        "  │  QED R²     │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        r2_b[1],
        r2_ff[1],
        r2_dru[1],
        r2_uni[1],
        (r2_uni[1] - r2_b[1]) / r2_b[1].abs() * 100.0
    );
    println!(
        "  │  SAS R²     │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        r2_b[2],
        r2_ff[2],
        r2_dru[2],
        r2_uni[2],
        (r2_uni[2] - r2_b[2]) / r2_b[2].abs() * 100.0
    );
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘\n");

    // =========================================================================
    // PHASE 5: MOLECULE #10640 BORN SOLVATION AUDIT
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 5] MOLECULE #10640 BORN SOLVATION RECOVERY AUDIT                 ");
    println!("=========================================================================");

    // 1. Baseline pred
    encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
    let mut z_10640_b = [0.0; N_QUBITS];
    ansatz.forward_reuploading_in_place(
        &mut l2_state,
        &circuit_params,
        &feats_10640,
        &zero_scales,
        SOURCE_QUBIT,
        &zero_alphas,
        &mut z_10640_b,
    );
    let mut p_lp_b = b_base[0];
    for q in 0..N_QUBITS {
        p_lp_b += w_base[q][0] * z_10640_b[q];
    }

    // 2. FF pred
    encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
    let mut z_10640_ff = [0.0; N_QUBITS];
    ansatz.forward_reuploading_in_place(
        &mut l2_state,
        &circuit_params,
        &feats_10640,
        &zero_scales,
        SOURCE_QUBIT,
        &feedforward_alphas,
        &mut z_10640_ff,
    );
    let mut p_lp_ff = b_ff[0];
    for q in 0..N_QUBITS {
        p_lp_ff += w_ff[q][0] * z_10640_ff[q];
    }

    // 3. DRU pred (12-obs)
    encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
    let mut z_10640_dru = [0.0; N_QUBITS];
    ansatz.forward_reuploading_in_place(
        &mut l2_state,
        &circuit_params,
        &feats_10640,
        &dru_scales,
        SOURCE_QUBIT,
        &feedforward_alphas,
        &mut z_10640_dru,
    );
    let mut p_lp_dru = b_dru[0];
    for q in 0..N_QUBITS {
        p_lp_dru += w_dru[q][0] * z_10640_dru[q];
    }

    // 4. Unified pred (78-obs)
    encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
    let mut z_10640_uni = [0.0; N_QUBITS];
    let mut zz_10640_uni = [0.0; N_PAIRS];
    ansatz.forward_reuploading_full_readout_in_place(
        &mut l2_state,
        &circuit_params,
        &feats_10640,
        &dru_scales,
        SOURCE_QUBIT,
        &feedforward_alphas,
        &mut z_10640_uni,
        &mut zz_10640_uni,
    );
    let mut obs_10640_uni = Vec::with_capacity(N_OBS_FULL);
    obs_10640_uni.extend_from_slice(&z_10640_uni);
    obs_10640_uni.extend_from_slice(&zz_10640_uni);
    let mut p_lp_uni = b_uni[0];
    for q in 0..N_OBS_FULL {
        p_lp_uni += w_uni[q][0] * obs_10640_uni[q];
    }

    let e_b = (p_lp_b - true_lp_10640).abs();
    let e_ff = (p_lp_ff - true_lp_10640).abs();
    let e_dru = (p_lp_dru - true_lp_10640).abs();
    let e_uni = (p_lp_uni - true_lp_10640).abs();

    println!(
        "  SMILES: O=[N+]([O-])c1ccc2c(c1)=[NH+][C@@H](CS[C@H]1[NH+]=c3ccc(Br)cc3=[NH+]1)[NH+]=2"
    );
    println!("  Physical Target : True logP = {:>+10.4}\n", true_lp_10640);
    println!("  ┌────────────────────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Architectural Stage           │  Predicted logP  │  Absolute Error  │");
    println!("  ├────────────────────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  Baseline 12-Qubit QNN         │ {:>16.4} │ {:>16.4} │",
        p_lp_b, e_b
    );
    println!(
        "  │  + Mech #2 (Feed-Forward)      │ {:>16.4} │ {:>16.4} │",
        p_lp_ff, e_ff
    );
    println!(
        "  │  + Mech #3 (Data Re-Uploading) │ {:>16.4} │ {:>16.4} │",
        p_lp_dru, e_dru
    );
    println!(
        "  │  + Unified (M1 + M2 + M3: 78)  │ {:>16.4} │ {:>16.4} │",
        p_lp_uni, e_uni
    );
    println!("  └────────────────────────────────┴──────────────────┴──────────────────┘\n");
    println!(
        "  Born Solvation Error Reduction: {:.4} -> {:.4} (Gain = {:>+6.2}%)\n",
        e_b,
        e_uni,
        (1.0 - e_uni / e_b) * 100.0
    );

    // =========================================================================
    // PHASE 6: TOP OUTLIER CORRECTIONS
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 6] TOP OUTLIER CORRECTIONS (Baseline vs Unified Stack)           ");
    println!("=========================================================================");

    let mut corrections: Vec<(usize, f64, f64, f64, &str)> = Vec::new();
    for (i, sample) in test_samples.iter().enumerate() {
        let err_b = (preds_base[i][0] - sample.log_p).abs();
        let err_u = (preds_uni[i][0] - sample.log_p).abs();
        let correction = err_b - err_u;
        corrections.push((i, correction, err_b, err_u, &sample.smiles));
    }
    corrections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(" Top 10 logP Error Reductions across Holdout Set:");
    println!(
        " ┌────────┬────────────────┬────────────────┬────────────────┬──────────────────────────┐"
    );
    println!(
        " │  Rank  │  Baseline |ε|  │  Unified |ε|   │  Δ Correction  │  SMILES (truncated)      │"
    );
    println!(
        " ├────────┼────────────────┼────────────────┼────────────────┼──────────────────────────┤"
    );
    for (rank, &(_idx, correction, err_b, err_u, smiles)) in corrections.iter().take(10).enumerate()
    {
        let smi_short = if smiles.len() > 24 {
            &smiles[..24]
        } else {
            smiles
        };
        println!(
            " │  {:>4}  │ {:>14.4} │ {:>14.4} │ {:>+14.4} │  {:<24} │",
            rank + 1,
            err_b,
            err_u,
            correction,
            smi_short
        );
    }
    println!(" └────────┴────────────────┴────────────────┴────────────────┴──────────────────────────┘\n");

    // =========================================================================
    // PHASE 7: THROUGHPUT BENCHMARK
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 7] THROUGHPUT BENCHMARK                                          ");
    println!("=========================================================================");

    let repeats = 3;
    let t_bb = Instant::now();
    for _ in 0..repeats {
        for s in &test_samples {
            encoding.encode_amplitude_into(&s.features, &mut l2_state);
            ansatz.forward_l2_in_place(&mut l2_state, &circuit_params[..48]);
            let mut z = [0.0; N_QUBITS];
            t1000::simd::measure_fused_ry_z_readout(
                &l2_state.real,
                &l2_state.imag,
                &circuit_params[48..60],
                &mut z,
            );
        }
    }
    let tp_baseline = (test_samples.len() * repeats) as f64 / t_bb.elapsed().as_secs_f64();

    let t_dd = Instant::now();
    for _ in 0..repeats {
        for s in &test_samples {
            encoding.encode_amplitude_into(&s.features, &mut l2_state);
            let mut z = [0.0; N_QUBITS];
            ansatz.forward_reuploading_in_place(
                &mut l2_state,
                &circuit_params,
                &s.features,
                &dru_scales,
                SOURCE_QUBIT,
                &feedforward_alphas,
                &mut z,
            );
        }
    }
    let tp_reupload = (test_samples.len() * repeats) as f64 / t_dd.elapsed().as_secs_f64();

    let t_uu = Instant::now();
    for _ in 0..repeats {
        for s in &test_samples {
            encoding.encode_amplitude_into(&s.features, &mut l2_state);
            let mut z = [0.0; N_QUBITS];
            let mut zz = [0.0; N_PAIRS];
            ansatz.forward_reuploading_full_readout_in_place(
                &mut l2_state,
                &circuit_params,
                &s.features,
                &dru_scales,
                SOURCE_QUBIT,
                &feedforward_alphas,
                &mut z,
                &mut zz,
            );
        }
    }
    let tp_unified = (test_samples.len() * repeats) as f64 / t_uu.elapsed().as_secs_f64();

    println!(
        "  Baseline Readout (12 obs) : {:>7.0} mols/sec",
        tp_baseline
    );
    println!(
        "  Re-Uploading     (12 obs) : {:>7.0} mols/sec (Overhead: {:>+.2}%)",
        tp_reupload,
        (1.0 - tp_reupload / tp_baseline) * 100.0
    );
    println!(
        "  Unified Readout  (78 obs) : {:>7.0} mols/sec\n",
        tp_unified
    );

    // =========================================================================
    // PHASE 8: PERSISTENCE (CHECKPOINT & REPORT)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 8] PERSISTING CHECKPOINT & TELEMETRY REPORT                      ");
    println!("=========================================================================");

    let checkpoint = ReuploadingCheckpoint {
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        n_qubits: N_QUBITS,
        depth: 3,
        source_qubit: SOURCE_QUBIT,
        circuit_params: circuit_params.clone(),
        feedforward_alphas: feedforward_alphas.clone(),
        upload_scales_l0: dru_scales.clone(),
        readout_weights_12: w_dru,
        readout_biases_12: b_dru,
        readout_weights_78: w_uni,
        readout_biases_78: b_uni,
    };

    let ckpt_save_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_reuploading.json",
    );
    if let Some(parent) = ckpt_save_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let ckpt_json = serde_json::to_string_pretty(&checkpoint).expect("Serialize checkpoint");
    t1000::results::write(&ckpt_save_path, ckpt_json).expect("Write checkpoint");
    println!(
        "  [✓] Saved Re-Uploading Checkpoint: {}",
        ckpt_save_path.display()
    );

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        train_samples
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
        test_samples
            .iter()
            .map(|s| (s.features.as_slice(), [s.log_p, s.qed, s.sas])),
    );
    let report = ReuploadingReport {
        classical_baselines: classical_baselines.clone(),
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        host_cpu: "Native Host AVX2+FMA".to_string(),
        n_qubits: N_QUBITS,
        source_qubit: SOURCE_QUBIT,
        train_samples: train_samples.len(),
        test_samples: test_samples.len(),
        feedforward_alphas,
        upload_scales_l0: dru_scales,
        baseline_mae_logp: mae_b[0],
        baseline_mae_qed: mae_b[1],
        baseline_mae_sas: mae_b[2],
        baseline_r2_logp: r2_b[0],
        baseline_r2_qed: r2_b[1],
        baseline_r2_sas: r2_b[2],
        feedforward_mae_logp: mae_ff[0],
        feedforward_mae_qed: mae_ff[1],
        feedforward_mae_sas: mae_ff[2],
        feedforward_r2_logp: r2_ff[0],
        feedforward_r2_qed: r2_ff[1],
        feedforward_r2_sas: r2_ff[2],
        reuploading_mae_logp: mae_dru[0],
        reuploading_mae_qed: mae_dru[1],
        reuploading_mae_sas: mae_dru[2],
        reuploading_r2_logp: r2_dru[0],
        reuploading_r2_qed: r2_dru[1],
        reuploading_r2_sas: r2_dru[2],
        unified_mae_logp: mae_uni[0],
        unified_mae_qed: mae_uni[1],
        unified_mae_sas: mae_uni[2],
        unified_r2_logp: r2_uni[0],
        unified_r2_qed: r2_uni[1],
        unified_r2_sas: r2_uni[2],
        mol10640_true_logp: true_lp_10640,
        mol10640_baseline_pred_logp: p_lp_b,
        mol10640_feedforward_pred_logp: p_lp_ff,
        mol10640_reuploading_pred_logp: p_lp_dru,
        mol10640_unified_pred_logp: p_lp_uni,
        baseline_throughput_mols_per_sec: tp_baseline,
        reuploading_throughput_mols_per_sec: tp_reupload,
        unified_throughput_mols_per_sec: tp_unified,
    };

    let report_save_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/reports/zinc_12qubit_reuploading_report.json",
    );
    if let Some(parent) = report_save_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    {
        use t1000::baselines::{print_comparison, QnnScore};
        let qnn_rows = [
            QnnScore::new("circuit, no mechanisms")
                .mae([
                    report.baseline_mae_logp,
                    report.baseline_mae_qed,
                    report.baseline_mae_sas,
                ])
                .r2([
                    report.baseline_r2_logp,
                    report.baseline_r2_qed,
                    report.baseline_r2_sas,
                ]),
            QnnScore::new("mid-circuit feed-forward")
                .mae([
                    report.feedforward_mae_logp,
                    report.feedforward_mae_qed,
                    report.feedforward_mae_sas,
                ])
                .r2([
                    report.feedforward_r2_logp,
                    report.feedforward_r2_qed,
                    report.feedforward_r2_sas,
                ]),
            QnnScore::new("data re-uploading")
                .mae([
                    report.reuploading_mae_logp,
                    report.reuploading_mae_qed,
                    report.reuploading_mae_sas,
                ])
                .r2([
                    report.reuploading_r2_logp,
                    report.reuploading_r2_qed,
                    report.reuploading_r2_sas,
                ]),
            QnnScore::new("re-uploading + ZZ readout")
                .mae([
                    report.unified_mae_logp,
                    report.unified_mae_qed,
                    report.unified_mae_sas,
                ])
                .r2([
                    report.unified_r2_logp,
                    report.unified_r2_qed,
                    report.unified_r2_sas,
                ]),
        ];
        print_comparison(&qnn_rows, &classical_baselines);
    }

    let report_json = serde_json::to_string_pretty(&report).expect("Serialize report");
    t1000::results::write(&report_save_path, report_json).expect("Write report");
    println!(
        "  [✓] Saved Machine-Readable Report: {}",
        report_save_path.display()
    );
    println!("\n=========================================================================");
    println!(" MECHANISM #3 DATA RE-UPLOADING VERIFICATION COMPLETE                     ");
    println!("=========================================================================");
}
