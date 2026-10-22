//! GuddaNetworks 12-Qubit QNN: Mid-Circuit Feed-Forward Engine
//!
//! MECHANISM #2: Non-destructive mid-circuit measurement of Qubit 8 expectation ⟨Z_8⟩_L1
//! dynamically modulating terminal rotation angles before readout:
//!   θ̃_{y, q} = θ_{y, q} + α_q · ⟨Z_8⟩_L1
//!
//! Theoretical Foundation:
//!   Single-upload unitary circuits are mathematically restricted to degree-2 quadratic
//!   forms in normalized descriptors: ⟨Z_q⟩ = (x^T M_q x) / ||x||^2.
//!   Molecule #10640 (tri-cationic zwitterion, true logP = -6.772 vs pred = +2.418) induces
//!   93.4% spin freeze on Qubit 8 (⟨Z_8⟩_L1 = -0.867).
//!   Feeding ⟨Z_8⟩_L1 forward into subsequent gate angles introduces cos(α_q · ⟨Z_8⟩) and
//!   sin(α_q · ⟨Z_8⟩) terms, yielding degree-4, degree-6, and higher-order transcendental
//!   non-linearities that break the quadratic ceiling and resolve deep-regime Born solvation.

use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};
use t1000::zinc_data::{self, ZincMolecule};
use t1000::zinc_training_common::{MidCircuitCheckpoint, PrunedCheckpoint, compute_mae_and_r2, resolve_workspace_path, solve_ridge_regression, utc_timestamp_rfc3339};

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4,096
const N_PAIRS: usize = N_QUBITS * (N_QUBITS - 1) / 2; // 66
const N_OBS_FULL: usize = N_QUBITS + N_PAIRS; // 78
const SOURCE_QUBIT: usize = 8; // Qubit 8 exhibiting severe spin freeze on anomalous molecules

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MidCircuitReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub n_qubits: usize,
    pub source_qubit: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub feedforward_alphas: Vec<f64>,
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
    pub combined_mae_logp: f64,
    pub combined_mae_qed: f64,
    pub combined_mae_sas: f64,
    pub combined_r2_logp: f64,
    pub combined_r2_qed: f64,
    pub combined_r2_sas: f64,
    pub mol10640_true_logp: f64,
    pub mol10640_baseline_pred_logp: f64,
    pub mol10640_feedforward_pred_logp: f64,
    pub mol10640_combined_pred_logp: f64,
    pub mol10640_midcircuit_z8: f64,
    pub baseline_throughput_mols_per_sec: f64,
    pub feedforward_throughput_mols_per_sec: f64,
    pub combined_throughput_mols_per_sec: f64,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
}

/// Computes composite MSE loss for a given alpha vector on a dataset:
fn eval_alpha_loss(
    ansatz: &CircuitAnsatz,
    encoding: &DataEncodingLayer,
    circuit_params: &[f64],
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
        ansatz.forward_midcircuit_feedforward_in_place(
            &mut l2_state,
            circuit_params,
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
        // Calibrated weights: 1.0 for logP, 5.0 for QED, 1.0 for SAS
        total_loss += 1.0 * (e_lp * e_lp) + 5.0 * (e_qed * e_qed) + 1.0 * (e_sas * e_sas);
    }
    total_loss /= samples.len() as f64;

    (total_loss, w, b)
}

fn main() {
    println!("=========================================================================");
    println!(" GUDDANETWORKS 12-QUBIT QNN: MID-CIRCUIT FEED-FORWARD ENGINE (QUBIT 8)  ");
    println!("=========================================================================");
    println!(" Mode               : Dynamic Feed-Forward Non-Linearity Injection");
    println!(
        " Source Qubit       : Qubit {} (⟨Z_{}⟩ Mid-Circuit Readout)",
        SOURCE_QUBIT, SOURCE_QUBIT
    );
    println!(
        " Mechanism          : θ̃_{{y,q}} = θ_{{y,q}} + α_q · ⟨Z_{}⟩_L1",
        SOURCE_QUBIT
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
        println!(
            "  -> Using pruned 99-parameter checkpoint: {}",
            pruned_ckpt_path.display()
        );
        let content = fs::read_to_string(&pruned_ckpt_path).expect("Read pruned checkpoint");
        let ckpt: PrunedCheckpoint =
            serde_json::from_str(&content).expect("Parse pruned checkpoint");
        circuit_params = ckpt.pruned_hybrid_params[..60].to_vec();
    } else {
        eprintln!(
            "[-] Pruned checkpoint not found at: {}",
            pruned_ckpt_path.display()
        );
        return;
    }
    println!(
        "  -> Active circuit parameters: {} (48 L1/L2 angles + 12 terminal Ry)\n",
        circuit_params.len()
    );

    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let mut l2_state = QuantumState::new(DIM);

    // =========================================================================
    // PHASE 1: BASELINE VERIFICATION (alpha = 0)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 1] BASELINE QUANTUM EVALUATION (Zero Feed-Forward: α = 0)        ");
    println!("=========================================================================");

    let alpha_zero = vec![0.0; N_QUBITS];
    let mut x_train_b = Vec::with_capacity(train_samples.len());
    let mut y_train = Vec::with_capacity(train_samples.len());

    for s in &train_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z_b = [0.0; N_QUBITS];
        ansatz.forward_midcircuit_feedforward_in_place(
            &mut l2_state,
            &circuit_params,
            SOURCE_QUBIT,
            &alpha_zero,
            &mut z_b,
        );
        x_train_b.push(z_b.to_vec());
        y_train.push(vec![s.log_p, s.qed, s.sas]);
    }

    let (w_base, b_base) = solve_ridge_regression(&x_train_b, &y_train, 1e-3);

    // Evaluate on test set
    let mut x_test_b = Vec::with_capacity(test_samples.len());
    let mut y_test = Vec::with_capacity(test_samples.len());
    let mut preds_base = Vec::with_capacity(test_samples.len());

    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z_b = [0.0; N_QUBITS];
        ansatz.forward_midcircuit_feedforward_in_place(
            &mut l2_state,
            &circuit_params,
            SOURCE_QUBIT,
            &alpha_zero,
            &mut z_b,
        );
        x_test_b.push(z_b.to_vec());
        y_test.push(vec![s.log_p, s.qed, s.sas]);

        let mut p = [b_base[0], b_base[1], b_base[2]];
        for q in 0..N_QUBITS {
            for k in 0..3 {
                p[k] += w_base[q][k] * z_b[q];
            }
        }
        preds_base.push(p.to_vec());
    }

    let (mae_b, r2_b) = compute_mae_and_r2(&preds_base, &y_test);

    println!("  Baseline Holdout Performance (2,000 molecules):");
    println!("  ┌─────────────┬──────────────┬──────────────┐");
    println!("  │  Property   │  MAE         │  R² Score    │");
    println!("  ├─────────────┼──────────────┼──────────────┤");
    println!("  │  logP       │ {:>12.6} │ {:>12.6} │", mae_b[0], r2_b[0]);
    println!("  │  QED        │ {:>12.6} │ {:>12.6} │", mae_b[1], r2_b[1]);
    println!("  │  SAS        │ {:>12.6} │ {:>12.6} │", mae_b[2], r2_b[2]);
    println!("  └─────────────┴──────────────┴──────────────┘\n");

    // Evaluate Molecule #10640 under baseline
    let feats_10640 = [
        24.0, 408.279, 15.0, 5.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 12.0, 0.5, 9.0, 0.375, 5.0, 0.0,
        4.0, 3.0, 4.0,
    ];
    let true_lp_10640 = -6.7716;

    encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
    let mut z_10640_b = [0.0; N_QUBITS];
    let m8_10640 = ansatz.forward_midcircuit_feedforward_in_place(
        &mut l2_state,
        &circuit_params,
        SOURCE_QUBIT,
        &alpha_zero,
        &mut z_10640_b,
    );

    let mut pred_lp_10640_b = b_base[0];
    for q in 0..N_QUBITS {
        pred_lp_10640_b += w_base[q][0] * z_10640_b[q];
    }

    println!("  [ANOMALY CHECK] Molecule #10640 Baseline Status:");
    println!("    SMILES           : O=[N+]([O-])c1ccc2c(c1)=[NH+][C@@H](CS[C@H]1[NH+]=c3ccc(Br)cc3=[NH+]1)[NH+]=2");
    println!("    True logP        : {:>+10.4}", true_lp_10640);
    println!(
        "    Baseline Pred    : {:>+10.4} (Absolute Error = {:.4})",
        pred_lp_10640_b,
        (pred_lp_10640_b - true_lp_10640).abs()
    );
    println!(
        "    ⟨Z_{}⟩_L1 Signal  : {:>+10.6} (Spin Polarization = {:.1}%)\n",
        SOURCE_QUBIT,
        m8_10640,
        (1.0 + m8_10640.abs()) * 50.0
    );

    // =========================================================================
    // PHASE 2: OPTIMIZING FEED-FORWARD ANGLES (α_q)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 2] TRAINING MID-CIRCUIT FEED-FORWARD PARAMETERS (α ∈ R¹²)        ");
    println!("=========================================================================");
    println!(" Optimizing 12 classical feed-forward weights via Adam + Nested Ridge Regression...");
    println!(" Objective: min_α L(α) = MSE(Z(α) W*(α) + b*(α), Y) across training molecules\n");

    let mut alphas = vec![0.0; N_QUBITS];
    // Seed with physics-motivated initialization: counter-rotate neighbors of Qubit 8
    alphas[7] = 0.25;
    alphas[8] = -0.40;
    alphas[9] = 0.25;

    let n_train_opt = 2_000; // Use representative subset for fast gradient calculation
    let opt_subset: Vec<ZincMolecule> = train_samples.iter().take(n_train_opt).cloned().collect();

    let n_epochs = 12;
    let lr = 0.08;
    let eps = 0.01; // Finite difference step
    let beta1 = 0.9;
    let beta2 = 0.999;
    let mut m_adam = vec![0.0; N_QUBITS];
    let mut v_adam = vec![0.0; N_QUBITS];

    let t_opt = Instant::now();

    println!(" Epoch │ Composite Loss │  logP Pred (#10640) │  ||α||_2  │ Top Alphas (q: α_q)");
    println!("───────┼────────────────┼─────────────────────┼───────────┼────────────────────────────────────");

    for epoch in 1..=n_epochs {
        let (loss_curr, w_curr, b_curr) = eval_alpha_loss(
            &ansatz,
            &encoding,
            &circuit_params,
            &alphas,
            &opt_subset,
            1e-3,
        );

        // Check Molecule #10640 prediction at current alpha
        encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
        let mut z_10640_curr = [0.0; N_QUBITS];
        ansatz.forward_midcircuit_feedforward_in_place(
            &mut l2_state,
            &circuit_params,
            SOURCE_QUBIT,
            &alphas,
            &mut z_10640_curr,
        );
        let mut pred_lp_10640_epoch = b_curr[0];
        for q in 0..N_QUBITS {
            pred_lp_10640_epoch += w_curr[q][0] * z_10640_curr[q];
        }

        // Compute gradient vector via central finite difference
        let mut grad = vec![0.0; N_QUBITS];
        for q in 0..N_QUBITS {
            let mut alphas_plus = alphas.clone();
            alphas_plus[q] += eps;
            let (l_plus, _, _) = eval_alpha_loss(
                &ansatz,
                &encoding,
                &circuit_params,
                &alphas_plus,
                &opt_subset,
                1e-3,
            );

            let mut alphas_minus = alphas.clone();
            alphas_minus[q] -= eps;
            let (l_minus, _, _) = eval_alpha_loss(
                &ansatz,
                &encoding,
                &circuit_params,
                &alphas_minus,
                &opt_subset,
                1e-3,
            );

            grad[q] = (l_plus - l_minus) / (2.0 * eps);
        }

        // Adam update
        for q in 0..N_QUBITS {
            m_adam[q] = beta1 * m_adam[q] + (1.0 - beta1) * grad[q];
            v_adam[q] = beta2 * v_adam[q] + (1.0 - beta2) * grad[q] * grad[q];
            let m_hat = m_adam[q] / (1.0 - beta1.powi(epoch as i32));
            let v_hat = v_adam[q] / (1.0 - beta2.powi(epoch as i32));
            alphas[q] -= lr * m_hat / (v_hat.sqrt() + 1e-8);
            // Clamp to avoid angle wrap-around
            alphas[q] = alphas[q].clamp(-std::f64::consts::PI, std::f64::consts::PI);
        }

        let norm_alpha = alphas.iter().map(|&a| a * a).sum::<f64>().sqrt();

        // Print top 3 alphas
        let mut alpha_pairs: Vec<(usize, f64)> = alphas.iter().cloned().enumerate().collect();
        alpha_pairs.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        let top_str = format!(
            "q{}: {:>+5.2}, q{}: {:>+5.2}, q{}: {:>+5.2}",
            alpha_pairs[0].0,
            alpha_pairs[0].1,
            alpha_pairs[1].0,
            alpha_pairs[1].1,
            alpha_pairs[2].0,
            alpha_pairs[2].1,
        );

        println!(
            "  {:>3}  │   {:>10.6}   │      {:>+10.4}     │  {:>7.4}  │ {}",
            epoch, loss_curr, pred_lp_10640_epoch, norm_alpha, top_str
        );
    }
    let el_opt = t_opt.elapsed();
    println!("───────┴────────────────┴─────────────────────┴───────────┴────────────────────────────────────");
    println!("  Optimization completed in {:.2}s\n", el_opt.as_secs_f64());

    // Print final alpha distribution across all 12 qubits
    println!(" Final Feed-Forward Weight Distribution (α_q for q ∈ 0..11):");
    print!("   ");
    for q in 0..N_QUBITS {
        print!("q{:02}: {:>+6.3}  ", q, alphas[q]);
        if (q + 1) % 6 == 0 {
            print!("\n   ");
        }
    }
    println!();

    // =========================================================================
    // PHASE 3: HOLDOUT EVALUATION (12 Observables with Mid-Circuit Feed-Forward)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 3] EVALUATING MID-CIRCUIT FEED-FORWARD (12 Observables)          ");
    println!("=========================================================================");

    // Fit final readout weights on full 10,000 training set
    let mut x_train_ff = Vec::with_capacity(train_samples.len());
    for s in &train_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z_ff = [0.0; N_QUBITS];
        ansatz.forward_midcircuit_feedforward_in_place(
            &mut l2_state,
            &circuit_params,
            SOURCE_QUBIT,
            &alphas,
            &mut z_ff,
        );
        x_train_ff.push(z_ff.to_vec());
    }

    let (w_ff, b_ff) = solve_ridge_regression(&x_train_ff, &y_train, 1e-3);

    // Evaluate on test set
    let mut x_test_ff = Vec::with_capacity(test_samples.len());
    let mut preds_ff = Vec::with_capacity(test_samples.len());

    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z_ff = [0.0; N_QUBITS];
        ansatz.forward_midcircuit_feedforward_in_place(
            &mut l2_state,
            &circuit_params,
            SOURCE_QUBIT,
            &alphas,
            &mut z_ff,
        );
        x_test_ff.push(z_ff.to_vec());

        let mut p = [b_ff[0], b_ff[1], b_ff[2]];
        for q in 0..N_QUBITS {
            for k in 0..3 {
                p[k] += w_ff[q][k] * z_ff[q];
            }
        }
        preds_ff.push(p.to_vec());
    }

    let (mae_ff, r2_ff) = compute_mae_and_r2(&preds_ff, &y_test);

    let imp_ff_lp = (1.0 - mae_ff[0] / mae_b[0]) * 100.0;
    let imp_ff_qed = (1.0 - mae_ff[1] / mae_b[1]) * 100.0;
    let imp_ff_sas = (1.0 - mae_ff[2] / mae_b[2]) * 100.0;

    println!("  Holdout Performance Comparison: Baseline vs Mid-Circuit Feed-Forward:");
    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline MAE    │  Feed-Fwd MAE    │  Improvement     │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  logP       │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[0], mae_ff[0], imp_ff_lp
    );
    println!(
        "  │  QED        │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[1], mae_ff[1], imp_ff_qed
    );
    println!(
        "  │  SAS        │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[2], mae_ff[2], imp_ff_sas
    );
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┘\n");
    println!("  ┌─────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline R²     │  Feed-Fwd R²     │");
    println!("  ├─────────────┼──────────────────┼──────────────────┤");
    println!("  │  logP       │ {:>16.6} │ {:>16.6} │", r2_b[0], r2_ff[0]);
    println!("  │  QED        │ {:>16.6} │ {:>16.6} │", r2_b[1], r2_ff[1]);
    println!("  │  SAS        │ {:>16.6} │ {:>16.6} │", r2_b[2], r2_ff[2]);
    println!("  └─────────────┴──────────────────┴──────────────────┘\n");

    // =========================================================================
    // PHASE 4: COMBINED MECHANISMS #1 + #2 (Feed-Forward + 66 ZZ Correlators)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 4] COMBINED MECHANISMS: FEED-FORWARD + 2-BODY ZZ (78 Observables)");
    println!("=========================================================================");
    println!(" Extracting 12 single-qubit marginals + 66 two-body correlators under dynamic");
    println!(" feed-forward angle modulation: θ̃_{{y,q}} = θ_{{y,q}} + α_q · ⟨Z_8⟩_L1\n");

    let mut x_train_comb = Vec::with_capacity(train_samples.len());
    for s in &train_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z_obs = [0.0; N_QUBITS];
        let mut zz_obs = [0.0; N_PAIRS];
        ansatz.forward_midcircuit_full_readout_in_place(
            &mut l2_state,
            &circuit_params,
            SOURCE_QUBIT,
            &alphas,
            &mut z_obs,
            &mut zz_obs,
        );
        let mut obs_78 = Vec::with_capacity(N_OBS_FULL);
        obs_78.extend_from_slice(&z_obs);
        obs_78.extend_from_slice(&zz_obs);
        x_train_comb.push(obs_78);
    }

    let (w_comb, b_comb) = solve_ridge_regression(&x_train_comb, &y_train, 1e-3);

    // Evaluate combined on test set
    let mut x_test_comb = Vec::with_capacity(test_samples.len());
    let mut preds_comb = Vec::with_capacity(test_samples.len());

    for s in &test_samples {
        encoding.encode_amplitude_into(&s.features, &mut l2_state);
        let mut z_obs = [0.0; N_QUBITS];
        let mut zz_obs = [0.0; N_PAIRS];
        ansatz.forward_midcircuit_full_readout_in_place(
            &mut l2_state,
            &circuit_params,
            SOURCE_QUBIT,
            &alphas,
            &mut z_obs,
            &mut zz_obs,
        );
        let mut obs_78 = Vec::with_capacity(N_OBS_FULL);
        obs_78.extend_from_slice(&z_obs);
        obs_78.extend_from_slice(&zz_obs);
        x_test_comb.push(obs_78.clone());

        let mut p = [b_comb[0], b_comb[1], b_comb[2]];
        for q in 0..N_OBS_FULL {
            for k in 0..3 {
                p[k] += w_comb[q][k] * obs_78[q];
            }
        }
        preds_comb.push(p.to_vec());
    }

    let (mae_comb, r2_comb) = compute_mae_and_r2(&preds_comb, &y_test);

    let imp_comb_lp = (1.0 - mae_comb[0] / mae_b[0]) * 100.0;
    let imp_comb_qed = (1.0 - mae_comb[1] / mae_b[1]) * 100.0;
    let imp_comb_sas = (1.0 - mae_comb[2] / mae_b[2]) * 100.0;

    println!("  Tri-Stage Architectural Evolution Ledger:");
    println!("  ┌─────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Property   │  Baseline (12)   │  Feed-Fwd (12)   │  Combined (78)   │  Total Imp. (%)  │");
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  logP MAE   │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[0], mae_ff[0], mae_comb[0], imp_comb_lp
    );
    println!(
        "  │  QED MAE    │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[1], mae_ff[1], mae_comb[1], imp_comb_qed
    );
    println!(
        "  │  SAS MAE    │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        mae_b[2], mae_ff[2], mae_comb[2], imp_comb_sas
    );
    println!("  ├─────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  logP R²    │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        r2_b[0],
        r2_ff[0],
        r2_comb[0],
        (r2_comb[0] - r2_b[0]) / r2_b[0].abs() * 100.0
    );
    println!(
        "  │  QED R²     │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        r2_b[1],
        r2_ff[1],
        r2_comb[1],
        (r2_comb[1] - r2_b[1]) / r2_b[1].abs() * 100.0
    );
    println!(
        "  │  SAS R²     │ {:>16.6} │ {:>16.6} │ {:>16.6} │ {:>+14.2}%  │",
        r2_b[2],
        r2_ff[2],
        r2_comb[2],
        (r2_comb[2] - r2_b[2]) / r2_b[2].abs() * 100.0
    );
    println!("  └─────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘\n");

    // =========================================================================
    // PHASE 5: MOLECULE #10640 RESOLUTION AUDIT
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 5] MOLECULE #10640 RESOLUTION AUDIT (MAX QUANTUM ANOMALY)        ");
    println!("=========================================================================");

    // Evaluate with Feed-Forward only
    encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
    let mut z_10640_ff = [0.0; N_QUBITS];
    ansatz.forward_midcircuit_feedforward_in_place(
        &mut l2_state,
        &circuit_params,
        SOURCE_QUBIT,
        &alphas,
        &mut z_10640_ff,
    );
    let mut pred_lp_10640_ff = b_ff[0];
    for q in 0..N_QUBITS {
        pred_lp_10640_ff += w_ff[q][0] * z_10640_ff[q];
    }

    // Evaluate with Combined (Feed-Forward + 78 ZZ)
    encoding.encode_amplitude_into(&feats_10640, &mut l2_state);
    let mut z_10640_comb = [0.0; N_QUBITS];
    let mut zz_10640_comb = [0.0; N_PAIRS];
    ansatz.forward_midcircuit_full_readout_in_place(
        &mut l2_state,
        &circuit_params,
        SOURCE_QUBIT,
        &alphas,
        &mut z_10640_comb,
        &mut zz_10640_comb,
    );
    let mut obs_10640_comb = Vec::with_capacity(N_OBS_FULL);
    obs_10640_comb.extend_from_slice(&z_10640_comb);
    obs_10640_comb.extend_from_slice(&zz_10640_comb);

    let mut pred_lp_10640_comb = b_comb[0];
    for q in 0..N_OBS_FULL {
        pred_lp_10640_comb += w_comb[q][0] * obs_10640_comb[q];
    }

    let err_base = (pred_lp_10640_b - true_lp_10640).abs();
    let err_ff = (pred_lp_10640_ff - true_lp_10640).abs();
    let err_comb = (pred_lp_10640_comb - true_lp_10640).abs();

    println!("  Compound Identifier : ZINC-250k Molecule #10640");
    println!("  SMILES              : O=[N+]([O-])c1ccc2c(c1)=[NH+][C@@H](CS[C@H]1[NH+]=c3ccc(Br)cc3=[NH+]1)[NH+]=2");
    println!("  Zwitterionic Bridge : Hyper-polarized thioether bridge, 3 formal iminium charges");
    println!(
        "  Mid-Circuit Signal  : ⟨Z_{}⟩_L1 = {:>+10.6}\n",
        SOURCE_QUBIT, m8_10640
    );
    println!("  ┌────────────────────────────────┬──────────────────┬──────────────────┐");
    println!("  │  Architecture Stage            │  Predicted logP  │  Absolute Error  │");
    println!("  ├────────────────────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │  True Physical Target          │ {:>16.4} │ {:>16.4} │",
        true_lp_10640, 0.0
    );
    println!(
        "  │  Baseline 12-Qubit QNN         │ {:>16.4} │ {:>16.4} │",
        pred_lp_10640_b, err_base
    );
    println!(
        "  │  + Mid-Circuit Feed-Forward    │ {:>16.4} │ {:>16.4} │",
        pred_lp_10640_ff, err_ff
    );
    println!(
        "  │  + Combined (FF + 78 ZZ)       │ {:>16.4} │ {:>16.4} │",
        pred_lp_10640_comb, err_comb
    );
    println!("  └────────────────────────────────┴──────────────────┴──────────────────┘\n");
    println!(
        "  Anomaly Error Reduction: {:.4} -> {:.4} (Gain = {:>+6.2}%)\n",
        err_base,
        err_comb,
        (1.0 - err_comb / err_base) * 100.0
    );

    // =========================================================================
    // PHASE 6: TOP OUTLIER CORRECTIONS
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 6] TOP OUTLIER CORRECTIONS (Largest Absolute Error Reduction)    ");
    println!("=========================================================================");

    let mut corrections: Vec<(usize, f64, f64, f64, &str)> = Vec::new();
    for (i, sample) in test_samples.iter().enumerate() {
        let err_b = (preds_base[i][0] - sample.log_p).abs();
        let err_c = (preds_comb[i][0] - sample.log_p).abs();
        let correction = err_b - err_c;
        corrections.push((i, correction, err_b, err_c, &sample.smiles));
    }
    corrections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!(" Top 10 logP Error Reductions (Baseline → Combined Feed-Forward + ZZ):");
    println!(
        " ┌────────┬────────────────┬────────────────┬────────────────┬──────────────────────────┐"
    );
    println!(
        " │  Rank  │  Baseline |ε|  │  Combined |ε|  │  Δ Correction  │  SMILES (truncated)      │"
    );
    println!(
        " ├────────┼────────────────┼────────────────┼────────────────┼──────────────────────────┤"
    );
    for (rank, &(_idx, correction, err_b, err_c, smiles)) in corrections.iter().take(10).enumerate()
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
            err_c,
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
    // 1. Baseline Throughput
    let t_bb = Instant::now();
    for _ in 0..repeats {
        for sample in &test_samples {
            encoding.encode_amplitude_into(&sample.features, &mut l2_state);
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

    // 2. Feed-Forward Throughput (12-obs)
    let t_ff = Instant::now();
    for _ in 0..repeats {
        for sample in &test_samples {
            encoding.encode_amplitude_into(&sample.features, &mut l2_state);
            let mut z = [0.0; N_QUBITS];
            ansatz.forward_midcircuit_feedforward_in_place(
                &mut l2_state,
                &circuit_params,
                SOURCE_QUBIT,
                &alphas,
                &mut z,
            );
        }
    }
    let tp_feedforward = (test_samples.len() * repeats) as f64 / t_ff.elapsed().as_secs_f64();

    // 3. Combined Throughput (78-obs)
    let t_cc = Instant::now();
    for _ in 0..repeats {
        for sample in &test_samples {
            encoding.encode_amplitude_into(&sample.features, &mut l2_state);
            let mut z = [0.0; N_QUBITS];
            let mut zz = [0.0; N_PAIRS];
            ansatz.forward_midcircuit_full_readout_in_place(
                &mut l2_state,
                &circuit_params,
                SOURCE_QUBIT,
                &alphas,
                &mut z,
                &mut zz,
            );
        }
    }
    let tp_combined = (test_samples.len() * repeats) as f64 / t_cc.elapsed().as_secs_f64();

    println!(
        "  Baseline Readout (12 obs) : {:>7.0} mols/sec",
        tp_baseline
    );
    println!(
        "  Feed-Forward     (12 obs) : {:>7.0} mols/sec (Overhead: {:>+.2}%)",
        tp_feedforward,
        (1.0 - tp_feedforward / tp_baseline) * 100.0
    );
    println!(
        "  Combined Readout (78 obs) : {:>7.0} mols/sec\n",
        tp_combined
    );

    // =========================================================================
    // PHASE 8: PERSISTENCE (CHECKPOINT & REPORT)
    // =========================================================================
    println!("=========================================================================");
    println!(" [PHASE 8] PERSISTING CHECKPOINT & TELEMETRY REPORT                      ");
    println!("=========================================================================");

    let checkpoint = MidCircuitCheckpoint {
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        n_qubits: N_QUBITS,
        depth: 3,
        source_qubit: SOURCE_QUBIT,
        circuit_params: circuit_params.clone(),
        feedforward_alphas: alphas.clone(),
        readout_weights_12: w_ff,
        readout_biases_12: b_ff,
        readout_weights_78: w_comb,
        readout_biases_78: b_comb,
    };

    let ckpt_save_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/checkpoints/zinc_12qubit_checkpoint_midcircuit.json",
    );
    if let Some(parent) = ckpt_save_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let ckpt_json = serde_json::to_string_pretty(&checkpoint).expect("Serialize checkpoint");
    t1000::results::write(&ckpt_save_path, ckpt_json).expect("Write checkpoint");
    println!(
        "  [✓] Saved Mid-Circuit Checkpoint: {}",
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
    let report = MidCircuitReport {
        classical_baselines: classical_baselines.clone(),
        timestamp_rfc3339: utc_timestamp_rfc3339(),
        host_cpu: "Native Host AVX2+FMA".to_string(),
        n_qubits: N_QUBITS,
        source_qubit: SOURCE_QUBIT,
        train_samples: train_samples.len(),
        test_samples: test_samples.len(),
        feedforward_alphas: alphas,
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
        combined_mae_logp: mae_comb[0],
        combined_mae_qed: mae_comb[1],
        combined_mae_sas: mae_comb[2],
        combined_r2_logp: r2_comb[0],
        combined_r2_qed: r2_comb[1],
        combined_r2_sas: r2_comb[2],
        mol10640_true_logp: true_lp_10640,
        mol10640_baseline_pred_logp: pred_lp_10640_b,
        mol10640_feedforward_pred_logp: pred_lp_10640_ff,
        mol10640_combined_pred_logp: pred_lp_10640_comb,
        mol10640_midcircuit_z8: m8_10640,
        baseline_throughput_mols_per_sec: tp_baseline,
        feedforward_throughput_mols_per_sec: tp_feedforward,
        combined_throughput_mols_per_sec: tp_combined,
    };

    let report_save_path = resolve_workspace_path(
        "tests/benchmarks/zinc_molecular/reports/zinc_12qubit_midcircuit_report.json",
    );
    if let Some(parent) = report_save_path.parent() {
        fs::create_dir_all(parent).ok();
    }
    {
        use t1000::baselines::{print_comparison, QnnScore};
        let qnn_rows = [
            QnnScore::new("circuit, no feed-forward")
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
            QnnScore::new("feed-forward + ZZ readout")
                .mae([
                    report.combined_mae_logp,
                    report.combined_mae_qed,
                    report.combined_mae_sas,
                ])
                .r2([
                    report.combined_r2_logp,
                    report.combined_r2_qed,
                    report.combined_r2_sas,
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
    println!(" MECHANISM #2 MID-CIRCUIT FEED-FORWARD VERIFICATION COMPLETE             ");
    println!("=========================================================================");
}
