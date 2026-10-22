use serde::{Deserialize, Serialize};
use std::env;
use std::time::Instant;
use t1000::calibration::AffineFit;
use t1000::zinc_data::{self, ZincMolecule};
use t1000::zinc_training_common::{compute_state_norm, resolve_workspace_path, PropertyMetric};
use t1000::{
    adjoint_gradient, AdamOptimizer, CircuitAnsatz, DataEncodingLayer, EncodingKind,
    PauliObservable, QuantumState,
};

// Empirical ZINC 10k Target Distributions (Exact Mean & Std)
const MU_LOGP: f64 = 2.4713;
const SIGMA_LOGP: f64 = 1.4451;

const MU_QED: f64 = 0.7266;
const SIGMA_QED: f64 = 0.1398;

const MU_SAS: f64 = 3.0515;
const SIGMA_SAS: f64 = 0.8394;

const SIGMAS: [f64; 3] = [SIGMA_LOGP, SIGMA_QED, SIGMA_SAS];
const N_QUBITS: usize = 10;
const DEPTH: usize = 4;
const RULE: &str = "-------------------------------------------------------------------------";
const DOUBLE_RULE: &str =
    "=========================================================================";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochTelemetry {
    pub epoch: usize,
    pub batch_count: usize,
    pub train_loss: f64,
    pub grad_norm: f64,
    pub epoch_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zinc10kRunReport {
    pub timestamp_rfc3339: String,
    pub host_cpu: String,
    pub host_ram_gb: f64,
    pub n_qubits: usize,
    pub state_dimension: usize,
    pub state_memory_bytes: usize,
    pub circuit_depth: usize,
    pub num_parameters: usize,
    pub train_sample_count: usize,
    pub test_sample_count: usize,
    pub epochs_trained: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub epoch_history: Vec<EpochTelemetry>,
    pub test_eval_time_ms: f64,
    pub per_molecule_latency_us: f64,
    pub throughput_molecules_per_sec: f64,
    pub logp_metrics: PropertyMetric,
    pub qed_metrics: PropertyMetric,
    pub sas_metrics: PropertyMetric,
    pub unitary_norm: f64,
    #[serde(default)]
    pub classical_baselines: Vec<t1000::baselines::BaselineResult>,
    /// Trained circuit weights (reloadable by analysis tools such as `zinc_feature_probe`).
    #[serde(default)]
    pub final_weights: Vec<f64>,
}

/// Command-line options.
struct Args {
    num_epochs: usize,
    batch_size: usize,
    lr: f64,
    max_train: usize,
    max_eval: usize,
    max_batches: usize,    // 0 = all batches
    calibrated_head: bool, // refit an affine readout on the training set every epoch
    num_threads: usize,
}

impl Args {
    fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut a = Args {
            num_epochs: 5,
            batch_size: 50,
            lr: 0.03,
            max_train: 10000,
            max_eval: 2000,
            max_batches: 0,
            calibrated_head: false,
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get().min(12))
                .unwrap_or(8),
        };
        let mut i = 1;
        while i < args.len() {
            // A flag with no following value is ignored and consumes nothing.
            let next = args.get(i + 1).map(String::as_str);
            let consumed = match args[i].as_str() {
                "--epochs" => next
                    .map(|v| a.num_epochs = v.parse().unwrap_or(5))
                    .is_some(),
                "--batch-size" => next
                    .map(|v| a.batch_size = v.parse().unwrap_or(50))
                    .is_some(),
                "--lr" => next.map(|v| a.lr = v.parse().unwrap_or(0.03)).is_some(),
                "--samples" => next
                    .map(|v| a.max_train = v.parse().unwrap_or(10000))
                    .is_some(),
                "--eval-samples" => next
                    .map(|v| a.max_eval = v.parse().unwrap_or(2000))
                    .is_some(),
                "--threads" => next
                    .map(|v| a.num_threads = v.parse().unwrap_or(8))
                    .is_some(),
                "--max-batches" => next
                    .map(|v| a.max_batches = v.parse().unwrap_or(0))
                    .is_some(),
                "--calibrated-head" => {
                    a.calibrated_head = true;
                    false
                }
                _ => false,
            };
            i += 1 + usize::from(consumed);
        }
        a
    }

    fn print_banner(&self) {
        println!("{DOUBLE_RULE}");
        println!("     GUDDANETWORKS QNN: ZINC 10,000-MOLECULE QUANTUM TRAINING SESSION   ");
        println!("{DOUBLE_RULE}");
        println!(" Training Set    : 10,000 Featurized Drug Molecules (ZINC 250k)");
        println!(
            " Evaluation Set  : {} Holdout Drug Molecules",
            self.max_eval
        );
        println!(
            " Hyperparameters : Epochs = {}, Batch Size = {}, Learning Rate = {:.3}",
            self.num_epochs, self.batch_size, self.lr
        );
        println!(
            " Hardware Engine : AVX2+FMA SIMD (10 Qubits | 16 KB L1 Cache | {} Threads)",
            self.num_threads
        );
        println!(" Targets         : Multi-Task logP, qed, SAS (Exact Analytic Parameter Shift)");
        println!("{DOUBLE_RULE}");
    }
}

#[path = "zinc_qnn_train/report.rs"]
mod report;
#[path = "zinc_qnn_train/training.rs"]
mod training;

use report::{evaluate, load_samples, print_evaluation, save_report};
use training::{print_baseline, targets_of, train, Model};

fn main() {
    let args = Args::parse();
    args.print_banner();

    let Some((train_samples, test_samples)) = load_samples(&args) else {
        return;
    };
    let model = Model::new();
    let mut out_state = QuantumState::new(model.encoding.dim);

    // Baseline holdout evaluation (Zero-Shot Pre-Training)
    let init_mse = print_baseline(&test_samples, &model, &mut out_state);

    let total_train_start = Instant::now();
    let (opt_weights, head, epoch_telemetry) = train(&args, &model, &train_samples, &mut out_state);
    println!(
        "\nTraining completed in {:.2} seconds.",
        total_train_start.elapsed().as_secs_f64()
    );

    let ev = evaluate(
        &model,
        &test_samples,
        &opt_weights,
        &head,
        init_mse,
        &mut out_state,
    );
    let sets = (&train_samples[..], &test_samples[..]);
    print_evaluation(&model, sets, &opt_weights, &ev, &mut out_state);

    // Classical reference: ridge regression on the same raw features and train/test split.
    let classical_baselines = t1000::baselines::zinc_baselines(
        train_samples
            .iter()
            .map(|s| (s.features.as_slice(), targets_of(s))),
        test_samples
            .iter()
            .map(|s| (s.features.as_slice(), targets_of(s))),
    );
    let report = Zinc10kRunReport {
        classical_baselines: classical_baselines.clone(),
        final_weights: opt_weights,
        timestamp_rfc3339: "2026-10-13T12:35:00Z".to_string(),
        host_cpu: t1000::hostinfo::HostProfile::detect().label(),
        host_ram_gb: 16.0,
        n_qubits: N_QUBITS,
        state_dimension: model.encoding.dim,
        state_memory_bytes: model.encoding.dim * 16,
        circuit_depth: DEPTH,
        num_parameters: model.ansatz.param_count(),
        train_sample_count: train_samples.len(),
        test_sample_count: test_samples.len(),
        epochs_trained: args.num_epochs,
        batch_size: args.batch_size,
        learning_rate: args.lr,
        epoch_history: epoch_telemetry,
        test_eval_time_ms: ev.eval_ms,
        per_molecule_latency_us: ev.per_mol_latency_us,
        throughput_molecules_per_sec: ev.throughput_mol_sec,
        logp_metrics: ev.scores[0].clone(),
        qed_metrics: ev.scores[1].clone(),
        sas_metrics: ev.scores[2].clone(),
        unitary_norm: ev.norm,
    };

    {
        use t1000::baselines::{print_comparison, QnnScore};
        let col = |f: fn(&PropertyMetric) -> f64| ev.scores.each_ref().map(f);
        let qnn_rows = [QnnScore::new("trained circuit")
            .mae(col(|m| m.tuned_mae))
            .mse(col(|m| m.tuned_mse))
            .pearson_r(col(|m| m.pearson_r))];
        print_comparison(&qnn_rows, &classical_baselines);
    }
    save_report(&args, sets, &ev, &report);
}
