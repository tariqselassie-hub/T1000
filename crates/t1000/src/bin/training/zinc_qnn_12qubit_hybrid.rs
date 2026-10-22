use std::time::Instant;
use t1000::zinc_data::{self, ZincMolecule};
use t1000::zinc_holdout_eval::{print_zero_shot, target_columns};
use t1000::zinc_training_common::{
    eval_all_z12, resolve_workspace_path, PropertyMetric, QnnCheckpoint,
};
use t1000::{
    adjoint_gradient, AdamOptimizer, CircuitAnsatz, DataEncodingLayer, EncodingKind,
    PauliObservable, QuantumState,
};

#[path = "zinc_qnn_12qubit_hybrid/args.rs"]
mod args;
#[path = "zinc_qnn_12qubit_hybrid/report.rs"]
mod report;
#[path = "zinc_qnn_12qubit_hybrid/training.rs"]
mod training;

use args::Args;
use report::EpochTelemetry;

// Empirical ZINC 10k Target Distributions (Exact Mean & Std)
const MU_LOGP: f64 = 2.4713;
const SIGMA_LOGP: f64 = 1.4451;

const MU_QED: f64 = 0.7266;
const SIGMA_QED: f64 = 0.1398;

const MU_SAS: f64 = 3.0515;
const SIGMA_SAS: f64 = 0.8394;

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4096 amplitudes (64 KB)
const N_PROPS: usize = 3;
const RULE: &str = "-------------------------------------------------------------------------";
const DOUBLE_RULE: &str =
    "=========================================================================";

/// Sizes and offsets of the flat hybrid parameter vector: quantum angles, then the 3x12
/// readout weights, then the 3 biases.
#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub n_quantum: usize,
    pub n_weights: usize,
    pub n_biases: usize,
    pub n_total: usize,
    pub w_offset: usize,
    pub b_offset: usize,
}

impl Layout {
    fn new(depth: usize) -> Self {
        let n_quantum = 2 * N_QUBITS * depth; // e.g. 2 * 12 * 3 = 72
        let n_weights = N_PROPS * N_QUBITS; // 3 * 12 = 36
        let n_biases = N_PROPS; // 3
        Layout {
            n_quantum,
            n_weights,
            n_biases,
            n_total: n_quantum + n_weights + n_biases, // 111
            w_offset: n_quantum,
            b_offset: n_quantum + n_weights,
        }
    }
}

/// The circuit plus the layout of its hybrid parameters.
#[derive(Debug)]
pub struct Model {
    encoding: DataEncodingLayer,
    ansatz: CircuitAnsatz,
    layout: Layout,
}

impl Model {
    fn new(depth: usize) -> Self {
        Model {
            encoding: DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude),
            ansatz: CircuitAnsatz::new(N_QUBITS, depth).with_pi_layer(true),
            layout: Layout::new(depth),
        }
    }

    /// Readout-head predictions (logP, QED, SAS) for every molecule, one `Vec` per property.
    fn predict(
        &self,
        samples: &[ZincMolecule],
        params: &[f64],
        state: &mut QuantumState,
    ) -> [Vec<f64>; 3] {
        let l = self.layout;
        let mut z_eval = [0.0; 12];
        let mut preds = [(); 3].map(|_| Vec::with_capacity(samples.len()));
        for s in samples {
            let state_in = self.encoding.encode(&s.features);
            self.ansatz
                .forward_into(&state_in, &params[..l.n_quantum], state);
            eval_all_z12(state, &mut z_eval);

            for (k, out) in preds.iter_mut().enumerate() {
                let mut p = params[l.b_offset + k];
                for (q, z) in z_eval.iter().enumerate() {
                    p += params[l.w_offset + k * 12 + q] * z;
                }
                out.push(p);
            }
        }
        preds
    }

    /// Fresh parameters: the ansatz angles, orthogonal-harmonic readout weights, and the
    /// empirical target means as biases.
    fn initial_params(&self) -> Vec<f64> {
        let l = self.layout;
        let mut params = vec![0.0; l.n_total];
        // 1. Quantum parameters: initial circuit angles from ansatz
        params[..l.n_quantum].copy_from_slice(&self.ansatz.weights);

        // 2. Readout weights (3x12): orthogonal harmonic initialization across 12 qubits
        let std_norm = (N_QUBITS as f64).sqrt();
        for q in 0..N_QUBITS {
            let angle = 2.0 * std::f64::consts::PI * (q as f64) / (N_QUBITS as f64);
            params[l.w_offset + q] = (SIGMA_LOGP / std_norm) * angle.cos();
            params[l.w_offset + 12 + q] = (SIGMA_QED / std_norm) * angle.sin();
            params[l.w_offset + 2 * 12 + q] = (SIGMA_SAS / std_norm) * (2.0 * angle).cos();
        }

        // 3. Readout biases (3): set to empirical means
        params[l.b_offset] = MU_LOGP;
        params[l.b_offset + 1] = MU_QED;
        params[l.b_offset + 2] = MU_SAS;
        params
    }
}

fn load_samples(args: &Args) -> Option<(Vec<ZincMolecule>, Vec<ZincMolecule>)> {
    let train_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_train_10k.csv");
    let test_path =
        resolve_workspace_path("tests/benchmarks/zinc_molecular/outputs/zinc_test_2k.csv");

    println!(
        "[INIT] Loading ZINC 10k train set from: {}",
        train_path.display()
    );
    let train_samples = match zinc_data::load(&train_path, args.max_train) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[-] Error loading train set: {}", e);
            return None;
        }
    };
    println!(
        "  -> Successfully loaded {} train molecules.",
        train_samples.len()
    );

    println!(
        "[INIT] Loading ZINC 2k evaluation set: {}",
        test_path.display()
    );
    let test_samples = match zinc_data::load(&test_path, args.max_eval) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[-] Error loading test set: {}", e);
            return None;
        }
    };
    println!(
        "  -> Successfully loaded {} test molecules.\n",
        test_samples.len()
    );
    Some((train_samples, test_samples))
}

fn main() {
    let args = Args::parse();
    let model = Model::new(args.depth);
    args.print_banner(&model.layout);

    let Some((train_samples, test_samples)) = load_samples(&args) else {
        return;
    };

    let mut hybrid_params = vec![0.0; model.layout.n_total];
    let resumed = args
        .resume
        .as_deref()
        .and_then(|p| training::try_resume(p, model.layout.n_total, &mut hybrid_params));
    if resumed.is_none() {
        hybrid_params = model.initial_params();
    }

    let mut out_state = QuantumState::new(DIM);
    // Baseline zero-shot holdout evaluation
    let targets = target_columns(&test_samples);
    println!("{RULE}");
    println!(" [ZERO-SHOT BASELINE] 2,000 HOLDOUT TEST METRICS (Pre-Training)");
    println!("{RULE}");
    let init_preds = model.predict(&test_samples, &hybrid_params, &mut out_state);
    let init_mse = print_zero_shot(&init_preds, &targets);
    println!("{RULE}\n");

    let total_train_start = Instant::now();
    let trained = training::train(&args, &model, &train_samples, &mut hybrid_params, resumed);
    println!(
        "\nTraining completed in {:.2} seconds.",
        total_train_start.elapsed().as_secs_f64()
    );

    let ev = report::evaluate(
        &model,
        &test_samples,
        &hybrid_params,
        init_mse,
        &mut out_state,
    );
    let leads = report::inverse_leads(&test_samples, &ev.preds);
    report::save_outputs(
        &args,
        &model,
        (&train_samples, &test_samples),
        &hybrid_params,
        trained,
        ev,
        leads,
    );
}
