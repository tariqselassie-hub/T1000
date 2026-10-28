// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use super::*;
use std::env;

/// Command-line options.
#[derive(Debug)]
pub struct Args {
    pub num_epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub max_train: usize,
    pub max_eval: usize,
    pub depth: usize,        // quantum layers
    pub qed_weight: f64,     // multiplier balancing QED gradients (Pareto-optimal)
    pub max_norm: f64,       // adaptive gradient clipping ceiling; <= 0 disables
    pub out: Option<String>, // report path override
    pub resume: Option<String>,
    pub save_weights: Option<String>,
    pub num_threads: usize,
}

impl Args {
    pub fn parse() -> Self {
        let args: Vec<String> = env::args().collect();
        let mut a = Args {
            num_epochs: 5,
            batch_size: 50,
            lr: 0.02,
            max_train: 10000,
            max_eval: 2000,
            depth: 3,
            qed_weight: 2.2,
            max_norm: 6.0,
            out: None,
            resume: None,
            save_weights: None,
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
                "--lr" => next.map(|v| a.lr = v.parse().unwrap_or(0.02)).is_some(),
                "--depth" => next.map(|v| a.depth = v.parse().unwrap_or(3)).is_some(),
                "--qed-weight" => next
                    .map(|v| a.qed_weight = v.parse().unwrap_or(2.2))
                    .is_some(),
                "--max-norm" => next
                    .map(|v| a.max_norm = v.parse().unwrap_or(6.0))
                    .is_some(),
                "--resume" => next.map(|v| a.resume = Some(v.to_string())).is_some(),
                "--save-weights" => next.map(|v| a.save_weights = Some(v.to_string())).is_some(),
                "--out" => next.map(|v| a.out = Some(v.to_string())).is_some(),
                "--samples" => next
                    .map(|v| a.max_train = v.parse().unwrap_or(10000))
                    .is_some(),
                "--eval-samples" => next
                    .map(|v| a.max_eval = v.parse().unwrap_or(2000))
                    .is_some(),
                "--threads" => next
                    .map(|v| a.num_threads = v.parse().unwrap_or(8))
                    .is_some(),
                _ => false,
            };
            i += 1 + usize::from(consumed);
        }
        a
    }

    pub fn print_banner(&self, layout: &Layout) {
        println!("{DOUBLE_RULE}");
        println!("     GUDDANETWORKS QNN: 12-QUBIT HYBRID VARIATIONAL QUANTUM SUITE        ");
        println!("{DOUBLE_RULE}");
        println!(" Hilbert Space   : 12 Qubits | Dim = 4,096 Complex Floats (64 KB L2 Cache)");
        println!(
            " Layers          : {} Quantum Variational Layers + 1 Trainable Readout Head",
            self.depth
        );
        println!(
            " Trainable Params: {} Total ({} Quantum Rotations + {} Classical Readout)",
            layout.n_total,
            layout.n_quantum,
            layout.n_weights + layout.n_biases
        );
        println!(
            " Training Set    : {} Featurized Drug Molecules (ZINC 250k)",
            self.max_train
        );
        println!(
            " Evaluation Set  : {} Holdout Drug Molecules",
            self.max_eval
        );
        let max_norm_str = if self.max_norm > 0.0 {
            format!("{:.2}", self.max_norm)
        } else {
            "None (Unclipped)".to_string()
        };
        println!(
            " Hyperparameters : Epochs = {}, Batch Size = {}, LR = {:.3}, Threads = {}, Max Norm = {}",
            self.num_epochs, self.batch_size, self.lr, self.num_threads, max_norm_str
        );
        println!(
            " Objectives      : Multi-Task Joint Optimization on logP, QED, SAS (QED Weight = {:.2}x)",
            self.qed_weight
        );
        println!("{DOUBLE_RULE}");
    }
}
