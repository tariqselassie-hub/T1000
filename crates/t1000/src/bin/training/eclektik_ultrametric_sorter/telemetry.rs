// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Telemetry Reports & Ultrametric Geometry Invariants for ECLEKTIK.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageMetrics {
    pub code: String,
    pub name: String,
    pub family: String,
    pub total: usize,
    pub correct: usize,
    pub accuracy_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyMetrics {
    pub family: String,
    pub total: usize,
    pub correct: usize,
    pub accuracy_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryMetrics {
    pub identical_leaf_overlap: f64,
    pub sibling_family_overlap: f64,
    pub disjoint_family_overlap: f64,
    pub strong_triangle_inequality_holds: bool,
    pub gamma1_sibling: f64,
    pub gamma2_disjoint: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EclektikUltrametricReport {
    pub timestamp_rfc3339: String,
    pub dataset: String,
    pub registers: String,
    pub n_qubits: usize,
    pub total_parameters: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub training_time_sec: f64,
    pub inference_per_sample_us: f64,
    pub test_coarse_family_accuracy_pct: f64,
    pub test_fine_within_family_accuracy_pct: f64,
    pub test_joint_12way_accuracy_pct: f64,
    pub geometry: GeometryMetrics,
    pub family_metrics: Vec<FamilyMetrics>,
    pub language_metrics: Vec<LanguageMetrics>,
}

pub fn save_eclektik_report(report: &EclektikUltrametricReport) {
    let results_dir = Path::new("results/experiments");
    fs::create_dir_all(results_dir).unwrap();
    let json_path = results_dir.join("eclektik_ultrametric_report.json");
    let csv_path = results_dir.join("eclektik_ultrametric_report.csv");

    let json_str = serde_json::to_string_pretty(report).unwrap();
    let settings = format!(
        "qubits={};train={};test={};params={}",
        report.n_qubits, report.train_samples, report.test_samples, report.total_parameters
    );
    let score = t1000::results::Score {
        metric: "test_joint_12way_accuracy_pct",
        value: report.test_joint_12way_accuracy_pct,
        better: t1000::results::Better::Higher,
        guard: Some(t1000::results::Guard {
            name: "test_coarse_family_accuracy_pct",
            value: report.test_coarse_family_accuracy_pct,
            better: t1000::results::Better::Higher,
            tolerance: 1.0,
        }),
        settings: &settings,
    };

    let outcome = t1000::results::write_scored(&json_path, json_str, &score).unwrap();
    let csv_path = if outcome == t1000::results::Outcome::WrittenVariant {
        t1000::results::variant_path(&csv_path, &settings)
    } else {
        csv_path
    };

    let mut csv_file = t1000::results::create(&csv_path).unwrap();
    writeln!(csv_file, "code,name,family,total,correct,accuracy_pct").unwrap();
    for l in &report.language_metrics {
        writeln!(
            csv_file,
            "{},{},{},{},{},{:.2}",
            l.code, l.name, l.family, l.total, l.correct, l.accuracy_pct
        )
        .unwrap();
    }

    println!("\n [PERSISTENCE]");
    println!("   JSON Telemetry Saved : {}", json_path.display());
    println!("   CSV Telemetry Saved  : {}", csv_path.display());
}
