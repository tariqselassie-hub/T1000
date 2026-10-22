// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! T-1000 Sovereign ZINC 250k Dual Chemical Search Engine:
//! 1. Search 3: Quantum Chemical Anomaly & Bioactivity Outlier Discovery
//! 2. Search 1: Frontier Pareto Drug Candidate Mining
//!
//! Evaluates all 249,455 drug candidates across 12-qubit Hilbert space.

use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS; // 4,096

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QnnFusedCheckpoint {
    pub timestamp_rfc3339: String,
    pub n_qubits: usize,
    pub depth: usize,
    pub full_params: usize,
    pub pruned_params: usize,
    pub fused_active_params: usize,
    pub epochs_completed: usize,
    pub fused_hybrid_params_98: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct RawMolecule {
    pub smiles: String,
    pub logp: f64,
    pub qed: f64,
    pub sas: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub rank: usize,
    pub smiles: String,
    pub true_logp: f64,
    pub pred_logp: f64,
    pub true_qed: f64,
    pub pred_qed: f64,
    pub true_sas: f64,
    pub pred_sas: f64,
    pub pareto_score: f64,
    pub qcd_anomaly: f64,
    pub hilbert_dispersion: f64,
    pub total_anomaly_score: f64,
    pub z_expectations: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualSearchSummary {
    pub total_screened: usize,
    pub inference_wall_clock_ms: f64,
    pub throughput_mols_per_sec: f64,
    pub top_anomalies_count: usize,
    pub top_pareto_count: usize,
    pub mean_pareto_score: f64,
    pub max_pareto_score: f64,
    pub max_anomaly_score: f64,
}

#[inline]
pub fn featurize_smiles(s: &str) -> [f64; 19] {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut n_cl = 0usize;
    let mut n_br = 0usize;
    let mut i = 0;
    while i < len {
        if i + 1 < len {
            if bytes[i] == b'C' && bytes[i + 1] == b'l' {
                n_cl += 1;
                i += 2;
                continue;
            }
            if bytes[i] == b'B' && bytes[i + 1] == b'r' {
                n_br += 1;
                i += 2;
                continue;
            }
        }
        i += 1;
    }

    let mut n_c = 0usize;
    let mut n_n = 0usize;
    let mut n_o = 0usize;
    let mut n_f = 0usize;
    let mut n_s = 0usize;
    let mut n_p = 0usize;
    let mut n_arom = 0usize;
    let mut n_double = 0usize;
    let mut n_triple = 0usize;
    let mut n_branches = 0usize;
    let mut n_chiral = 0usize;
    let mut n_digits = 0usize;

    i = 0;
    while i < len {
        if i + 1 < len
            && ((bytes[i] == b'C' && bytes[i + 1] == b'l')
                || (bytes[i] == b'B' && bytes[i + 1] == b'r'))
        {
            i += 2;
            continue;
        }
        match bytes[i] {
            b'C' => n_c += 1,
            b'c' => {
                n_c += 1;
                n_arom += 1;
            }
            b'N' => n_n += 1,
            b'n' => {
                n_n += 1;
                n_arom += 1;
            }
            b'O' => n_o += 1,
            b'o' => {
                n_o += 1;
                n_arom += 1;
            }
            b'F' => n_f += 1,
            b'S' => n_s += 1,
            b's' => {
                n_s += 1;
                n_arom += 1;
            }
            b'P' | b'p' => n_p += 1,
            b'=' => n_double += 1,
            b'#' => n_triple += 1,
            b'(' => n_branches += 1,
            b'@' => n_chiral += 1,
            b'0'..=b'9' => n_digits += 1,
            _ => {}
        }
        i += 1;
    }

    let n_rings = (n_digits / 2).max(1);
    let heavy_atoms = (n_c + n_n + n_o + n_f + n_cl + n_br + n_s + n_p).max(1);
    let mut mw_est = n_c as f64 * 12.011
        + n_n as f64 * 14.007
        + n_o as f64 * 15.999
        + n_f as f64 * 18.998
        + n_cl as f64 * 35.453
        + n_br as f64 * 79.904
        + n_s as f64 * 32.065
        + n_p as f64 * 30.974;

    let h_est_val =
        n_c as i64 * 2 + 2 - n_double as i64 * 2 - n_triple as i64 * 4 - n_rings as i64 * 2;
    mw_est += h_est_val.max(0) as f64 * 1.008;

    let arom_ratio = n_arom as f64 / heavy_atoms as f64;
    let hetero_count = n_n + n_o + n_f + n_cl + n_br + n_s + n_p;
    let hetero_ratio = hetero_count as f64 / heavy_atoms as f64;
    let halogen_count = n_f + n_cl + n_br;

    [
        heavy_atoms as f64,
        mw_est,
        n_c as f64,
        n_n as f64,
        n_o as f64,
        n_s as f64,
        n_f as f64,
        n_cl as f64,
        n_br as f64,
        halogen_count as f64,
        n_arom as f64,
        arom_ratio,
        hetero_count as f64,
        hetero_ratio,
        n_double as f64,
        n_triple as f64,
        n_branches as f64,
        n_chiral as f64,
        n_rings as f64,
    ]
}

fn parse_zinc_csv(csv_path: &Path) -> Result<Vec<RawMolecule>, String> {
    let content = fs::read_to_string(csv_path)
        .map_err(|e| format!("Failed to read {}: {}", csv_path.display(), e))?;

    let mut molecules = Vec::with_capacity(250_000);
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut pos = 0;

    while pos < len && bytes[pos] != b'\n' {
        pos += 1;
    }
    if pos < len && bytes[pos] == b'\n' {
        pos += 1;
    }

    while pos < len {
        while pos < len && (bytes[pos] == b'\r' || bytes[pos] == b'\n' || bytes[pos] == b' ') {
            pos += 1;
        }
        if pos >= len {
            break;
        }

        let smiles: String;
        if bytes[pos] == b'"' {
            pos += 1;
            let start = pos;
            while pos < len && bytes[pos] != b'"' {
                pos += 1;
            }
            let raw_s = &bytes[start..pos];
            if pos < len && bytes[pos] == b'"' {
                pos += 1;
            }
            smiles = String::from_utf8_lossy(raw_s)
                .replace(['\r', '\n'], "")
                .trim()
                .to_string();
        } else {
            let start = pos;
            while pos < len && bytes[pos] != b',' && bytes[pos] != b'\n' && bytes[pos] != b'\r' {
                pos += 1;
            }
            smiles = String::from_utf8_lossy(&bytes[start..pos])
                .trim()
                .to_string();
        }

        if smiles.is_empty() {
            while pos < len && bytes[pos] != b'\n' {
                pos += 1;
            }
            if pos < len {
                pos += 1;
            }
            continue;
        }

        if pos < len && bytes[pos] == b',' {
            pos += 1;
        }

        let line_start = pos;
        while pos < len && bytes[pos] != b'\n' {
            pos += 1;
        }
        let line_end = pos;
        if pos < len && bytes[pos] == b'\n' {
            pos += 1;
        }

        let num_slice = String::from_utf8_lossy(&bytes[line_start..line_end]);
        let parts: Vec<&str> = num_slice.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 3 {
            if let (Ok(lp), Ok(qd), Ok(ss)) = (
                parts[0].parse::<f64>(),
                parts[1].parse::<f64>(),
                parts[2].parse::<f64>(),
            ) {
                molecules.push(RawMolecule {
                    smiles,
                    logp: lp,
                    qed: qd,
                    sas: ss,
                });
            }
        }
    }
    Ok(molecules)
}

fn resolve_path(rel: &str) -> PathBuf {
    let target = Path::new(rel);
    if target.exists() {
        return target.to_path_buf();
    }
    if let Ok(cwd) = env::current_dir() {
        let mut curr = cwd.as_path();
        for _ in 0..6 {
            let candidate = curr.join(rel);
            if candidate.exists() {
                return candidate;
            }
            if let Some(parent) = curr.parent() {
                curr = parent;
            } else {
                break;
            }
        }
    }
    target.to_path_buf()
}

fn export_hits_csv(path: &Path, hits: &[SearchHit]) -> Result<(), std::io::Error> {
    let mut lines = Vec::with_capacity(hits.len() + 1);
    lines.push("rank,smiles,true_logp,pred_logp,true_qed,pred_qed,true_sas,pred_sas,pareto_score,qcd_anomaly,hilbert_dispersion,total_anomaly_score".to_string());
    for h in hits {
        lines.push(format!(
            "{},\"{}\",{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},{:.6},{:.6},{:.6}",
            h.rank,
            h.smiles,
            h.true_logp,
            h.pred_logp,
            h.true_qed,
            h.pred_qed,
            h.true_sas,
            h.pred_sas,
            h.pareto_score,
            h.qcd_anomaly,
            h.hilbert_dispersion,
            h.total_anomaly_score
        ));
    }
    t1000::results::write(path, lines.join("\n"))
}

fn main() {
    println!("=========================================================================");
    println!("    T-1000 SOVEREIGN ZINC 250k DUAL SEARCH: ANOMALIES (3) & PARETO (1)   ");
    println!("=========================================================================");

    let default_raw_path = PathBuf::from(
        r"C:\Users\zoddj\zenith_workspace_divizion\GuddaLM\data\zinc_250\250k_rndm_zinc_drugs_clean_3.csv",
    );
    let raw_csv_path = if default_raw_path.exists() {
        default_raw_path
    } else {
        resolve_path("data/zinc_250k.csv")
    };
    let raw_mols = match parse_zinc_csv(&raw_csv_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[-] Error reading CSV: {}", e);
            return;
        }
    };
    let n_mols = raw_mols.len();
    println!("  -> Ingested {} molecules.", n_mols);

    let ckpt_path = resolve_path(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_checkpoint_fused98.json",
    );
    let ckpt_str = fs::read_to_string(&ckpt_path).expect("Failed to read checkpoint");
    let ckpt: QnnFusedCheckpoint = serde_json::from_str(&ckpt_str).expect("Failed to parse JSON");
    let fused_params = ckpt.fused_hybrid_params_98;

    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);
    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(20);

    let chunk_size = (n_mols + num_threads - 1) / num_threads;
    let chunks: Vec<&[RawMolecule]> = raw_mols.chunks(chunk_size).collect();

    let ansatz_ref = &ansatz;
    let encoding_ref = &encoding;
    let p_ref = &fused_params;
    let theta_y = &p_ref[48..59];
    let w_off = 59;
    let b_off = 59 + 36;

    let t_start = Instant::now();

    let thread_results: Vec<Vec<SearchHit>> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(chunks.len());
        for &chunk in &chunks {
            handles.push(s.spawn(move || {
                let mut l2_state = QuantumState::new(DIM);
                let mut z_fused_11 = [0.0; 11];
                let mut out = Vec::with_capacity(chunk.len());

                for mol in chunk {
                    let feats = featurize_smiles(&mol.smiles);
                    encoding_ref.encode_amplitude_into(&feats, &mut l2_state);
                    ansatz_ref.forward_l2_in_place(&mut l2_state, &p_ref[..48]);
                    t1000::simd::measure_fused_ry_z_readout(
                        &l2_state.real,
                        &l2_state.imag,
                        theta_y,
                        &mut z_fused_11,
                    );

                    let mut p_logp = p_ref[b_off + 0];
                    let mut p_qed = p_ref[b_off + 1];
                    let mut p_sas = p_ref[b_off + 2];
                    for q in 0..11 {
                        p_logp += p_ref[w_off + q] * z_fused_11[q];
                        p_qed += p_ref[w_off + 1 * 12 + q] * z_fused_11[q];
                        p_sas += p_ref[w_off + 2 * 12 + q] * z_fused_11[q];
                    }

                    let lp_term = ((mol.logp - 2.5).powi(2) / 4.5).exp();
                    let pareto_score = (mol.qed / mol.sas.max(0.1)) / lp_term;

                    let qcd_anomaly = (p_logp - mol.logp).abs()
                        + 2.0 * (p_qed - mol.qed).abs()
                        + (p_sas - mol.sas).abs();

                    // Hilbert space expectation dispersion norm
                    let hilbert_disp = z_fused_11.iter().map(|&z| z * z).sum::<f64>().sqrt();
                    let total_anomaly = qcd_anomaly + 2.0 * hilbert_disp;

                    out.push(SearchHit {
                        rank: 0,
                        smiles: mol.smiles.clone(),
                        true_logp: mol.logp,
                        pred_logp: p_logp,
                        true_qed: mol.qed,
                        pred_qed: p_qed,
                        true_sas: mol.sas,
                        pred_sas: p_sas,
                        pareto_score,
                        qcd_anomaly,
                        hilbert_dispersion: hilbert_disp,
                        total_anomaly_score: total_anomaly,
                        z_expectations: z_fused_11.to_vec(),
                    });
                }
                out
            }));
        }

        let mut collected = Vec::with_capacity(chunks.len());
        for h in handles {
            collected.push(h.join().unwrap());
        }
        collected
    });

    let el_time = t_start.elapsed();
    let tp_mols = n_mols as f64 / el_time.as_secs_f64();
    println!(
        "  -> Screened {} molecules in {:.2} ms ({:.1} mols/sec)",
        n_mols,
        el_time.as_secs_f64() * 1000.0,
        tp_mols
    );

    let mut all_hits: Vec<SearchHit> = Vec::with_capacity(n_mols);
    for sub in thread_results {
        all_hits.extend(sub);
    }

    // SEARCH 3: TOP QUANTUM ANOMALIES & OUTLIERS
    let mut anomaly_hits = all_hits.clone();
    anomaly_hits.sort_by(|a, b| {
        b.total_anomaly_score
            .partial_cmp(&a.total_anomaly_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, h) in anomaly_hits.iter_mut().take(100).enumerate() {
        h.rank = i + 1;
    }
    let top_100_anomalies: Vec<SearchHit> = anomaly_hits.into_iter().take(100).collect();

    // SEARCH 1: TOP FRONTIER PARETO DRUG CANDIDATES
    let mut pareto_hits = all_hits;
    pareto_hits.sort_by(|a, b| {
        b.pareto_score
            .partial_cmp(&a.pareto_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for (i, h) in pareto_hits.iter_mut().take(100).enumerate() {
        h.rank = i + 1;
    }
    let top_100_pareto: Vec<SearchHit> = pareto_hits.into_iter().take(100).collect();

    // SAVE RESULTS
    let out_dir = resolve_path("results/experiments");
    fs::create_dir_all(&out_dir).ok();

    let anom_json = out_dir.join("zinc_quantum_anomalies.json");
    let anom_csv = out_dir.join("zinc_quantum_anomalies.csv");
    let pareto_json = out_dir.join("zinc_pareto_leads.json");
    let pareto_csv = out_dir.join("zinc_pareto_leads.csv");
    let sum_json = out_dir.join("zinc_dual_search_summary.json");

    t1000::results::write(
        &anom_json,
        serde_json::to_string_pretty(&top_100_anomalies).unwrap(),
    )
    .unwrap();
    export_hits_csv(&anom_csv, &top_100_anomalies).unwrap();

    t1000::results::write(
        &pareto_json,
        serde_json::to_string_pretty(&top_100_pareto).unwrap(),
    )
    .unwrap();
    export_hits_csv(&pareto_csv, &top_100_pareto).unwrap();

    let mean_pareto =
        top_100_pareto.iter().map(|h| h.pareto_score).sum::<f64>() / top_100_pareto.len() as f64;
    let summary = DualSearchSummary {
        total_screened: n_mols,
        inference_wall_clock_ms: el_time.as_secs_f64() * 1000.0,
        throughput_mols_per_sec: tp_mols,
        top_anomalies_count: top_100_anomalies.len(),
        top_pareto_count: top_100_pareto.len(),
        mean_pareto_score: mean_pareto,
        max_pareto_score: top_100_pareto[0].pareto_score,
        max_anomaly_score: top_100_anomalies[0].total_anomaly_score,
    };
    t1000::results::write(&sum_json, serde_json::to_string_pretty(&summary).unwrap()).unwrap();

    println!("\n=========================================================================");
    println!("             SEARCH 3: TOP 5 QUANTUM CHEMICAL ANOMALIES & OUTLIERS       ");
    println!("=========================================================================");
    for h in top_100_anomalies.iter().take(5) {
        println!(
            " #{} | Anomaly Score: {:.4} (QCD: {:.4}, Hilbert: {:.4})",
            h.rank, h.total_anomaly_score, h.qcd_anomaly, h.hilbert_dispersion
        );
        println!(
            "     logP: {:.3} (pred: {:.3}) | QED: {:.3} (pred: {:.3}) | SAS: {:.3} (pred: {:.3})",
            h.true_logp, h.pred_logp, h.true_qed, h.pred_qed, h.true_sas, h.pred_sas
        );
        println!("     SMILES: {}", h.smiles);
    }

    println!("\n=========================================================================");
    println!("             SEARCH 1: TOP 5 FRONTIER PARETO DRUG CANDIDATES             ");
    println!("=========================================================================");
    for h in top_100_pareto.iter().take(5) {
        println!(
            " #{} | Pareto Score: {:.4} | logP: {:.3} | QED: {:.3} | SAS: {:.3}",
            h.rank, h.pareto_score, h.true_logp, h.true_qed, h.true_sas
        );
        println!("     SMILES: {}", h.smiles);
    }
    println!("=========================================================================\n");
}
