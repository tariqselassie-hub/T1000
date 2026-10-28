// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use std::fs;
use std::path::PathBuf;
use t1000::{CircuitAnsatz, DataEncodingLayer, EncodingKind, QuantumState};

const N_QUBITS: usize = 12;
const DIM: usize = 1 << N_QUBITS;

fn main() {
    let smi = "O=[N+]([O-])c1ccc2c(c1)=[NH+][C@@H](CS[C@H]1[NH+]=c3ccc(Br)cc3=[NH+]1)[NH+]=2";

    // Featurize
    let feats = [
        24.0, 408.279, 15.0, 5.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0, 12.0, 0.5, 9.0, 0.375, 5.0, 0.0,
        4.0, 3.0, 4.0,
    ];

    let ckpt_path = PathBuf::from(
        "tests/benchmarks/zinc_molecular/outputs/zinc_12qubit_checkpoint_fused98.json",
    );
    let ckpt_str = fs::read_to_string(&ckpt_path).expect("Read ckpt");
    let v: serde_json::Value = serde_json::from_str(&ckpt_str).expect("Parse json");
    let p_arr = v["fused_hybrid_params_98"]
        .as_array()
        .expect("params array");
    let p: Vec<f64> = p_arr.iter().map(|x| x.as_f64().unwrap()).collect();

    let ansatz = CircuitAnsatz::new(N_QUBITS, 3).with_pi_layer(true);
    let encoding = DataEncodingLayer::new(N_QUBITS, EncodingKind::Amplitude);

    let mut l2_state = QuantumState::new(DIM);
    let mut z_fused_11 = [0.0f64; 11];

    encoding.encode_amplitude_into(&feats, &mut l2_state);
    ansatz.forward_l2_in_place(&mut l2_state, &p[..48]);
    t1000::simd::measure_fused_ry_z_readout(
        &l2_state.real,
        &l2_state.imag,
        &p[48..59],
        &mut z_fused_11,
    );

    let w_off = 59;
    let b_off = 59 + 36;
    let mut p_logp = p[b_off + 0];
    let mut p_qed = p[b_off + 1];
    let mut p_sas = p[b_off + 2];
    for q in 0..11 {
        p_logp += p[w_off + q] * z_fused_11[q];
        p_qed += p[w_off + 1 * 12 + q] * z_fused_11[q];
        p_sas += p[w_off + 2 * 12 + q] * z_fused_11[q];
    }

    println!("=========================================================================");
    println!(" EXACT QUANTUM EXPECTATION VECTOR FOR MOLECULE #10640                   ");
    println!("=========================================================================");
    println!(" SMILES: {}", smi);
    println!(" Pred logP : {:.4}", p_logp);
    println!(" Pred QED  : {:.4}", p_qed);
    println!(" Pred SAS  : {:.4}", p_sas);
    println!("-------------------------------------------------------------------------");
    for q in 0..11 {
        println!(" <Z_{:02}>   : {:>+10.6}", q, z_fused_11[q]);
    }
    let mean_z: f64 = z_fused_11.iter().sum::<f64>() / 11.0;
    println!(" Mean <Z>  : {:>+10.6}", mean_z);
    println!("=========================================================================");
}
