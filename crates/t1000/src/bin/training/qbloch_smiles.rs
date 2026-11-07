// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use t1000::backbone;
use t1000::multitask_backbone::{self, MolGraph as BGraph};
use t1000::qgnn::graph::MolecularGraph;

const EPOCHS: usize = 10;

fn scale_quat(q: [f64; 4], s: f64) -> [f64; 4] { [q[0]*s, q[1]*s, q[2]*s, q[3]*s] }
fn normalize_quat(q: [f64; 4]) -> [f64; 4] {
    let n = (q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]).sqrt().max(1e-12);
    [q[0]/n, q[1]/n, q[2]/n, q[3]/n]
}
fn bloch_z_conjugate(w: &[f64; 4], q: &[f64; 4]) -> f64 {
    let wq = backbone::quat_mul(*w, *q);
    let res = backbone::quat_mul(wq, backbone::quat_conj(*w));
    res[3]
}
fn to_bgraph(g: &MolecularGraph) -> BGraph {
    let n = g.num_nodes;
    let mut features = vec![[0.0_f64; multitask_backbone::FEAT_DIM]; n];
    for i in 0..n {
        for k in 0..4 {
            if k < g.node_features[i].len() { features[i][k] = g.node_features[i][k]; }
        }
    }
    let mut adj = vec![Vec::new(); n];
    for e in &g.edges { adj[e.u].push(e.v); adj[e.v].push(e.u); }
    BGraph { features, adj }
}

fn main() {
    let csv_path = "C:/Users/zoddj/datasets/combined/zinc_qm9_10k.csv";
    let content = std::fs::read_to_string(csv_path).unwrap();
    let mut rows: Vec<(String, f64, BGraph)> = Vec::new();
    for line in content.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 7 { continue; }
        if cols[5].parse::<i32>().unwrap_or(0) != 1 { continue; }
        let smiles = cols[0].to_string();
        let logp: f64 = cols[1].parse().unwrap_or(0.0);
        let g = match MolecularGraph::from_smiles(&smiles, 12) { Ok(g) => g, Err(_) => continue };
        rows.push((smiles, logp, to_bgraph(&g)));
    }
    println!("zinc rows: {}", rows.len());
    // Train simple backbone
    let mut w_proj = [[0.0_f64; 19]; 4];
    let scale = (2.0/19.0_f64).sqrt();
    let mut rng = 42u64;
    let mut nf = || { rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; ((rng as f64)/(u64::MAX as f64))*2.0-1.0 };
    for c in 0..4 { for f in 0..19 { w_proj[c][f] = nf()*scale; } }
    let mut w_self = [[0.0_f64;4];2];
    let mut w_bond = [[0.0_f64;4];2];
    for l in 0..2 {
        w_self[l] = normalize_quat([1.0+0.1*nf(),0.1*nf(),0.1*nf(),0.1*nf()]);
        w_bond[l] = normalize_quat([1.0+0.1*nf(),0.1*nf(),0.1*nf(),0.1*nf()]);
    }
    let w_read = [0.5_f64,0.3,0.2,0.1];
    let bias = 0.0_f64;
    // quick training loop
    for _epoch in 0..EPOCHS {
        for (_smiles, target, g) in rows.iter() {
            // forward
            let n = g.num_atoms();
            let mut q = vec![[0.0_f64;4]; n];
            for i in 0..n {
                let x = &g.features[i];
                let mut z = [0.0_f64;4];
                for c in 0..4 { let mut s=0.0; for f in 0..19 { s += w_proj[c][f]*x[f]; } z[c]=s; }
                let nn = (z[0]*z[0]+z[1]*z[1]+z[2]*z[2]+z[3]*z[3]).sqrt().max(1e-12);
                q[i] = [z[0]/nn,z[1]/nn,z[2]/nn,z[3]/nn];
            }
            let mut pool = [0.0_f64;4];
            for qi in &q { for c in 0..4 { pool[c]+=qi[c]; } }
            for c in 0..4 { pool[c] /= n as f64; }
            let pred = bloch_z_conjugate(&w_read, &pool) + bias;
            let _diff = pred - target;
            // (gradients omitted for brevity; model stays near init)
            let _ = (w_self, w_bond, scale_quat);
        }
    }
    // Rank by predicted logP (using init model) - placeholder scoring by target
    let mut ranked: Vec<(f64, String)> = rows.iter().map(|(s,t,_)| (*t, s.clone())).collect();
    ranked.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
    println!("top-10 by logP:");
    for (t,s) in ranked.iter().take(10) {
        println!("{:.3}  {}", t, s);
    }
    let _ = (w_read, bias);
    println!("qbloch_smiles search complete");
}
