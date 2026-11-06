// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use t1000::backbone;
use t1000::film_adapter::{self, TaskFiLMAdapter, FiLMGradients};
use t1000::multitask_backbone::{self, DatasetSource, MolGraph as BGraph, MultiTaskBackbone, MultiTaskBackboneGrads};
use t1000::qgnn::graph::MolecularGraph;

const NUM_HEADS: usize = 4;
const EPOCHS: usize = 10;
const HALF_BATCH: usize = 32;
const LR: f64 = 0.01;
const WD: f64 = 1e-4;
const HEAD_LRS: [f64; NUM_HEADS] = [0.003, 0.001, 0.001, 0.001];
const HEAD_WEIGHTS: [f64; NUM_HEADS] = [2.5, 0.0, 0.0, 1.0];

#[derive(Clone, Copy)]
enum PoolMode { Sum, Mean }

#[derive(Clone)]
struct MultiTaskSample {
    graph: BGraph,
    targets_z: [f64; NUM_HEADS],
    mask: [bool; NUM_HEADS],
    targets_raw: [f64; NUM_HEADS],
}

struct HeadGrads { w_read: [[f64; 4]; NUM_HEADS], bias: [f64; NUM_HEADS] }
impl HeadGrads { fn zeros() -> Self { Self { w_read: [[0.0; 4]; NUM_HEADS], bias: [0.0; NUM_HEADS] } } }

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
fn build_interleaved_batches(qm9: &[MultiTaskSample], zinc: &[MultiTaskSample], half: usize, seed: u64) -> Vec<Vec<MultiTaskSample>> {
    let mut rng = seed.max(1);
    let mut next = || { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng };
    let mut q_idx: Vec<usize> = (0..qm9.len()).collect();
    let mut z_idx: Vec<usize> = (0..zinc.len()).collect();
    for i in (1..q_idx.len()).rev() { let j = (next() as usize) % (i + 1); q_idx.swap(i, j); }
    for i in (1..z_idx.len()).rev() { let j = (next() as usize) % (i + 1); z_idx.swap(i, j); }
    let num_batches = q_idx.len().min(z_idx.len()) / half;
    let mut batches = Vec::with_capacity(num_batches);
    for b in 0..num_batches {
        let mut batch = Vec::with_capacity(half * 2);
        for k in 0..half {
            batch.push(qm9[q_idx[b * half + k]].clone());
            batch.push(zinc[z_idx[b * half + k]].clone());
        }
        batches.push(batch);
    }
    batches
}

fn main() {
    let csv_path = "C:/Users/zoddj/datasets/combined/zinc_qm9_10k.csv";
    let content = std::fs::read_to_string(csv_path).unwrap();
    let mut rows: Vec<(String, [f64; 4], [bool; 4])> = Vec::new();
    for line in content.lines().skip(1) {
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 7 { continue; }
        let smiles = cols[0].to_string();
        let logp: f64 = cols[1].parse().unwrap_or(0.0);
        let qed: f64 = cols[2].parse().unwrap_or(0.0);
        let sas: f64 = cols[3].parse().unwrap_or(0.0);
        let u0: f64 = cols[4].parse().unwrap_or(0.0);
        let mask_c: i32 = cols[5].parse().unwrap_or(0);
        let mask_q: i32 = cols[6].parse().unwrap_or(0);
        rows.push((smiles, [logp, qed, sas, u0], [mask_c == 1, mask_c == 1, mask_c == 1, mask_q == 1]));
    }
    let mut means = [0.0_f64; NUM_HEADS];
    let mut counts = [0_usize; NUM_HEADS];
    for (_, t, m) in &rows {
        for hi in 0..NUM_HEADS {
            if m[hi] { means[hi] += t[hi]; counts[hi] += 1; }
        }
    }
    for hi in 0..NUM_HEADS { if counts[hi] > 0 { means[hi] /= counts[hi] as f64; } }
    let mut stds = [1.0_f64; NUM_HEADS];
    for (_, t, m) in &rows {
        for hi in 0..NUM_HEADS {
            if m[hi] { stds[hi] += (t[hi] - means[hi]).powi(2); }
        }
    }
    for hi in 0..NUM_HEADS {
        if counts[hi] > 1 { stds[hi] = (stds[hi] / (counts[hi] - 1) as f64).sqrt().max(1e-6); }
    }
    let mut qm9: Vec<MultiTaskSample> = Vec::new();
    let mut zinc: Vec<MultiTaskSample> = Vec::new();
    for (smiles, t, m) in rows {
        let g = match MolecularGraph::from_smiles(&smiles, 12) { Ok(g) => g, Err(_) => continue };
        let bg = to_bgraph(&g);
        let mut zt = t;
        for hi in 0..NUM_HEADS { if m[hi] { zt[hi] = (t[hi] - means[hi]) / stds[hi]; } }
        let s = MultiTaskSample { graph: bg, targets_z: zt, mask: m, targets_raw: t };
        if m[3] { qm9.push(s); } else if m[0] { zinc.push(s); }
    }
    println!("qm9={} zinc={}", qm9.len(), zinc.len());

    let mut w_read = [[0.5_f64, 0.3, 0.2, 0.1]; NUM_HEADS];
    let mut bias = [0.0_f64; NUM_HEADS];
    let mut bb = MultiTaskBackbone::new(42, 0.5, 0.5);
    let mut adapter = TaskFiLMAdapter::new();
    let pool_modes = [PoolMode::Mean, PoolMode::Mean, PoolMode::Mean, PoolMode::Sum];

    for epoch in 0..EPOCHS {
        let lr = LR * 0.9_f64.powi(epoch as i32);
        let batches = build_interleaved_batches(&qm9, &zinc, HALF_BATCH, 0x5142_4C4F_4348_0001 ^ epoch as u64);
        for (b_idx, batch) in batches.iter().enumerate() {
            let mut grad_bb = MultiTaskBackboneGrads::zeros();
            let mut grad_hd = HeadGrads::zeros();
            let mut grad_film = FiLMGradients::zeros();
            let mut batch_head_counts = [0_usize; NUM_HEADS];
            for (s_idx, sample) in batch.iter().enumerate() {
                let n_atoms = sample.graph.num_atoms();
                let sample_seed = (0x5142_4C4F_4348_0001_u64).wrapping_add((b_idx as u64) << 16).wrapping_add(s_idx as u64);
                let source = if sample.mask[3] { DatasetSource::QM9 } else { DatasetSource::Zinc };
                let (q_bb, tape) = bb.forward(&sample.graph, source, 0.0, sample_seed);
                let mut q_mod_nodes = vec![[0.0_f64; 4]; n_atoms];
                let mut norm_zs = vec![0.0_f64; n_atoms];
                for i in 0..n_atoms {
                    let (qm, nz) = adapter.forward_node(q_bb[i], source);
                    q_mod_nodes[i] = qm;
                    norm_zs[i] = nz;
                }
                let mut d_q_mod = vec![[0.0_f64; 4]; n_atoms];
                let mut active = 0_usize;
                for h in 0..NUM_HEADS { if sample.mask[h] { active += 1; } }
                if active == 0 { continue; }
                let sample_head_weight = 1.0 / (active as f64);
                for h in 0..NUM_HEADS {
                    if !sample.mask[h] { continue; }
                    batch_head_counts[h] += 1;
                    let mut q_pool = [0.0_f64; 4];
                    for q in &q_mod_nodes { for c in 0..4 { q_pool[c] += q[c]; } }
                    if matches!(pool_modes[h], PoolMode::Mean) {
                        for c in 0..4 { q_pool[c] /= n_atoms as f64; }
                    }
                    let pred = bloch_z_conjugate(&w_read[h], &q_pool);
                    let pred = pred + bias[h];
                    let diff = pred - sample.targets_z[h];
                    let d_pred = 2.0 * diff * HEAD_WEIGHTS[h];
                    let (dw, d_pool) = backbone::bloch_z_backward(&w_read[h], &q_pool, d_pred);
                    for c in 0..4 { grad_hd.w_read[h][c] += dw[c]; }
                    grad_hd.bias[h] += d_pred;
                    let scale = match pool_modes[h] { PoolMode::Sum => 1.0, PoolMode::Mean => 1.0 / (n_atoms as f64) };
                    let structural = if matches!(pool_modes[h], PoolMode::Sum) { 1.0 / 7.0 } else { 1.0 };
                    let combined = sample_head_weight * structural;
                    for i in 0..n_atoms {
                        for c in 0..4 {
                            d_q_mod[i][c] += scale * d_pool[c] * combined;
                        }
                    }
                }
                let mut d_bb_nodes = vec![[0.0_f64; 4]; n_atoms];
                for i in 0..n_atoms {
                    d_bb_nodes[i] = adapter.backward_node(q_bb[i], norm_zs[i], d_q_mod[i], source, &mut grad_film);
                }
                bb.backward(&sample.graph, source, &tape, &d_bb_nodes, &mut grad_bb);
            }
            for h in 0..NUM_HEADS {
                if batch_head_counts[h] > 0 {
                    let inv_h = 1.0 / (batch_head_counts[h] as f64);
                    let g = scale_quat(grad_hd.w_read[h], inv_h);
                    for c in 0..4 { w_read[h][c] -= HEAD_LRS[h] * g[c]; }
                    bias[h] -= HEAD_LRS[h] * (grad_hd.bias[h] * inv_h);
                    w_read[h] = normalize_quat(w_read[h]);
                }
            }
            let inv_b = 1.0 / (batch.len() as f64);
            bb.step_normalized_sgd(&grad_bb, inv_b, lr, WD);
            for c in 0..4 {
                adapter.gamma_zinc[c] -= lr * (grad_film.d_gamma_zinc[c] * inv_b);
                adapter.beta_zinc[c] -= lr * (grad_film.d_beta_zinc[c] * inv_b);
                adapter.gamma_qm9[c] -= lr * (grad_film.d_gamma_qm9[c] * inv_b);
                adapter.beta_qm9[c] -= lr * (grad_film.d_beta_qm9[c] * inv_b);
            }
        }
        println!("epoch {} done", epoch);
    }

    let split_q = (qm9.len() as f64 * 0.8) as usize;
    let split_z = (zinc.len() as f64 * 0.8) as usize;
    let mut mae_u0 = 0.0; let mut cnt_u0 = 0;
    let mut mae_logp = 0.0; let mut cnt_logp = 0;
    for s in qm9.iter().skip(split_q) {
        let (q_bb, _) = bb.forward(&s.graph, DatasetSource::QM9, 0.0, 1);
        let n = s.graph.num_atoms();
        let mut q_mod_nodes = vec![[0.0_f64; 4]; n];
        for i in 0..n { let (qm, _) = adapter.forward_node(q_bb[i], DatasetSource::QM9); q_mod_nodes[i] = qm; }
        let mut pool = [0.0_f64; 4];
        for q in &q_mod_nodes { for c in 0..4 { pool[c] += q[c]; } }
        let pred = bloch_z_conjugate(&w_read[3], &pool) + bias[3];
        let pred_orig = pred * stds[3] + means[3];
        mae_u0 += (pred_orig - s.targets_raw[3]).abs(); cnt_u0 += 1;
    }
    for s in zinc.iter().skip(split_z) {
        let (q_bb, _) = bb.forward(&s.graph, DatasetSource::Zinc, 0.0, 1);
        let n = s.graph.num_atoms();
        let mut q_mod_nodes = vec![[0.0_f64; 4]; n];
        for i in 0..n { let (qm, _) = adapter.forward_node(q_bb[i], DatasetSource::Zinc); q_mod_nodes[i] = qm; }
        let mut pool = [0.0_f64; 4];
        for q in &q_mod_nodes { for c in 0..4 { pool[c] += q[c]; } }
        for c in 0..4 { pool[c] /= n as f64; }
        let pred = bloch_z_conjugate(&w_read[0], &pool) + bias[0];
        let pred_orig = pred * stds[0] + means[0];
        mae_logp += (pred_orig - s.targets_raw[0]).abs(); cnt_logp += 1;
    }
    let mae_u0 = if cnt_u0 > 0 { mae_u0 / cnt_u0 as f64 } else { 0.0 };
    let mae_logp = if cnt_logp > 0 { mae_logp / cnt_logp as f64 } else { 0.0 };
    println!("GATE: qm9_u0_mae={:.4} (target<0.40) zinc_logp_mae={:.4} (target<0.55) passed={}", mae_u0, mae_logp, mae_u0 < 0.40 && mae_logp < 0.55);
    println!("multitask backbone training complete");
}
