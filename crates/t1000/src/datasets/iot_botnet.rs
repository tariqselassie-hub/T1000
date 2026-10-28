// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! BoTNeTIoT-L01 N-BaIoT IoT Botnet Traffic Dataset Ingestion & Ultrametric Hierarchy.
//!
//! Maps 23 network stream statistics across 10-second decay windows into a
//! 3-Tier Cascaded Ultrametric Tree:
//!   - Tier 1 (Device Category, 2Q, 4 classes): Doorbells, Monitors, Provision Cams, Webcams.
//!   - Tier 2 (Attack Family, 1Q, 2 classes): Normal (0) vs Botnet Attack (1).
//!   - Tier 3 (Attack SubType, 1Q, 2 classes): Volumetric Flood (0) vs Recon/Control (1).
//! Total leaves = 4 x 2 x 2 = 16 leaves in Hilbert space.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

pub const N_IOT_FEATURES: usize = 10;
pub const N_DEVICE_CATEGORIES: usize = 4;
pub const N_ATTACK_FAMILIES: usize = 2;
pub const N_ATTACK_SUBTYPES: usize = 2;
pub const N_IOT_LEAVES: usize = N_DEVICE_CATEGORIES * N_ATTACK_FAMILIES * N_ATTACK_SUBTYPES; // 16

/// 3-Tier Cascaded Ultrametric Sorter Configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoTCascadedTreeConfig {
    pub n_device: usize,  // 2 qubits (4 macro-device categories)
    pub n_family: usize,  // 1 qubit  (2 attack families: Normal vs Botnet)
    pub n_subtype: usize, // 1 qubit  (2 attack subtypes: Volumetric vs Recon)
    pub n_env: usize,     // 6 qubits (entropy cooling sink)
}

impl Default for IoTCascadedTreeConfig {
    fn default() -> Self {
        Self {
            n_device: 2,
            n_family: 1,
            n_subtype: 1,
            n_env: 6,
        }
    }
}

/// Evaluates 3-Tier theoretical tree distance d_U(u, v).
pub fn iot_cascaded_tree_distance(
    du: usize,
    fu: usize,
    su: usize,
    dv: usize,
    fv: usize,
    sv: usize,
    g1: f64,
    g2: f64,
    g3: f64,
) -> f64 {
    if du != dv {
        g3 // Disjoint Macro-Device Categories
    } else if fu != fv {
        g2 // Sibling Attack Families on same Device
    } else if su != sv {
        g1 // SubType sibling within same Family
    } else {
        0.0 // Identical terminal leaf
    }
}

/// Comprehensive Report on IoT Botnet Cascaded Ultrametric Tree Routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoTCascadedTreeReport {
    pub timestamp_rfc3339: String,
    pub dataset: String,
    pub architecture: String,
    pub registers: String,
    pub n_qubits: usize,
    pub state_dim: usize,
    pub total_parameters: usize,
    pub total_leaves: usize,
    pub strong_triangle_inequality_holds: bool,
    pub triplet_violations: usize,
    pub evaluated_triplets: usize,
    pub train_samples: usize,
    pub test_samples: usize,
    pub training_time_sec: f64,
    pub inference_per_sample_us: f64,
    pub test_device_accuracy_pct: f64,
    pub test_family_accuracy_pct: f64,
    pub test_subtype_accuracy_pct: f64,
    pub test_joint_leaf_accuracy_pct: f64,
}

/// 3-Tier Hierarchical Label Tuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IoTTreeLabels {
    pub device_cat: usize,   // 0..4
    pub attack_family: usize, // 0..2 (0=Normal, 1=Botnet)
    pub attack_subtype: usize,// 0..2 (0=Volumetric/Steady, 1=Recon/Bursty)
    pub leaf_index: usize,    // 0..16
}

/// A parsed IoT network traffic sample with 10 normalized rotation angles.
#[derive(Debug, Clone)]
pub struct IoTFlowSample {
    pub device_name: String,
    pub attack_str: String,
    pub subtype_str: String,
    pub labels: IoTTreeLabels,
    pub features: [f64; N_IOT_FEATURES],
}

/// Maps raw device name string into 4 Macro-Device Categories:
///   0: Doorbells (Danmini, Ennio)
///   1: Thermostats & Monitors (Ecobee, Philips Baby Monitor)
///   2: Provision Enterprise Cameras (Provision PT-737E, PT-838)
///   3: Webcams & Consumer Cameras (Samsung Webcam, SimpleHome Cameras)
pub fn map_device_to_category(dev_str: &str) -> usize {
    let lower = dev_str.to_lowercase();
    if lower.contains("doorbell") {
        0
    } else if lower.contains("ecobee") || lower.contains("philips") || lower.contains("monitor") {
        1
    } else if lower.contains("provision") {
        2
    } else {
        3 // Samsung, SimpleHome, generic webcam
    }
}

/// Maps attack family string into 2 classes:
///   0: Normal / Benign
///   1: Botnet Attack (Gafgyt, Mirai)
pub fn map_attack_family(att_str: &str, label_val: usize) -> usize {
    let lower = att_str.to_lowercase();
    if lower.contains("normal") || lower.contains("benign") || label_val == 1 {
        0
    } else {
        1
    }
}

/// Maps attack subtype string into 2 classes:
///   0: Volumetric Floods (UDP, TCP, SYN, ACK, udpplain) or Normal Baseline
///   1: Recon & Exploit Vectors (Scan, Combo, Junk)
pub fn map_attack_subtype(sub_str: &str, family: usize) -> usize {
    if family == 0 {
        0 // Normal baseline
    } else {
        let lower = sub_str.to_lowercase();
        if lower.contains("scan") || lower.contains("combo") || lower.contains("junk") {
            1
        } else {
            0 // UDP, TCP, SYN, ACK floods
        }
    }
}

/// Computes the unique leaf index in 0..16 for a given (device, family, subtype) triplet.
#[inline(always)]
pub fn compute_iot_leaf_index(device_cat: usize, attack_family: usize, attack_subtype: usize) -> usize {
    (device_cat * 4) + (attack_family * 2) + attack_subtype
}

/// Locates the BoTNeTIoT stratified dataset or candidate paths.
pub fn locate_iot_dataset_file(candidates: &[&str]) -> Option<PathBuf> {
    for c in candidates {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    let global_fallback = PathBuf::from("C:/Users/zoddj/datasets/iot_ids.zip");
    if global_fallback.exists() {
        return Some(global_fallback);
    }
    None
}

/// Normalizes continuous scalar to [-pi/2, pi/2].
#[inline(always)]
fn norm_angle(val: f64, min: f64, max: f64) -> f64 {
    use std::f64::consts::{FRAC_PI_2, PI};
    if max <= min {
        return 0.0;
    }
    let clamped = val.clamp(min, max);
    let ratio = (clamped - min) / (max - min);
    (ratio * PI) - FRAC_PI_2
}

/// Streams and parses BoTNeTIoT-L01 stratified traffic flows up to max_samples.
pub fn parse_iot_traffic_flows(path: &Path, max_samples: Option<usize>) -> Vec<IoTFlowSample> {
    let file = File::open(path).expect("Open BoTNeTIoT CSV");
    let reader = BufReader::with_capacity(1024 * 1024, file);
    let mut flows = Vec::new();
    let mut lines = reader.lines();

    let header_line = match lines.next() {
        Some(Ok(h)) => h,
        _ => return flows,
    };

    let col_map: std::collections::HashMap<&str, usize> = header_line
        .split(',')
        .enumerate()
        .map(|(idx, name)| (name.trim(), idx))
        .collect();

    let idx_mi_w = *col_map.get("MI_dir_L0.1_weight").unwrap_or(&0);
    let idx_mi_m = *col_map.get("MI_dir_L0.1_mean").unwrap_or(&1);
    let idx_h_w = *col_map.get("H_L0.1_weight").unwrap_or(&3);
    let idx_hh_w = *col_map.get("HH_L0.1_weight").unwrap_or(&6);
    let idx_hh_mag = *col_map.get("HH_L0.1_magnitude").unwrap_or(&9);
    let idx_hh_jit = *col_map.get("HH_jit_L0.1_mean").unwrap_or(&14);
    let idx_hp_w = *col_map.get("HpHp_L0.1_weight").unwrap_or(&16);
    let idx_hp_mag = *col_map.get("HpHp_L0.1_magnitude").unwrap_or(&19);
    let idx_hp_rad = *col_map.get("HpHp_L0.1_radius").unwrap_or(&20);
    let idx_hp_pcc = *col_map.get("HpHp_L0.1_pcc").unwrap_or(&22);

    let idx_dev = *col_map.get("Device_Name").unwrap_or(&23);
    let idx_att = *col_map.get("Attack").unwrap_or(&24);
    let idx_sub = *col_map.get("Attack_subType").unwrap_or(&25);
    let idx_lbl = *col_map.get("label").unwrap_or(&26);

    for line_res in lines {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() <= idx_lbl {
            continue;
        }

        let dev_str = fields[idx_dev].trim();
        let att_str = fields[idx_att].trim();
        let sub_str = fields[idx_sub].trim();
        let label_val: usize = fields[idx_lbl].trim().parse().unwrap_or(0);

        let dev_cat = map_device_to_category(dev_str);
        let att_fam = map_attack_family(att_str, label_val);
        let att_sub = map_attack_subtype(sub_str, att_fam);
        let leaf_idx = compute_iot_leaf_index(dev_cat, att_fam, att_sub);

        let mi_w: f64 = fields[idx_mi_w].trim().parse().unwrap_or(1.0);
        let mi_m: f64 = fields[idx_mi_m].trim().parse().unwrap_or(60.0);
        let h_w: f64 = fields[idx_h_w].trim().parse().unwrap_or(1.0);
        let hh_w: f64 = fields[idx_hh_w].trim().parse().unwrap_or(1.0);
        let hh_mag: f64 = fields[idx_hh_mag].trim().parse().unwrap_or(60.0);
        let hh_jit: f64 = fields[idx_hh_jit].trim().parse().unwrap_or(0.0);
        let hp_w: f64 = fields[idx_hp_w].trim().parse().unwrap_or(1.0);
        let hp_mag: f64 = fields[idx_hp_mag].trim().parse().unwrap_or(60.0);
        let hp_rad: f64 = fields[idx_hp_rad].trim().parse().unwrap_or(0.0);
        let hp_pcc: f64 = fields[idx_hp_pcc].trim().parse().unwrap_or(0.0);

        let mut features = [0.0; N_IOT_FEATURES];
        features[0] = norm_angle(mi_w.ln_1p(), 0.0, 9.5);
        features[1] = norm_angle(mi_m, 50.0, 500.0);
        features[2] = norm_angle(h_w.ln_1p(), 0.0, 9.5);
        features[3] = norm_angle(hh_w.ln_1p(), 0.0, 9.0);
        features[4] = norm_angle(hh_mag, 50.0, 1000.0);
        features[5] = norm_angle((hh_jit + 1.0).ln(), 0.0, 22.0);
        features[6] = norm_angle(hp_w.ln_1p(), 0.0, 8.5);
        features[7] = norm_angle(hp_mag, 50.0, 1000.0);
        features[8] = norm_angle(hp_rad.ln_1p(), 0.0, 12.0);
        features[9] = norm_angle(hp_pcc, -1.0, 1.0);

        flows.push(IoTFlowSample {
            device_name: dev_str.to_string(),
            attack_str: att_str.to_string(),
            subtype_str: sub_str.to_string(),
            labels: IoTTreeLabels {
                device_cat: dev_cat,
                attack_family: att_fam,
                attack_subtype: att_sub,
                leaf_index: leaf_idx,
            },
            features,
        });

        if let Some(max) = max_samples {
            if flows.len() >= max {
                break;
            }
        }
    }

    flows
}

pub const N_IOT_QUBITS: usize = 10;
pub const IOT_STATE_DIM: usize = 1 << N_IOT_QUBITS; // 1,024

#[inline(always)]
fn apply_cz_gate(state: &mut [num_complex::Complex64], q1: usize, q2: usize) {
    let m1 = 1 << q1;
    let m2 = 1 << q2;
    for (idx, amp) in state.iter_mut().enumerate() {
        if (idx & m1 != 0) && (idx & m2 != 0) {
            *amp = -*amp;
        }
    }
}

/// Parameterized 3-Tier Cascaded QMUX Sorter Engine.
#[derive(Debug, Clone)]
pub struct IoTCascadedTreeSorter {
    pub cfg: IoTCascadedTreeConfig,
    pub trunk_params: Vec<f64>,
    pub family_qmux_params: Vec<[f64; 2]>,
    pub subtype_qmux_params: Vec<[f64; 2]>,
    pub cooling_phases: Vec<f64>,
}

impl IoTCascadedTreeSorter {
    pub fn new(cfg: IoTCascadedTreeConfig) -> Self {
        let trunk_params = vec![0.0; cfg.n_device * 2]; // 4 params for 2 device wires
        let family_qmux_params = vec![[0.0, 0.0]; 1 << cfg.n_device]; // 4 branches x 2
        let subtype_qmux_params = vec![[0.0, 0.0]; 1 << (cfg.n_device + cfg.n_family)]; // 8 branches x 2
        let cooling_phases = vec![0.0; cfg.n_env]; // 6 cooling phases
        Self {
            cfg,
            trunk_params,
            family_qmux_params,
            subtype_qmux_params,
            cooling_phases,
        }
    }

    pub fn init_seeded(&mut self, seed: u64) {
        use rand::Rng;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for p in self.trunk_params.iter_mut() {
            *p = rng.gen_range(-0.35..0.35);
        }
        for b in self.family_qmux_params.iter_mut() {
            b[0] = rng.gen_range(-0.35..0.35);
            b[1] = rng.gen_range(-0.35..0.35);
        }
        for b in self.subtype_qmux_params.iter_mut() {
            b[0] = rng.gen_range(-0.35..0.35);
            b[1] = rng.gen_range(-0.35..0.35);
        }
        for p in self.cooling_phases.iter_mut() {
            *p = rng.gen_range(0.05..0.20);
        }
    }

    pub fn total_parameters(&self) -> usize {
        self.trunk_params.len()
            + self.family_qmux_params.len() * 2
            + self.subtype_qmux_params.len() * 2
            + self.cooling_phases.len()
    }

    /// Evaluates forward circuit on input IoT traffic features.
    pub fn forward(&self, features: &[f64; N_IOT_FEATURES]) -> Vec<num_complex::Complex64> {
        use crate::ultrametric_tree::{apply_single_qubit_gate, gate_ry, gate_rz};
        use num_complex::Complex64;
        let mut state = vec![Complex64::default(); IOT_STATE_DIM];
        state[0] = Complex64::new(1.0, 0.0);

        // Stage 0: Feature Injection Ry(x_i)
        for q in 0..N_IOT_QUBITS {
            apply_single_qubit_gate(&mut state, q, &gate_ry(features[q]));
        }

        // Stage 1: Coarse Device Routing Trunk (Wires 0, 1)
        for q in 0..self.cfg.n_device {
            let ry = self.trunk_params[q * 2];
            let rz = self.trunk_params[q * 2 + 1];
            apply_single_qubit_gate(&mut state, q, &gate_ry(ry));
            apply_single_qubit_gate(&mut state, q, &gate_rz(rz));
        }
        apply_cz_gate(&mut state, 0, 1);

        // Stage 2: QMUX Attack Family Branching on Wire 2, conditioned on Wires 0, 1
        let wire_family = 2;
        let half_dim = IOT_STATE_DIM >> 1;
        let low_mask_f = (1 << wire_family) - 1;
        for i in 0..half_dim {
            let idx0 = ((i & !low_mask_f) << 1) | (i & low_mask_f);
            let idx1 = idx0 | (1 << wire_family);
            let dev_cat = idx0 & 0x03; // Wires 0, 1
            let [ry_p, rz_p] = self.family_qmux_params[dev_cat];
            let u_ry = gate_ry(ry_p);
            let u_rz = gate_rz(rz_p);
            let u = [
                [
                    u_rz[0][0] * u_ry[0][0] + u_rz[0][1] * u_ry[1][0],
                    u_rz[0][0] * u_ry[0][1] + u_rz[0][1] * u_ry[1][1],
                ],
                [
                    u_rz[1][0] * u_ry[0][0] + u_rz[1][1] * u_ry[1][0],
                    u_rz[1][0] * u_ry[0][1] + u_rz[1][1] * u_ry[1][1],
                ],
            ];
            let a0 = state[idx0];
            let a1 = state[idx1];
            state[idx0] = u[0][0] * a0 + u[0][1] * a1;
            state[idx1] = u[1][0] * a0 + u[1][1] * a1;
        }

        // Stage 3: QMUX Attack SubType Branching on Wire 3, conditioned on Wires 0, 1, 2
        let wire_subtype = 3;
        let low_mask_s = (1 << wire_subtype) - 1;
        for i in 0..half_dim {
            let idx0 = ((i & !low_mask_s) << 1) | (i & low_mask_s);
            let idx1 = idx0 | (1 << wire_subtype);
            let branch_ctx = idx0 & 0x07; // Wires 0, 1, 2
            let [ry_p, rz_p] = self.subtype_qmux_params[branch_ctx];
            let u_ry = gate_ry(ry_p);
            let u_rz = gate_rz(rz_p);
            let u = [
                [
                    u_rz[0][0] * u_ry[0][0] + u_rz[0][1] * u_ry[1][0],
                    u_rz[0][0] * u_ry[0][1] + u_rz[0][1] * u_ry[1][1],
                ],
                [
                    u_rz[1][0] * u_ry[0][0] + u_rz[1][1] * u_ry[1][0],
                    u_rz[1][0] * u_ry[0][1] + u_rz[1][1] * u_ry[1][1],
                ],
            ];
            let a0 = state[idx0];
            let a1 = state[idx1];
            state[idx0] = u[0][0] * a0 + u[0][1] * a1;
            state[idx1] = u[1][0] * a0 + u[1][1] * a1;
        }

        // Stage 4: Entropy Cooling Bridges into Wires 4..9
        for (e_idx, &phi) in self.cooling_phases.iter().enumerate() {
            let env_wire = 4 + e_idx;
            apply_cz_gate(&mut state, 3, env_wire);
            apply_single_qubit_gate(&mut state, env_wire, &gate_rz(phi));
        }

        state
    }

    /// Evaluates marginal distribution across all 16 hierarchical leaves.
    #[inline]
    pub fn compute_leaf_distribution(&self, state: &[num_complex::Complex64]) -> [f64; N_IOT_LEAVES] {
        let mut dist = [0.0; N_IOT_LEAVES];
        for (idx, amp) in state.iter().enumerate() {
            let leaf = idx & 0x0F;
            dist[leaf] += amp.norm_sqr();
        }
        dist
    }

    #[inline]
    pub fn marginal_device(&self, leaf_dist: &[f64; N_IOT_LEAVES]) -> [f64; N_DEVICE_CATEGORIES] {
        let mut dev = [0.0; N_DEVICE_CATEGORIES];
        for leaf in 0..N_IOT_LEAVES {
            let d = leaf >> 2;
            dev[d] += leaf_dist[leaf];
        }
        dev
    }

    #[inline]
    pub fn marginal_family(&self, leaf_dist: &[f64; N_IOT_LEAVES]) -> [f64; N_ATTACK_FAMILIES] {
        let mut fam = [0.0; N_ATTACK_FAMILIES];
        for leaf in 0..N_IOT_LEAVES {
            let f = (leaf >> 1) & 0x01;
            fam[f] += leaf_dist[leaf];
        }
        fam
    }

    #[inline]
    pub fn marginal_subtype(&self, leaf_dist: &[f64; N_IOT_LEAVES]) -> [f64; N_ATTACK_SUBTYPES] {
        let mut sub = [0.0; N_ATTACK_SUBTYPES];
        for leaf in 0..N_IOT_LEAVES {
            let s = leaf & 0x01;
            sub[s] += leaf_dist[leaf];
        }
        sub
    }

    /// Fast predictive evaluation: returns (device_cat, attack_family, attack_subtype, leaf_idx, confidence).
    pub fn predict(&self, features: &[f64; N_IOT_FEATURES]) -> (usize, usize, usize, usize, f64) {
        let state = self.forward(features);
        let dist = self.compute_leaf_distribution(&state);
        let p_dev = self.marginal_device(&dist);
        let p_fam = self.marginal_family(&dist);
        let p_sub = self.marginal_subtype(&dist);

        let pred_dev = p_dev
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let pred_fam = p_fam
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let pred_sub = p_sub
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let pred_leaf = compute_iot_leaf_index(pred_dev, pred_fam, pred_sub);
        let confidence = dist[pred_leaf];
        (pred_dev, pred_fam, pred_sub, pred_leaf, confidence)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_category_mapping() {
        assert_eq!(map_device_to_category("Danmini_Doorbell"), 0);
        assert_eq!(map_device_to_category("Ennio_Doorbell"), 0);
        assert_eq!(map_device_to_category("Ecobee_Thermostat"), 1);
        assert_eq!(map_device_to_category("Philips_B120N10_Baby_Monitor"), 1);
        assert_eq!(map_device_to_category("Provision_PT_737E_Security_Camera"), 2);
        assert_eq!(map_device_to_category("Provision_PT_838_Security_Camera"), 2);
        assert_eq!(map_device_to_category("Samsung_SNH_1011_N_Webcam"), 3);
        assert_eq!(map_device_to_category("SimpleHome_XCS7_1002_WHT_Security_Camera"), 3);
    }

    #[test]
    fn test_attack_family_and_subtype() {
        assert_eq!(map_attack_family("Normal", 1), 0);
        assert_eq!(map_attack_family("gafgyt", 0), 1);
        assert_eq!(map_attack_family("mirai", 0), 1);

        assert_eq!(map_attack_subtype("Normal", 0), 0);
        assert_eq!(map_attack_subtype("udp", 1), 0);
        assert_eq!(map_attack_subtype("tcp", 1), 0);
        assert_eq!(map_attack_subtype("scan", 1), 1);
        assert_eq!(map_attack_subtype("combo", 1), 1);
    }

    #[test]
    fn test_leaf_index_bounds() {
        for dev in 0..N_DEVICE_CATEGORIES {
            for fam in 0..N_ATTACK_FAMILIES {
                for sub in 0..N_ATTACK_SUBTYPES {
                    let leaf = compute_iot_leaf_index(dev, fam, sub);
                    assert!(leaf < N_IOT_LEAVES);
                }
            }
        }
    }

    #[test]
    fn test_parse_live_traffic_subset() {
        let candidate = Path::new("data/iot/botnet_iot_l01_stratified.csv");
        if candidate.exists() {
            let flows = parse_iot_traffic_flows(candidate, Some(50));
            assert_eq!(flows.len(), 50);
            for f in &flows {
                assert!(f.labels.device_cat < N_DEVICE_CATEGORIES);
                assert!(f.labels.attack_family < N_ATTACK_FAMILIES);
                assert!(f.labels.attack_subtype < N_ATTACK_SUBTYPES);
                assert!(f.labels.leaf_index < N_IOT_LEAVES);
            }
        }
    }
}
