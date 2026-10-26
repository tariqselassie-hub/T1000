// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Harmonized System (HS) Trade Taxonomy & Customs Ingestion Engine.
//!
//! Ingests international customs shipment declarations and maps them into an
//! empirical 3-tier taxonomy tree:
//! - Tier 1: Macro-Sector $S \in \{0, 1, 2, 3\}$ (Machinery, Textiles, Plastics, Precision)
//! - Tier 2: 2-Digit Chapter $C \in \{0, 1\}$ conditioned on Sector
//! - Tier 3: Compliance Status $F \in \{0, 1\}$ (0 = Legitimate, 1 = Fraudulent)
//!
//! Encodes logistics/financial trade shipment features into 10-dimensional
//! normalized rotation angles for multi-register Hilbert state injection.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const N_FEATURES: usize = 10;
pub const N_SECTORS: usize = 4;
pub const N_CHAPTERS_PER_SECTOR: usize = 2;
pub const N_LEAVES: usize = N_SECTORS * N_CHAPTERS_PER_SECTOR * 2; // 16 terminal leaves

/// Ingested customs shipment declaration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarmonizedShipment {
    pub declaration_id: u64,
    /// Macro-Sector S in 0..4
    pub sector_id: usize,
    /// 2-digit chapter branch C in 0..2
    pub chapter_id: usize,
    /// Raw 2-digit chapter code (e.g. 84, 85, 61, 62)
    pub chapter_code: u8,
    /// Compliance label: 0 = Legitimate, 1 = Fraudulent
    pub fraud: usize,
    /// Critical fraud label: 0 or 1
    pub critical_fraud: usize,
    /// 10 normalized feature angles in [-pi/2, pi/2]
    pub features: [f64; N_FEATURES],
}

impl HarmonizedShipment {
    /// Maps (sector_id, chapter_id, fraud) into a flat leaf index in 0..16.
    #[inline(always)]
    pub fn leaf_index(&self) -> usize {
        (self.sector_id * N_CHAPTERS_PER_SECTOR + self.chapter_id) * 2 + self.fraud
    }
}

/// Resolves the candidate paths for the Harmonized System CSV files.
pub fn locate_harmonized_file(filename_candidates: &[&str]) -> Option<PathBuf> {
    let candidate_dirs = [
        "data/harmonized",
        "../data/harmonized",
        "data",
        "../data",
        "C:/Users/zoddj/datasets",
    ];

    for dir in &candidate_dirs {
        for candidate in filename_candidates {
            let p1 = Path::new(dir).join(candidate);
            if p1.is_file() {
                return Some(p1);
            }
        }
    }
    None
}

/// Maps a 2-digit HS Chapter string into (Sector ID, Sub-chapter ID).
///
/// Supported Macro-Sectors:
/// - Sector 0 (XVI: Machinery & Electronics): Ch 84 (c=0), Ch 85 (c=1)
/// - Sector 1 (XI: Textiles & Apparel): Ch 61 (c=0), Ch 62 (c=1)
/// - Sector 2 (VII & VIII: Plastics & Leather): Ch 39 (c=0), Ch 42 (c=1)
/// - Sector 3 (XVIII & XX: Precision & Furniture): Ch 90 (c=0), Ch 94 (c=1)
#[inline]
pub fn map_chapter_to_sector(chapter_str: &str) -> Option<(usize, usize, u8)> {
    let chapter_code: u8 = chapter_str.parse().ok()?;
    match chapter_code {
        84 => Some((0, 0, 84)),
        85 => Some((0, 1, 85)),
        61 => Some((1, 0, 61)),
        62 => Some((1, 1, 62)),
        39 => Some((2, 0, 39)),
        42 => Some((2, 1, 42)),
        90 => Some((3, 0, 90)),
        94 => Some((3, 1, 94)),
        _ => None,
    }
}

/// Normalizes continuous trade scalar to [-pi/2, pi/2].
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

/// Parses the Harmonized trade customs dataset from CSV.
pub fn parse_harmonized_shipments(path: &Path, max_samples: Option<usize>) -> Vec<HarmonizedShipment> {
    let content = fs::read_to_string(path).expect("Read harmonized dataset CSV");
    let mut shipments = Vec::new();
    let mut lines = content.lines();

    let header = match lines.next() {
        Some(h) => h,
        None => return shipments,
    };

    let col_map: std::collections::HashMap<&str, usize> = header
        .split(',')
        .enumerate()
        .map(|(idx, name)| (name.trim(), idx))
        .collect();

    let idx_id = *col_map.get("Declaration ID").unwrap_or(&0);
    let idx_office = *col_map.get("Office ID").unwrap_or(&2);
    let idx_transport = *col_map.get("Mode of Transport").unwrap_or(&7);
    let idx_hs = *col_map.get("HS6 Code").unwrap_or(&12);
    let idx_dep = *col_map.get("Country of Departure").unwrap_or(&13);
    let idx_orig = *col_map.get("Country of Origin").unwrap_or(&14);
    let idx_tax = *col_map.get("Tax Rate").unwrap_or(&15);
    let idx_mass = *col_map.get("Net Mass").unwrap_or(&18);
    let idx_price = *col_map.get("Item Price").unwrap_or(&19);
    let idx_fraud = *col_map.get("Fraud").unwrap_or(&20);
    let idx_crit = *col_map.get("Critical Fraud").unwrap_or(&21);

    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() <= idx_crit {
            continue;
        }

        let hs_code = fields[idx_hs].trim();
        if hs_code.len() < 2 {
            continue;
        }
        let chapter_str = &hs_code[0..2];
        let (sector_id, chapter_id, chapter_code) = match map_chapter_to_sector(chapter_str) {
            Some(res) => res,
            None => continue,
        };

        let decl_id: u64 = fields[idx_id].trim().parse().unwrap_or(0);
        let office_id: f64 = fields[idx_office].trim().parse().unwrap_or(10.0);
        let mode_transport: f64 = fields[idx_transport].trim().parse().unwrap_or(10.0);
        let tax_rate: f64 = fields[idx_tax].trim().parse().unwrap_or(0.0);
        let net_mass = fields[idx_mass].trim().parse::<f64>().unwrap_or(1.0).max(0.01);
        let item_price = fields[idx_price].trim().parse::<f64>().unwrap_or(1.0).max(0.01);
        let fraud: usize = fields[idx_fraud].trim().parse().unwrap_or(0);
        let crit_fraud: usize = fields[idx_crit].trim().parse().unwrap_or(0);

        let dep_country = fields[idx_dep].trim();
        let orig_country = fields[idx_orig].trim();
        let direct_origin = if dep_country == orig_country { 1.0 } else { -1.0 };

        let log_price = item_price.ln();
        let log_mass = net_mass.ln();
        let log_unit_val = (item_price / net_mass).ln();

        let mut features = [0.0; N_FEATURES];
        features[0] = norm_angle(log_price, 3.0, 18.0);
        features[1] = norm_angle(log_mass, -2.0, 12.0);
        features[2] = norm_angle(log_unit_val, -2.0, 10.0);
        features[3] = norm_angle(tax_rate, 0.0, 30.0);
        features[4] = if mode_transport >= 30.0 { 0.785 } else { -0.785 }; // Air vs Sea/Ground
        features[5] = direct_origin * std::f64::consts::FRAC_PI_2;
        features[6] = norm_angle(office_id, 10.0, 50.0);
        features[7] = norm_angle((log_price - log_mass).abs(), 0.0, 8.0);
        features[8] = norm_angle(log_price * (tax_rate + 1.0).ln(), 0.0, 50.0);
        features[9] = norm_angle((office_id + mode_transport) * 0.5, 10.0, 45.0);

        shipments.push(HarmonizedShipment {
            declaration_id: decl_id,
            sector_id,
            chapter_id,
            chapter_code,
            fraud,
            critical_fraud: crit_fraud,
            features,
        });

        if let Some(max) = max_samples {
            if shipments.len() >= max {
                break;
            }
        }
    }

    shipments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chapter_to_sector_mapping() {
        assert_eq!(map_chapter_to_sector("84"), Some((0, 0, 84)));
        assert_eq!(map_chapter_to_sector("85"), Some((0, 1, 85)));
        assert_eq!(map_chapter_to_sector("61"), Some((1, 0, 61)));
        assert_eq!(map_chapter_to_sector("62"), Some((1, 1, 62)));
        assert_eq!(map_chapter_to_sector("39"), Some((2, 0, 39)));
        assert_eq!(map_chapter_to_sector("42"), Some((2, 1, 42)));
        assert_eq!(map_chapter_to_sector("90"), Some((3, 0, 90)));
        assert_eq!(map_chapter_to_sector("94"), Some((3, 1, 94)));
        assert_eq!(map_chapter_to_sector("01"), None);
    }

    #[test]
    fn test_leaf_index_bounds() {
        let shipment = HarmonizedShipment {
            declaration_id: 12345,
            sector_id: 3,
            chapter_id: 1,
            chapter_code: 94,
            fraud: 1,
            critical_fraud: 1,
            features: [0.0; N_FEATURES],
        };
        // (3 * 2 + 1) * 2 + 1 = 7 * 2 + 1 = 15
        assert_eq!(shipment.leaf_index(), 15);
        assert!(shipment.leaf_index() < N_LEAVES);
    }

    #[test]
    fn test_parse_live_shipments_subset() {
        if let Some(path) = locate_harmonized_file(&["harmonized-2nd-set.csv"]) {
            let shipments = parse_harmonized_shipments(&path, Some(50));
            assert_eq!(shipments.len(), 50);
            for s in &shipments {
                assert!(s.sector_id < N_SECTORS);
                assert!(s.chapter_id < N_CHAPTERS_PER_SECTOR);
                assert!(s.fraud <= 1);
                for &f in &s.features {
                    assert!(f.is_finite());
                }
            }
        }
    }
}
