// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Bank Fraud Anomaly Detection & Financial Transaction Ingestion Engine.
//!
//! Streams real-world financial transaction records from `bank_fraud.csv` (1,000,000 samples)
//! and extracts 10 normalized quantum feature angles for Hilbert statevector injection:
//! - Transaction magnitude & balance ratio
//! - Transit velocity (distance / elapsed time)
//! - Security alarms (failed attempts, recent PIN resets, international hops)
//! - Temporal circular phase & credit risk profile

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

pub const N_FRAUD_FEATURES: usize = 10;
pub const N_FRAUD_TYPES: usize = 7;

/// Parsed financial transaction record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BankTransaction {
    pub transaction_id: String,
    pub is_fraud: usize,     // 0 = Clean, 1 = Fraud
    pub fraud_type_id: usize, // 0 = None, 1..6 = Specific fraud pattern
    pub transaction_amount: f64,
    pub account_balance: f64,
    pub features: [f64; N_FRAUD_FEATURES],
}

/// Locates candidate paths for the bank fraud dataset.
pub fn locate_bank_fraud_file(filename_candidates: &[&str]) -> Option<PathBuf> {
    let candidate_dirs = [
        "data/fraud",
        "../data/fraud",
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

/// Maps string fraud_type into a canonical index in 0..7.
#[inline]
pub fn map_fraud_type(ftype_str: &str) -> usize {
    match ftype_str.trim() {
        "Friendly Fraud" => 1,
        "Card Cloning" => 2,
        "Phishing" => 3,
        "Synthetic Identity" => 4,
        "Identity Theft" => 5,
        "Account Takeover" => 6,
        _ => 0, // "None" or unclassified
    }
}

/// Streams and parses bank fraud transactions up to max_samples.
pub fn parse_bank_transactions(path: &Path, max_samples: Option<usize>) -> Vec<BankTransaction> {
    let file = File::open(path).expect("Open bank fraud CSV");
    let reader = BufReader::with_capacity(1024 * 1024, file);
    let mut transactions = Vec::new();
    let mut lines = reader.lines();

    let header_line = match lines.next() {
        Some(Ok(h)) => h,
        _ => return transactions,
    };

    let col_map: std::collections::HashMap<&str, usize> = header_line
        .split(',')
        .enumerate()
        .map(|(idx, name)| (name.trim(), idx))
        .collect();

    let idx_hour = *col_map.get("hour_of_day").unwrap_or(&4);
    let idx_night = *col_map.get("is_night_transaction").unwrap_or(&6);
    let idx_score = *col_map.get("credit_score").unwrap_or(&13);
    let idx_bal = *col_map.get("account_balance").unwrap_or(&15);
    let idx_amt = *col_map.get("transaction_amount").unwrap_or(&16);
    let idx_dist = *col_map.get("distance_from_home_km").unwrap_or(&19);
    let idx_time = *col_map.get("time_since_last_txn_hrs").unwrap_or(&20);
    let idx_intl = *col_map.get("is_international").unwrap_or(&21);
    let idx_fail = *col_map.get("failed_attempts").unwrap_or(&22);
    let idx_pin = *col_map.get("pin_changed_recently").unwrap_or(&23);
    let idx_fraud = *col_map.get("is_fraud").unwrap_or(&24);
    let idx_type = *col_map.get("fraud_type").unwrap_or(&25);

    for line_res in lines {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() <= idx_type {
            continue;
        }

        let _hour: f64 = fields[idx_hour].trim().parse().unwrap_or(12.0);
        let is_night: f64 = fields[idx_night].trim().parse().unwrap_or(0.0);
        let credit_score: f64 = fields[idx_score].trim().parse().unwrap_or(650.0);
        let balance = fields[idx_bal].trim().parse::<f64>().unwrap_or(100.0).max(0.0);
        let amount = fields[idx_amt].trim().parse::<f64>().unwrap_or(10.0).max(0.01);
        let distance = fields[idx_dist].trim().parse::<f64>().unwrap_or(1.0).max(0.01);
        let elapsed_hrs = fields[idx_time].trim().parse::<f64>().unwrap_or(1.0).max(0.01);
        let is_intl: f64 = fields[idx_intl].trim().parse().unwrap_or(0.0);
        let failed_att: f64 = fields[idx_fail].trim().parse().unwrap_or(0.0);
        let pin_changed: f64 = fields[idx_pin].trim().parse().unwrap_or(0.0);
        let is_fraud: usize = fields[idx_fraud].trim().parse().unwrap_or(0);
        let fraud_type_str = fields[idx_type].trim();
        let fraud_type_id = map_fraud_type(fraud_type_str);

        // Feature Engineering
        let log_amount = amount.ln();
        let bal_ratio = (amount / (balance + 10.0)).min(10.0);
        let log_dist = distance.ln();
        let velocity_kmh = (distance / elapsed_hrs).min(1000.0);

        use std::f64::consts::FRAC_PI_2;
        let mut features = [0.0; N_FRAUD_FEATURES];
        features[0] = (failed_att / 5.0).clamp(0.0, 1.0) * FRAC_PI_2;
        features[1] = (if is_intl > 0.5 { 1.0 } else { 0.0 }) * FRAC_PI_2;
        features[2] = (if is_night > 0.5 || pin_changed > 0.5 { 1.0 } else { 0.0 }) * (FRAC_PI_2 * 0.7);
        features[3] = (velocity_kmh / 250.0).clamp(0.0, 1.0) * (FRAC_PI_2 * 0.7);
        features[4] = (bal_ratio / 3.0).clamp(0.0, 1.0) * (FRAC_PI_2 * 0.6);
        features[5] = (log_amount * (failed_att + 1.0) / 25.0).clamp(0.0, 1.0) * (FRAC_PI_2 * 0.6);
        features[6] = (log_dist / 8.0).clamp(0.0, 1.0) * (FRAC_PI_2 * 0.5);
        features[7] = ((850.0 - credit_score) / 550.0).clamp(0.0, 1.0) * (FRAC_PI_2 * 0.5);
        features[8] = (log_amount / 9.0).clamp(0.0, 1.0) * (FRAC_PI_2 * 0.4);
        features[9] = (if is_night > 0.5 { 1.0 } else { 0.0 }) * (FRAC_PI_2 * 0.4);

        transactions.push(BankTransaction {
            transaction_id: fields[0].to_string(),
            is_fraud,
            fraud_type_id,
            transaction_amount: amount,
            account_balance: balance,
            features,
        });

        if let Some(max) = max_samples {
            if transactions.len() >= max {
                break;
            }
        }
    }

    transactions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fraud_type_mapping() {
        assert_eq!(map_fraud_type("None"), 0);
        assert_eq!(map_fraud_type("Friendly Fraud"), 1);
        assert_eq!(map_fraud_type("Card Cloning"), 2);
        assert_eq!(map_fraud_type("Phishing"), 3);
        assert_eq!(map_fraud_type("Synthetic Identity"), 4);
        assert_eq!(map_fraud_type("Identity Theft"), 5);
        assert_eq!(map_fraud_type("Account Takeover"), 6);
    }

    #[test]
    fn test_parse_live_transactions_subset() {
        if let Some(path) = locate_bank_fraud_file(&["bank_fraud.csv"]) {
            let txns = parse_bank_transactions(&path, Some(100));
            assert_eq!(txns.len(), 100);
            for t in &txns {
                assert!(t.is_fraud <= 1);
                assert!(t.fraud_type_id < N_FRAUD_TYPES);
                for &f in &t.features {
                    assert!(f.is_finite());
                }
            }
        }
    }
}
