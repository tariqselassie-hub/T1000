// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! ECLEKTIK Multilingual Taxonomy & Dataset Ingestion.
//!
//! Hierarchical Tree Structure:
//! H = H_C (2Q, 4 Macro-Families) \otimes H_F (2Q, 4 Fine Leaves) \otimes H_E (6Q, Entropy Sink) = 10 Qubits.
//!
//! Family 0 (Germanic):                 en (s=0), de (s=1)
//! Family 1 (Romance):                  es (s=0), fr (s=1), it (s=2), pt (s=3)
//! Family 2 (East Asian):               zh (s=0), ja (s=1), ko (s=2)
//! Family 3 (Afroasiatic/Indic/Austro): he (s=0), hi (s=1), id (s=2)

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

pub const N_LANGUAGES: usize = 12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageMeta {
    pub code: &'static str,
    pub name: &'static str,
    pub family: &'static str,
    pub y: usize, // Coarse register index in 0..4
    pub s: usize, // Fine register index in 0..4
    pub leaf_id: usize, // Global leaf ID in 0..12
}

pub const LANGUAGES: [LanguageMeta; N_LANGUAGES] = [
    LanguageMeta { code: "en", name: "English",    family: "Germanic", y: 0, s: 0, leaf_id: 0 },
    LanguageMeta { code: "de", name: "German",     family: "Germanic", y: 0, s: 1, leaf_id: 1 },
    LanguageMeta { code: "es", name: "Spanish",    family: "Romance",  y: 1, s: 0, leaf_id: 2 },
    LanguageMeta { code: "fr", name: "French",     family: "Romance",  y: 1, s: 1, leaf_id: 3 },
    LanguageMeta { code: "it", name: "Italian",    family: "Romance",  y: 1, s: 2, leaf_id: 4 },
    LanguageMeta { code: "pt", name: "Portuguese", family: "Romance",  y: 1, s: 3, leaf_id: 5 },
    LanguageMeta { code: "zh", name: "Chinese",    family: "EastAsian",y: 2, s: 0, leaf_id: 6 },
    LanguageMeta { code: "ja", name: "Japanese",   family: "EastAsian",y: 2, s: 1, leaf_id: 7 },
    LanguageMeta { code: "ko", name: "Korean",     family: "EastAsian",y: 2, s: 2, leaf_id: 8 },
    LanguageMeta { code: "he", name: "Hebrew",     family: "AfroIndic",y: 3, s: 0, leaf_id: 9 },
    LanguageMeta { code: "hi", name: "Hindi",      family: "AfroIndic",y: 3, s: 1, leaf_id: 10 },
    LanguageMeta { code: "id", name: "Indonesian", family: "AfroIndic",y: 3, s: 2, leaf_id: 11 },
];

#[allow(dead_code)]
pub fn leaf_to_language(y: usize, s: usize) -> Option<usize> {
    for lang in &LANGUAGES {
        if lang.y == y && lang.s == s {
            return Some(lang.leaf_id);
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct EclektikSample {
    pub text: String,
    pub meta: LanguageMeta,
}

pub fn locate_eclektik_file() -> PathBuf {
    let candidates = [
        "data/eclektik/eclektic_main.jsonl",
        "../data/eclektik/eclektic_main.jsonl",
        "../../data/eclektik/eclektic_main.jsonl",
    ];
    for c in &candidates {
        let p = Path::new(c);
        if p.is_file() {
            return p.to_path_buf();
        }
    }
    panic!("Could not locate eclektic_main.jsonl in workspace data paths");
}

pub fn load_eclektik_samples(path: &Path, max_rows: Option<usize>) -> Vec<EclektikSample> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Failed to open {}: {}", path.display(), e));
    let reader = BufReader::new(file);
    let mut samples = Vec::new();

    for (row_idx, line) in reader.lines().enumerate() {
        if let Some(max) = max_rows {
            if row_idx >= max {
                break;
            }
        }
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        if line.trim().is_empty() {
            continue;
        }

        let val: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        for meta in &LANGUAGES {
            let q_key = format!("{}_q", meta.code);
            let a_key = format!("{}_a", meta.code);
            let q_text = val.get(&q_key).and_then(|v| v.as_str()).unwrap_or("");
            let a_text = val.get(&a_key).and_then(|v| v.as_str()).unwrap_or("");

            if !q_text.is_empty() {
                let full_text = if !a_text.is_empty() {
                    format!("{} {}", q_text, a_text)
                } else {
                    q_text.to_string()
                };

                samples.push(EclektikSample {
                    text: full_text,
                    meta: *meta,
                });
            }
        }
    }

    samples
}
