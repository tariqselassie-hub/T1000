// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use crate::ansatz::DeepQuantumNeuralNetwork;
use crate::memory::QuantumAssociativeMemory;
use crate::povm_readout::POVMClassifier;
use crate::state::{QuantumFeatureMap, QuantumState};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Corpus document descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestedDoc {
    pub path: String,
    pub title: String,
    pub category: String,
    pub token_count: usize,
    pub entropy: f64,
    pub povm_top_class: String,
    pub povm_top_prob: f64,
}

/// Ingestion result summarizing quantum corpus encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionTelemetry {
    pub total_documents: usize,
    pub total_tokens: usize,
    pub categories: Vec<String>,
    pub documents: Vec<IngestedDoc>,
    pub mean_entropy: f64,
    pub cross_category_interference: Vec<(String, String, f64)>,
}

/// The Corpus Ingestor engine.
#[derive(Debug)]
pub struct CorpusIngestor {
    pub feature_map: QuantumFeatureMap,
    pub qnn: DeepQuantumNeuralNetwork,
    pub povm: POVMClassifier,
    pub memory: QuantumAssociativeMemory,
}

impl CorpusIngestor {
    pub fn new(dim: usize, qnn_depth: usize) -> Self {
        let categories = vec![
            "bridges".to_string(),
            "deen".to_string(),
            "percival".to_string(),
            "python".to_string(),
            "rust".to_string(),
        ];

        Self {
            feature_map: QuantumFeatureMap::new(dim, 1337),
            qnn: DeepQuantumNeuralNetwork::new(dim, qnn_depth, 0.25, 0.15),
            povm: POVMClassifier::new(dim, categories),
            memory: QuantumAssociativeMemory::new(dim),
        }
    }

    /// Recursively scan and ingest all corpus files in the given root path.
    pub fn ingest_corpus_dir(&mut self, corpus_dir: &Path) -> Result<IngestionTelemetry, String> {
        let files = collect_corpus_files(corpus_dir)?;
        if files.is_empty() {
            return Err(format!("No corpus files found under {:?}", corpus_dir));
        }

        struct IntermediateDoc {
            path: PathBuf,
            title: String,
            category: &'static str,
            token_count: usize,
            entropy: f64,
            evolved_state: QuantumState,
        }

        let mut intermediate_docs = Vec::new();
        let mut total_tokens = 0;
        let mut total_entropy = 0.0;

        // --- PASS 1: Quantum Trajectory Encoding & Evolved State Generation ---
        for path in &files {
            let content = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            let category = detect_category(path);
            let title = extract_title(&content, path);
            let tokens = tokenize_text(&content);
            let token_count = tokens.len();
            total_tokens += token_count;

            let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();

            // Quantum State Encoding via FHRR feature map trajectory
            let raw_state = self.feature_map.encode_sequence(&token_refs);

            // Variational Quantum Layer Evolution
            let evolved_state = self.qnn.forward(&raw_state);

            // Compute Shannon Entropy in Hilbert space
            let reg = evolved_state.to_register(42);
            let entropy = reg.entropy();
            total_entropy += entropy;

            intermediate_docs.push(IntermediateDoc {
                path: path.clone(),
                title,
                category,
                token_count,
                entropy,
                evolved_state,
            });
        }

        // --- CALIBRATION: Build and Set Empirical Category Subspace Centroids ---
        let mut category_accumulators: std::collections::HashMap<&'static str, QuantumState> =
            std::collections::HashMap::new();

        for doc in &intermediate_docs {
            let acc = category_accumulators
                .entry(doc.category)
                .or_insert_with(|| QuantumState::zero(self.feature_map.dim));
            acc.superpose(&doc.evolved_state, 1.0);
        }

        for (cat_name, centroid) in &category_accumulators {
            if let Some(cat_idx) = self
                .povm
                .class_labels
                .iter()
                .position(|c| c.as_str() == *cat_name)
            {
                self.povm.set_class_centroid(cat_idx, centroid);
            }
        }

        // --- PASS 2: Calibrated POVM Readout & Associative Memory Insertion ---
        let mut ingested_docs = Vec::new();

        for doc in intermediate_docs {
            let class_probs = self.povm.predict_probabilities(&doc.evolved_state);
            let (top_class, top_prob) = class_probs
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(("unknown".to_string(), 0.0));

            let doc_id = doc
                .path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "doc".to_string());

            self.memory.insert(
                doc_id.clone(),
                doc.title.clone(),
                doc.category.to_string(),
                doc.token_count,
                doc.evolved_state,
            );

            ingested_docs.push(IngestedDoc {
                path: doc.path.display().to_string(),
                title: doc.title,
                category: doc.category.to_string(),
                token_count: doc.token_count,
                entropy: doc.entropy,
                povm_top_class: top_class,
                povm_top_prob: top_prob,
            });
        }

        // 6. Compute Cross-Category Quantum Interference
        let interference_pairs = self.compute_category_interference();

        let mean_entropy = if !ingested_docs.is_empty() {
            total_entropy / ingested_docs.len() as f64
        } else {
            0.0
        };

        Ok(IngestionTelemetry {
            total_documents: ingested_docs.len(),
            total_tokens,
            categories: self.povm.class_labels.clone(),
            documents: ingested_docs,
            mean_entropy,
            cross_category_interference: interference_pairs,
        })
    }

    /// Query the ingested corpus with a natural language or axiom prompt.
    pub fn query_associative_recall(
        &self,
        query_text: &str,
        top_k: usize,
    ) -> Vec<(&crate::memory::MemoryEntry, f64)> {
        let tokens = tokenize_text(query_text);
        let token_refs: Vec<&str> = tokens.iter().map(|s| s.as_str()).collect();
        let raw_query = self.feature_map.encode_sequence(&token_refs);
        let evolved_query = self.qnn.forward(&raw_query);

        self.memory.recall_fidelity(&evolved_query, top_k)
    }

    fn compute_category_interference(&self) -> Vec<(String, String, f64)> {
        let mut results = Vec::new();
        // Compute interference between representative states of distinct categories
        for i in 0..self.memory.entries.len() {
            for j in (i + 1)..self.memory.entries.len() {
                let e1 = &self.memory.entries[i];
                let e2 = &self.memory.entries[j];
                if e1.category != e2.category {
                    let score =
                        POVMClassifier::net_interference(&e1.quantum_state, &e2.quantum_state);
                    results.push((e1.id.clone(), e2.id.clone(), score));
                }
            }
        }
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(10);
        results
    }
}

fn collect_corpus_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.is_dir() {
                let mut sub = collect_corpus_files(&path)?;
                files.append(&mut sub);
            } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if matches!(ext, "md" | "percival" | "rs" | "py" | "txt") {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    Ok(files)
}

fn detect_category(path: &Path) -> &'static str {
    let s = path.to_string_lossy().to_lowercase();
    if s.contains("bridges") {
        "bridges"
    } else if s.contains("deen") {
        "deen"
    } else if s.contains("percival") {
        "percival"
    } else if s.contains("python") {
        "python"
    } else if s.contains("rust") {
        "rust"
    } else {
        "general"
    }
}

fn extract_title(content: &str, path: &Path) -> String {
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("# ") {
            return trimmed.trim_start_matches("# ").trim().to_string();
        }
    }
    path.file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "Untitled".to_string())
}

pub fn tokenize_text(text: &str) -> Vec<String> {
    const STOPWORDS: &[&str] = &[
        "the", "and", "in", "of", "to", "is", "for", "with", "as", "by", "on", "at", "it", "this",
        "that", "from", "an", "be", "or", "are", "we", "can", "our", "all", "which", "each",
        "into", "has", "have", "more", "also", "its", "then", "such", "when", "where", "how",
        "what", "their", "they", "will", "would",
    ];

    text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|s| !s.is_empty() && s.len() > 1)
        .map(|s| s.to_lowercase())
        .filter(|s| !STOPWORDS.contains(&s.as_str()))
        .collect()
}
