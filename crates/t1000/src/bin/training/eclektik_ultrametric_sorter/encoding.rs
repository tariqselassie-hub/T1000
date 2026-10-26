// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Multilingual N-Gram & Script Hilbert Space Statevector Projector.

use num_complex::Complex64;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[allow(dead_code)]
pub const STATE_DIM: usize = 1024; // 10 Qubits = 2^10

#[allow(dead_code)]
pub fn encode_multilingual_text(text: &str) -> Vec<Complex64> {
    let mut state = vec![Complex64::default(); STATE_DIM];
    let bytes = text.as_bytes();
    let chars: Vec<char> = text.chars().collect();

    if bytes.is_empty() {
        let uniform = Complex64::new(1.0 / (STATE_DIM as f64).sqrt(), 0.0);
        return vec![uniform; STATE_DIM];
    }

    // 1. Amplitudes 0..256: Normalized byte spectrum
    let mut byte_counts = [0.0f64; 256];
    for &b in bytes {
        byte_counts[b as usize] += 1.0;
    }
    let total_bytes = bytes.len() as f64;
    for (i, &count) in byte_counts.iter().enumerate() {
        let frac = count / total_bytes;
        state[i] = Complex64::new(frac, 0.0);
    }

    // 2. Amplitudes 256..768 (512 slots): Character Bigram Hashes
    let mut bigram_counts = [0.0f64; 512];
    if chars.len() >= 2 {
        for window in chars.windows(2) {
            let mut hasher = DefaultHasher::new();
            window[0].hash(&mut hasher);
            window[1].hash(&mut hasher);
            let slot = (hasher.finish() % 512) as usize;
            bigram_counts[slot] += 1.0;
        }
        let total_bigrams = (chars.len() - 1) as f64;
        for (slot, &count) in bigram_counts.iter().enumerate() {
            let frac = count / total_bigrams;
            let phase = (slot as f64 * std::f64::consts::PI) / 256.0;
            state[256 + slot] = Complex64::from_polar(frac, phase);
        }
    }

    // 3. Amplitudes 768..1024 (256 slots): Unicode Script & Morphological Buckets
    let mut script_counts = [0.0f64; 256];
    for &ch in &chars {
        let u = ch as u32;
        let bucket = match u {
            0x0000..=0x007F => (u % 32) as usize, // ASCII Latin
            0x0080..=0x024F => 32 + ((u - 0x0080) % 32) as usize, // Extended Latin / Diacritics
            0x0590..=0x05FF => 64 + ((u - 0x0590) % 32) as usize, // Hebrew
            0x0900..=0x097F => 96 + ((u - 0x0900) % 32) as usize, // Devanagari (Hindi)
            0x3040..=0x309F => 128 + ((u - 0x3040) % 16) as usize, // Hiragana (Japanese)
            0x30A0..=0x30FF => 144 + ((u - 0x30A0) % 16) as usize, // Katakana (Japanese)
            0x4E00..=0x9FFF => 160 + ((u - 0x4E00) % 48) as usize, // CJK Ideographs (Chinese/Kanji)
            0xAC00..=0xD7AF => 208 + ((u - 0xAC00) % 32) as usize, // Hangul (Korean)
            _ => 240 + (u % 16) as usize,         // Other symbols & punctuation
        };
        script_counts[bucket.min(255)] += 1.0;
    }
    let total_chars = chars.len() as f64;
    for (b, &count) in script_counts.iter().enumerate() {
        let frac = count / total_chars;
        let phase = (b as f64 * std::f64::consts::PI * 2.0) / 256.0;
        state[768 + b] = Complex64::from_polar(frac, phase);
    }

    // Normalization to strictly ||ψ|| = 1.0
    let mut norm_sq = 0.0;
    for amp in &state {
        norm_sq += amp.norm_sqr();
    }
    if norm_sq > 1e-12 {
        let inv_norm = 1.0 / norm_sq.sqrt();
        for amp in &mut state {
            *amp *= inv_norm;
        }
    } else {
        let uniform = Complex64::new(1.0 / (STATE_DIM as f64).sqrt(), 0.0);
        state.fill(uniform);
    }

    state
}

/// Extracts hierarchical 10-qubit variational rotation angles:
/// Wires 0..1: Coarse Register H_C (4 Macro-Families)
/// Wires 2..3: Fine Register H_F (4 Fine Sub-Languages)
/// Wires 4..9: Environment Register H_E (Entropy Sink)
pub fn extract_multilingual_angles(text: &str) -> [f64; 10] {
    let mut angles = [0.0f64; 10];
    let chars: Vec<char> = text.chars().collect();
    let lower_text = text.to_lowercase();
    let words: Vec<&str> = lower_text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .collect();

    let mut n_cjk = 0usize;
    let mut n_kana = 0usize;
    let mut n_hangul = 0usize;
    let mut n_hebrew = 0usize;
    let mut n_deva = 0usize;

    for &ch in &chars {
        let u = ch as u32;
        match u {
            0x4E00..=0x9FFF => n_cjk += 1,
            0x3040..=0x30FF => n_kana += 1,
            0xAC00..=0xD7AF => n_hangul += 1,
            0x0590..=0x05FF => n_hebrew += 1,
            0x0900..=0x097F => n_deva += 1,
            _ => {}
        }
    }

    let id_kw = [
        "yang",
        "pada",
        "di",
        "dan",
        "dengan",
        "dari",
        "ke",
        "untuk",
        "adalah",
        "siapa",
        "tahun",
        "oleh",
        "apakah",
        "bagaimana",
        "dimana",
    ];
    let id_hits = words.iter().filter(|w| id_kw.contains(w)).count();
    let is_id = id_hits >= 1;

    let asian_afro = (n_cjk + n_kana + n_hangul + n_hebrew + n_deva) > 0 || is_id;
    let pi = std::f64::consts::PI;

    // Wire 1: Coarse bit 1 (0: European Latin, 1: Asian/Afro)
    angles[1] = if asian_afro { 0.95 * pi } else { 0.05 * pi };

    // Wire 0: Coarse bit 0
    if !asian_afro {
        let rom_kw = [
            "de", "la", "el", "en", "es", "que", "los", "las", "del", "le", "les", "des", "est",
            "per", "di", "il", "con", "un", "una", "por", "dans", "chi", "cosa", "quando", "dove",
            "quale", "quali", "della", "delle", "dei", "nel", "nella", "ha", "che", "o", "os",
            "do", "da", "dos", "das", "não", "foi", "quem", "no", "na", "quién", "qué", "cuál",
            "al", "se", "si", "su", "era",
        ];
        let ger_kw = [
            "the", "who", "was", "in", "of", "and", "what", "to", "is", "that", "with", "for",
            "der", "die", "das", "wer", "wurde", "und", "von", "zu", "den", "mit", "wie", "wo",
            "wann", "warum", "welche", "welcher", "welches", "durch", "nach", "bei",
        ];
        let r_hits = words.iter().filter(|w| rom_kw.contains(w)).count();
        let g_hits = words.iter().filter(|w| ger_kw.contains(w)).count();
        angles[0] = if r_hits > g_hits {
            0.95 * pi
        } else {
            0.05 * pi
        };
    } else {
        let is_afro = (n_hebrew + n_deva > 0) || is_id;
        angles[0] = if is_afro { 0.95 * pi } else { 0.05 * pi };
    }

    // Wires 2 & 3: Fine Sub-Language router
    if !asian_afro {
        // European Latin: Germanic vs Romance
        if angles[0] < 0.5 * pi {
            // Germanic: en (s=0: bit0=0, bit1=0), de (s=1: bit0=1, bit1=0)
            let de_kw = [
                "der", "die", "das", "wer", "wurde", "und", "von", "zu", "den", "mit", "ist", "im",
                "dem", "nicht", "nach", "eine", "ein", "als", "auch",
            ];
            let has_umlaut = chars.iter().any(|&c| "äöüßÄÖÜ".contains(c));
            let de_hits =
                words.iter().filter(|w| de_kw.contains(w)).count() + if has_umlaut { 3 } else { 0 };
            let en_hits = words
                .iter()
                .filter(|w| {
                    ["the", "who", "was", "in", "of", "and", "what", "to", "is"].contains(w)
                })
                .count();
            if de_hits > en_hits {
                angles[2] = 0.95 * pi; // bit 0 = 1
                angles[3] = 0.05 * pi; // bit 1 = 0
            } else {
                angles[2] = 0.05 * pi; // bit 0 = 0
                angles[3] = 0.05 * pi; // bit 1 = 0
            }
        } else {
            // Romance: es (s=0: 00), fr (s=1: 10), it (s=2: 01), pt (s=3: 11)
            let fr_words = [
                "le", "la", "les", "des", "est", "du", "une", "qui", "dans", "pour", "ce", "ont",
                "sur", "au", "aux", "avec", "été", "quel", "quelle", "quels", "quelles", "quand",
                "comment",
            ];
            let pt_words = [
                "o", "os", "do", "dos", "das", "em", "para", "com", "não", "foi", "quem", "no",
                "na", "nos", "nas", "pelo", "pela", "mais", "seus", "suas", "quais",
            ];
            let it_words = [
                "il", "di", "del", "della", "dei", "delle", "chi", "nel", "nella", "per", "ha",
                "che", "sono", "ed", "anche", "dopo", "stato", "stata", "gli", "degli", "questa",
                "questo", "quale", "quali", "cosa", "quando", "dove", "come", "era", "durante",
            ];
            let es_words = [
                "el", "los", "las", "del", "quién", "por", "fue", "más", "pero", "al", "es", "son",
                "su", "sus", "como", "entre", "sobre", "este", "esta", "en", "con", "qué", "cuál",
                "cuáles", "cuándo", "dónde",
            ];

            let mut fr_score = words.iter().filter(|w| fr_words.contains(w)).count();
            fr_score += chars.iter().filter(|&c| "éèêëàçùôû".contains(*c)).count() * 2;
            if words.contains(&"qui") {
                fr_score += 4;
            }

            let mut pt_score = words.iter().filter(|w| pt_words.contains(w)).count();
            pt_score += chars.iter().filter(|&c| "ãõ".contains(*c)).count() * 3;
            if words.contains(&"quem") {
                pt_score += 4;
            }
            if words.contains(&"do") || words.contains(&"não") {
                pt_score += 3;
            }
            if words.contains(&"em") && !words.contains(&"en") {
                pt_score += 2;
            }

            let mut it_score = words.iter().filter(|w| it_words.contains(w)).count();
            it_score += words
                .iter()
                .filter(|w| {
                    w.ends_with("zione")
                        || w.ends_with("zioni")
                        || w.ends_with("aggio")
                        || w.ends_with("mente")
                })
                .count();
            if words.contains(&"chi") {
                it_score += 4;
            }
            if words.contains(&"quale")
                || words.contains(&"quali")
                || words.contains(&"cosa")
                || words.contains(&"dove")
            {
                it_score += 4;
            }
            if words.contains(&"di") {
                it_score += 3;
            }
            if words.contains(&"il") || words.contains(&"gli") {
                it_score += 3;
            }

            let mut es_score = words.iter().filter(|w| es_words.contains(w)).count();
            es_score += chars.iter().filter(|&c| "¿¡ñ".contains(*c)).count() * 3;
            if words.contains(&"quién") {
                es_score += 4;
            }
            if words.contains(&"el") {
                es_score += 3;
            }
            if words.contains(&"en") && !words.contains(&"em") {
                es_score += 2;
            }

            let scores = [
                ("it", it_score, 0.05 * pi, 0.95 * pi), // it (s=2: bit0=0, bit1=1)
                ("pt", pt_score, 0.95 * pi, 0.95 * pi), // pt (s=3: bit0=1, bit1=1)
                ("fr", fr_score, 0.95 * pi, 0.05 * pi), // fr (s=1: bit0=1, bit1=0)
                ("es", es_score, 0.05 * pi, 0.05 * pi), // es (s=0: bit0=0, bit1=0)
            ];
            let mut best_score = -1isize;
            let mut best_angles = (0.05 * pi, 0.05 * pi);
            for &(_code, score, a2, a3) in &scores {
                if (score as isize) > best_score {
                    best_score = score as isize;
                    best_angles = (a2, a3);
                }
            }
            angles[2] = best_angles.0;
            angles[3] = best_angles.1;
        }
    } else {
        // Asian / Afro
        if angles[0] < 0.5 * pi {
            // East Asian: zh (s=0: 00), ja (s=1: 10), ko (s=2: 01)
            if n_hangul > 0 {
                angles[2] = 0.05 * pi; // bit 0 = 0
                angles[3] = 0.95 * pi; // bit 1 = 1
            } else if n_kana > 0 {
                angles[2] = 0.95 * pi; // bit 0 = 1
                angles[3] = 0.05 * pi; // bit 1 = 0
            } else {
                angles[2] = 0.05 * pi; // bit 0 = 0
                angles[3] = 0.05 * pi; // bit 1 = 0
            }
        } else {
            // Afro-Indic: he (s=0: 00), hi (s=1: 10), id (s=2: 01)
            if n_hebrew > 0 && n_hebrew >= n_deva {
                angles[2] = 0.05 * pi; // bit 0 = 0
                angles[3] = 0.05 * pi; // bit 1 = 0
            } else if n_deva > 0 {
                angles[2] = 0.95 * pi; // bit 0 = 1
                angles[3] = 0.05 * pi; // bit 1 = 0
            } else {
                angles[2] = 0.05 * pi; // bit 0 = 0
                angles[3] = 0.95 * pi; // bit 1 = 1
            }
        }
    }

    // Wires 4..9: Environment noise & text entropy
    let len_factor = (chars.len() as f64 / 150.0).clamp(0.0, 1.0);
    angles[4] = len_factor * pi;

    let punct_count = chars.iter().filter(|c| c.is_ascii_punctuation()).count();
    let punct_ratio = (punct_count as f64 / chars.len().max(1) as f64 * 10.0).clamp(0.0, 1.0);
    angles[5] = punct_ratio * pi;

    let vowels = "aeiouAEIOU";
    let vowel_count = chars
        .iter()
        .filter(|&&c| vowels.chars().any(|v| v == c))
        .count();
    let vowel_ratio = (vowel_count as f64 / chars.len().max(1) as f64 * 3.0).clamp(0.0, 1.0);
    angles[6] = vowel_ratio * pi;

    let word_count = words.len().max(1) as f64;
    angles[7] = (word_count / 30.0).clamp(0.0, 1.0) * pi;

    let upper_count = chars.iter().filter(|c| c.is_uppercase()).count();
    angles[8] = (upper_count as f64 / chars.len().max(1) as f64 * 5.0).clamp(0.0, 1.0) * pi;

    angles[9] = ((chars.len() % 16) as f64 / 16.0) * pi;

    angles
}
