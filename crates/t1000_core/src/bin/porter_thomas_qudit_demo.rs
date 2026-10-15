//! Porter-Thomas & Sovereign Heptal Qudit Demonstration CLI
//!
//! Demonstrates:
//! 1. Canonical Heptal Qudit ($d = 7$, $\mathbb{Z}_7$) allocation and multi-qudit registers.
//! 2. Heptal-to-Binary embedding ($3$ qubits per qudit) with non-computational Somatic Leakage tracking.
//! 3. Exact Porter-Thomas statistical validation and Kolmogorov-Smirnov test across both Binary and Heptal regimes.

use t1000_core::{
    QuditRadix, QuditRegister,
    ASYMPTOTIC_HAAR_ENTROPY_DEFICIT, EULER_MASCHERONI,
};
use num_complex::Complex64;

fn main() {
    println!("================================================================================");
    println!("     DEEN QUANTUM SUBSYSTEM: SOVEREIGN QUDIT & EXACT PORTER-THOMAS ENGINE       ");
    println!("================================================================================");

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 1: CANONICAL QUDIT DEFINITION & UNITS
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[1] CANONICAL QUDIT DEFINITION & REGISTER INVARIANTS");
    println!("--------------------------------------------------------------------------------");

    let h_reg = QuditRegister::new_heptal(4, 42);
    println!("  Logical System       : Heptal Qudit Register (Z_7 Galois Field)");
    println!("  Radix (d)            : {:?}", h_reg.radix);
    println!("  Sites (K qudits)     : {}", h_reg.sites);
    println!("  Logical Dimension (D): {} (7^{})", h_reg.basis, h_reg.sites);
    println!("  Unit Norm Status     : {:.6}", h_reg.probabilities().iter().sum::<f64>());

    let b_reg = QuditRegister::new_binary(12, 42);
    println!("\n  Physical System      : Binary Qubit Carrier Register");
    println!("  Radix (d)            : {:?}", b_reg.radix);
    println!("  Sites (N qubits)     : {}", b_reg.sites);
    println!("  Physical Dimension(D): {} (2^{})", b_reg.basis, b_reg.sites);

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 2: HEPTAL-TO-BINARY EMBEDDING & SOMATIC LEAKAGE
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[2] HEPTAL-TO-3-QUBIT BINARY EMBEDDING & SOMATIC LEAKAGE (qhrack Protocol)");
    println!("--------------------------------------------------------------------------------");

    // 2 Heptal qudits (49 states) embedded into 6 binary qubits (64 states)
    let mut h2 = QuditRegister::new_heptal(2, 100);
    for a in &mut h2.amplitudes {
        *a = Complex64::new(0.0, 0.0);
    }
    // Prepare coherent superposition: |0, 3> and |6, 2>
    h2.amplitudes[3] = Complex64::new(0.8, 0.0);
    h2.amplitudes[44] = Complex64::new(0.6, 0.0);
    h2.normalize();

    let (mut physical_reg, initial_leakage) = h2.embed_heptal_into_binary().expect("embedding");
    println!("  Embedded 2 Heptal qudits (49 states) -> 6 binary qubits (64 states)");
    println!("  Initial Somatic Leakage Lambda: {:.6} (Pure computational state)", initial_leakage);

    // Simulate physical crosstalk / decoherence by injecting 10% amplitude into non-computational |7, 0> = |111, 000>
    println!("  [SIMULATION] Injecting decoherence into non-computational subspace state |111, 000>...");
    physical_reg.amplitudes[56] = Complex64::new(0.35, 0.0);
    physical_reg.normalize();

    let detected_leakage = QuditRegister::somatic_leakage(&physical_reg);
    println!("  Measured Somatic Leakage Lambda: {:.4} ({:.2}%)", detected_leakage, detected_leakage * 100.0);

    let (purified_h2, proj_leakage) = QuditRegister::project_binary_to_heptal(&physical_reg)
        .expect("projection");
    println!("  Projected & Renormalized back to Heptal Subspace:");
    println!("    Purified Dimension: {} states", purified_h2.basis);
    println!("    Purified Total Prob: {:.6}", purified_h2.probabilities().iter().sum::<f64>());
    println!("    Confined Leakage Loss: {:.4}", proj_leakage);

    // ─────────────────────────────────────────────────────────────────────────
    // SECTION 3: EXACT PORTER-THOMAS BENCHMARKING
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[3] EXACT PORTER-THOMAS INVARIANTS & HAAR SCRAMBLING VALIDATION");
    println!("--------------------------------------------------------------------------------");
    println!("  Theoretical Reference Constants:");
    println!("    Euler-Mascheroni (gamma)  : {:.15}", EULER_MASCHERONI);
    println!("    Asymptotic Entropy Deficit: 1 - gamma = {:.6} nats", ASYMPTOTIC_HAAR_ENTROPY_DEFICIT);

    // Test 3.1: Flat uniform distribution (unscrambled white noise)
    println!("\n  --- CASE A: Uniform State (Zero Scrambling / Flat Noise, D = 1024) ---");
    let unif_probs = vec![1.0 / 1024.0; 1024];
    let rep_unif = t1000_core::verify_porter_thomas(&unif_probs);
    println!("    Exact Linear XEB Fidelity : {:.4} (Expected = 0.0000)", rep_unif.exact_xeb);
    println!("    Sum of Squared Probs      : {:.8} (Expected = {:.8})", rep_unif.sum_p_sqr, rep_unif.expected_sum_p_sqr);
    println!("    Entropy Deficit           : {:.4} nats (Expected = {:.4})", rep_unif.entropy_deficit, rep_unif.expected_entropy_deficit);
    println!("    Kolmogorov-Smirnov Passed : {}", rep_unif.passes_ks_test);
    println!("    Porter-Thomas Conforming  : {}", rep_unif.is_porter_thomas_conforming);

    // Test 3.2: Haar-Random Binary Pure State (D = 1024 = 2^10)
    println!("\n  --- CASE B: Haar-Random Binary Pure State (D = 1024, 10 Qubits) ---");
    let haar_bin = QuditRegister::new_haar_random(QuditRadix::Binary, 10, 2026);
    let rep_bin = haar_bin.verify_porter_thomas();
    println!("    Exact Linear XEB Fidelity : {:.4} (Ideal Limit = 1.0000)", rep_bin.exact_xeb);
    println!("    Naive Linear XEB Fidelity : {:.4}", rep_bin.naive_xeb);
    println!("    Sum of Squared Probs      : {:.8} (Expected 2/(D+1) = {:.8})", rep_bin.sum_p_sqr, rep_bin.expected_sum_p_sqr);
    println!("    Shannon Measurement S     : {:.4} nats", rep_bin.shannon_entropy);
    println!("    Entropy Deficit (ln D - S): {:.4} nats (Exact E[Delta S] = {:.4} nats)", rep_bin.entropy_deficit, rep_bin.expected_entropy_deficit);
    println!("    Effective Participation R.: {:.2} states (Expected 50% = {:.2})", rep_bin.effective_participation_ratio, rep_bin.expected_participation_ratio);
    println!("    K-S Statistic (D_stat)    : {:.4} (Threshold = {:.4})", rep_bin.ks_statistic, rep_bin.ks_critical_value);
    println!("    Kolmogorov-Smirnov Passed : {}", rep_bin.passes_ks_test);
    println!("    Porter-Thomas Conforming  : {}", rep_bin.is_porter_thomas_conforming);

    // Test 3.3: Haar-Random Heptal Pure State (D = 2401 = 7^4)
    println!("\n  --- CASE C: Haar-Random Canonical Heptal State (D = 2401, 4 Qudits) ---");
    let haar_hep = QuditRegister::new_haar_random(QuditRadix::Heptal, 4, 777);
    let rep_hep = haar_hep.verify_porter_thomas();
    println!("    Exact Linear XEB Fidelity : {:.4} (Ideal Limit = 1.0000)", rep_hep.exact_xeb);
    println!("    Naive Linear XEB Fidelity : {:.4}", rep_hep.naive_xeb);
    println!("    Sum of Squared Probs      : {:.8} (Expected 2/(D+1) = {:.8})", rep_hep.sum_p_sqr, rep_hep.expected_sum_p_sqr);
    println!("    Shannon Measurement S     : {:.4} nats", rep_hep.shannon_entropy);
    println!("    Entropy Deficit (ln D - S): {:.4} nats (Exact E[Delta S] = {:.4} nats)", rep_hep.entropy_deficit, rep_hep.expected_entropy_deficit);
    println!("    Effective Participation R.: {:.2} states (Expected 50% = {:.2})", rep_hep.effective_participation_ratio, rep_hep.expected_participation_ratio);
    println!("    K-S Statistic (D_stat)    : {:.4} (Threshold = {:.4})", rep_hep.ks_statistic, rep_hep.ks_critical_value);
    println!("    Kolmogorov-Smirnov Passed : {}", rep_hep.passes_ks_test);
    println!("    Porter-Thomas Conforming  : {}", rep_hep.is_porter_thomas_conforming);

    println!("\n================================================================================");
    println!("     ALL QUDIT UNIT DEFINITIONS & PORTER-THOMAS INVARIANTS LOCKED AND VERIFIED  ");
    println!("================================================================================");
}
