use crate::channel::{apply, ChannelKind, DecoherenceChannel};
use crate::hamiltonian::{hamiltonian_time_evolve, Hamiltonian};
use crate::interference::net_interference;
use crate::phase_gate::PhaseGate;
use crate::projector::collapse_with_seed;
use crate::QubitRegister;
use num_complex::Complex64;

fn born_probs(reg: &QubitRegister) -> Vec<f64> {
    let total: f64 = reg.amplitudes.iter().map(|a| a.norm_sqr()).sum();
    if total > 0.0 {
        reg.amplitudes.iter().map(|a| a.norm_sqr() / total).collect()
    } else {
        vec![0.0; reg.basis]
    }
}

/// Directly nudge amplitude phases toward the low-energy target phase vector.
fn rotate_amplitudes_toward_phases(reg: &mut QubitRegister, target_phases: &[f64], step: f64) {
    assert_eq!(reg.basis, target_phases.len());
    let s = step.clamp(0.01, 0.5);
    for (i, a) in reg.amplitudes.iter_mut().enumerate() {
        let target = target_phases[i];
        let current = a.arg();
        let mut diff = target - current;
        diff = diff.rem_euclid(2.0 * std::f64::consts::PI);
        if diff > std::f64::consts::PI {
            diff -= 2.0 * std::f64::consts::PI;
        }
        let rot = diff * s;
        let (sr, cr) = rot.sin_cos();
        let re = a.re * cr - a.im * sr;
        let im = a.re * sr + a.im * cr;
        *a = Complex64::new(re, im);
    }
}

/// Blend amplitudes toward a target magnitude profile: low-E states grow, high-E states shrink.
fn nudge_magnitudes_toward_energy(
    reg: &mut QubitRegister,
    norm_energies: &[f64],
    strength: f64,
) {
    assert_eq!(reg.basis, norm_energies.len());
    let s = strength.clamp(0.0, 0.4);
    for (i, a) in reg.amplitudes.iter_mut().enumerate() {
        let target_mag = if norm_energies[i] < 0.5 {
            1.0 - norm_energies[i]
        } else {
            0.0
        };
        let cur_phase = a.arg();
        let new_mag = a.norm() * (1.0 - s) + target_mag * s;
        *a = Complex64::from_polar(new_mag, cur_phase);
    }
}

/// Configuration for the quantum-inspired simulated bifurcation search.
#[derive(Debug, Clone, Copy)]
pub struct BifurcationConfig {
    /// Number of candidate configurations (= QubitRegister basis size).
    pub basis: usize,
    /// Seed for register initialization + RNG.
    pub seed: u64,
    /// Maximum iterations of phase→evolve→measure cycles.
    pub max_iters: usize,
    /// Phase slope toward low-energy candidates.
    /// γ in [0.1, π] works; higher concentrates amplitude faster.
    pub gamma: f64,
    /// Time step per Hamiltonian evolution.
    pub dt: f64,
    /// Seed for final Projector-based collapse (deterministic re-runs).
    pub collapse_seed: u64,
}

impl Default for BifurcationConfig {
    fn default() -> Self {
        Self {
            basis: 16,
            seed: 42,
            max_iters: 6,
            gamma: 1.2,
            dt: 0.5,
            collapse_seed: 7,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BifurcationResult {
    /// Index of the winning candidate.
    pub winner_index: usize,
    /// Raw energy of the winner.
    pub winner_energy: f64,
    /// Born-rule probability of the winner after final POVM measure.
    pub winner_probability: f64,
    /// Top-K contenders ranked by POVM probability.
    pub top_k: Vec<(usize, f64, f64)>,
    /// Final register state (useful for interference checks vs target).
    pub final_register: QubitRegister,
    /// Actual iterations run.
    pub iterations_ran: usize,
}

/// Run a quantum-inspired simulated bifurcation over a discrete energy landscape.
///
/// # Algorithm
/// 1. Normalize energies to [0,1].
/// 2. Map each normalized energy to a phase rotation: low energy → large phase
///    (constructive interference for minima).
/// 3. Iterate: phase gate → Hamiltonian time evolve → normalize → POVM measure
///    → keep top-K contenders.
/// 4. Amplitude-damp → collapse to final winner.
///
/// # Panics
/// Panics if `energies` is empty or `energies.len() != cfg.basis`.
pub fn simulated_bifurcation(
    energies: &[f64],
    cfg: BifurcationConfig,
) -> BifurcationResult {
    assert!(!energies.is_empty(), "energies must be non-empty");
    assert_eq!(
        energies.len(),
        cfg.basis,
        "energies len {} must match basis {}",
        energies.len(),
        cfg.basis
    );

    // --- energy normalization ---
    let e_min = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let e_max = energies
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(e_min + 1e-12);
    let e_range = e_max - e_min;

    let norm_energies: Vec<f64> = energies
        .iter()
        .map(|&e| ((e - e_min) / e_range).clamp(0.0, 1.0))
        .collect();

    // --- phase map: low energy → large phase ---
    let phases: Vec<f64> = norm_energies
        .iter()
        .map(|&ne| cfg.gamma * (1.0 - ne) * std::f64::consts::PI)
        .collect();

    // --- Hamiltonian with shifted energies (diagonal) ---
    let diag: Vec<f64> = energies.iter().map(|&e| e - e_min).collect();
    let hamiltonian = Hamiltonian::new(cfg.basis, diag);

    // --- initial uniform superposition ---
    let mut reg = QubitRegister::new(cfg.basis, cfg.seed);

    // --- keep top-K basis indices for collapse projector ---
    const TOP_K: usize = 8;
    let mut top_indices: Vec<usize> = (0..cfg.basis.min(TOP_K)).collect();

    let mut iterations_ran = 0usize;

    for iter_idx in 0..cfg.max_iters {
        // 1. Phase toward low-energy states
        PhaseGate::new(phases.clone()).apply(&mut reg);
        // 2. Rotate amplitude phases toward the low-energy target
        let step = 0.35 * (1.0 + iter_idx as f64 / cfg.max_iters.max(1) as f64);
        rotate_amplitudes_toward_phases(&mut reg, &phases, step);
        // 3. Grow low-E magnitudes, drain high-E magnitudes
        nudge_magnitudes_toward_energy(&mut reg, &norm_energies, 0.25);
        // 4. Hamiltonian evolve with growing time
        let t = cfg.dt * (1.0 + iter_idx as f64 * 0.5);
        hamiltonian_time_evolve(&hamiltonian, &mut reg, t);
        reg.normalize();

        // 5. Born-rule measurement → update top-K
        let probs = born_probs(&reg);

        let mut indexed: Vec<(usize, f64)> = probs
            .into_iter()
            .enumerate()
            .filter(|(_, p)| p.is_finite())
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        top_indices.clear();
        for (idx, _) in indexed.iter().take(TOP_K) {
            top_indices.push(*idx);
        }

        iterations_ran = iter_idx + 1;

        // early-stop if winner dominates
        if let Some((_, top_p)) = indexed.first() {
            if *top_p > 0.998 {
                break;
            }
        }
    }

    // 4. Decoherence then collapse
    let ch = DecoherenceChannel::new(
        ChannelKind::AmplitudeDamping { gamma: cfg.gamma.clamp(0.01, 0.5) },
        cfg.collapse_seed,
    );
    apply(&mut reg, &ch);
    reg.normalize();

    let proj = crate::Projector::new(top_indices);
    let outcome = collapse_with_seed(&reg, &proj, cfg.collapse_seed);

    let (winner_idx, winner_prob) = outcome.unwrap_or((0, 0.0));

    // 5. Final Born-rule prob accounting
    let final_probs = born_probs(&reg);
    let winner_energy = energies[winner_idx];

    let mut top_k: Vec<(usize, f64, f64)> = final_probs
        .into_iter()
        .enumerate()
        .filter(|(_, p)| p.is_finite() && *p > 1e-12)
        .map(|(i, p)| (i, p, energies[i]))
        .collect();
    top_k.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top_k.truncate(TOP_K);

    BifurcationResult {
        winner_index: winner_idx,
        winner_energy,
        winner_probability: winner_prob,
        top_k,
        final_register: reg.clone(),
        iterations_ran,
    }
}

/// Compute net interference between the final winner register and the
/// single-candidate register that encodes a candidate index.
pub fn bifurcation_interference(
    result: &BifurcationResult,
    target_candidate: usize,
) -> f64 {
    let mut target = QubitRegister::new(result.final_register.basis, result.final_register.seed);
    target.amplitudes[target_candidate % target.basis] = Complex64::new(1.0, 0.0);
    net_interference(&result.final_register, &target)
}
