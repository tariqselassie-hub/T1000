use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::QubitRegister;

/// Selective projection operator for `QubitRegister` measurement.
#[derive(Debug, Clone)]
pub struct Projector {
    pub basis: Vec<usize>,
}

impl Projector {
    pub fn new(basis: Vec<usize>) -> Self {
        Self { basis }
    }
}

/// Collapse a register onto the projector's subspace using `seed`-consistent selection.
/// Returns `(outcome_index, probability)` if the projector has nonzero overlap,
/// otherwise `None`.
pub fn collapse_with_seed(
    reg: &QubitRegister,
    proj: &Projector,
    seed: u64,
) -> Option<(usize, f64)> {
    if proj.basis.is_empty() {
        return None;
    }
    let weights: Vec<f64> = proj
        .basis
        .iter()
        .map(|&i| {
            let idx = i.min(reg.basis.saturating_sub(1));
            reg.probability(idx)
        })
        .collect();
    let total: f64 = weights.iter().sum();
    if total < 1e-12 {
        return None;
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut r = rng.gen::<f64>() * total;
    for (idx, w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            let outcome = proj.basis[idx];
            return Some((outcome, *w / total));
        }
    }
    let last = proj.basis.len() - 1;
    Some((proj.basis[last], weights[last] / total))
}

/// Sample all projector outcomes with probabilities.
pub fn measure_all(reg: &QubitRegister, proj: &Projector) -> Vec<(usize, f64)> {
    proj.basis
        .iter()
        .filter_map(|&i| {
            let idx = i.min(reg.basis.saturating_sub(1));
            let p = reg.probability(idx);
            if p > 0.0 {
                Some((i, p))
            } else {
                None
            }
        })
        .collect()
}
