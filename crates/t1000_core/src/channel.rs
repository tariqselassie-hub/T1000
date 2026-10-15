use crate::QubitRegister;
use num_complex::Complex64;
use rand::Rng;
use rand::SeedableRng;

#[derive(Debug, Clone, Copy)]
pub enum ChannelKind {
    AmplitudeDamping { gamma: f64 },
    Depolarizing { p: f64 },
    BitFlip { p: f64 },
}

impl Default for ChannelKind {
    fn default() -> Self {
        ChannelKind::AmplitudeDamping { gamma: 0.05 }
    }
}

#[derive(Debug, Clone)]
pub struct DecoherenceChannel {
    pub kind: ChannelKind,
    pub seed: u64,
}

impl DecoherenceChannel {
    pub fn new(kind: ChannelKind, seed: u64) -> Self {
        Self { kind, seed }
    }
}

/// Apply a decoherence channel to `reg` in-place.
/// Returns the total lost coherence: sum of squared probability mass that moved.
pub fn apply(reg: &mut QubitRegister, ch: &DecoherenceChannel) -> f64 {
    let mut lost = 0.0;
    match ch.kind {
        ChannelKind::AmplitudeDamping { gamma } => {
            for a in &mut reg.amplitudes {
                let old_p = a.norm_sqr();
                let damped = *a * Complex64::new(1.0 - gamma, 0.0);
                let new_p = damped.norm_sqr();
                lost += (old_p - new_p).clamp(0.0, f64::MAX);
                *a = damped;
            }
            // Redistribute lost coherence uniformly.
            let basis_f = reg.basis as f64;
            for a in &mut reg.amplitudes {
                *a += Complex64::new((lost / basis_f).sqrt(), 0.0);
            }
        }
        ChannelKind::Depolarizing { p } => {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(ch.seed);
            for a in &mut reg.amplitudes {
                if rng.gen::<f64>() < p {
                    let old_p = a.norm_sqr();
                    lost += old_p.clamp(0.0, f64::MAX);
                    *a = Complex64::default();
                }
            }
            let basis_f = reg.basis as f64;
            for a in &mut reg.amplitudes {
                *a += Complex64::new((lost / basis_f).sqrt(), 0.0);
            }
        }
        ChannelKind::BitFlip { p } => {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(ch.seed);
            for a in &mut reg.amplitudes {
                if rng.gen::<f64>() < p {
                    *a = -*a;
                }
            }
        }
    }
    reg.normalize();
    lost
}
