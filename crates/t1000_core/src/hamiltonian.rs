use num_complex::Complex64;

use crate::QubitRegister;

/// Diagonal Hermitian generator for amplitudes.
#[derive(Debug, Clone)]
pub struct Hamiltonian {
    pub dim: usize,
    pub diagonal_energies: Vec<f64>,
}

impl Hamiltonian {
    pub fn new(dim: usize, energies: Vec<f64>) -> Self {
        assert_eq!(energies.len(), dim);
        Self {
            dim,
            diagonal_energies: energies,
        }
    }

    /// Evolves `reg` in-place under U(t) = exp(-i * H * t).
    pub fn time_evolve(&self, reg: &mut QubitRegister, t: f64) {
        assert_eq!(reg.basis, self.dim);
        for (amp, e) in reg.amplitudes.iter_mut().zip(self.diagonal_energies.iter()) {
            let phase = -e * t;
            *amp *= Complex64::from_polar(1.0, phase);
        }
    }
}

pub fn hamiltonian_time_evolve(ham: &Hamiltonian, reg: &mut QubitRegister, t: f64) {
    ham.time_evolve(reg, t);
}
