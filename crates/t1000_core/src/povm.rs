use num_complex::Complex64;

/// Generalized measurement without full collapse.
#[derive(Debug, Clone)]
pub struct POVM {
    pub operators: Vec<Vec<Complex64>>,
}

impl POVM {
    pub fn new(operators: Vec<Vec<Complex64>>) -> Self {
        Self { operators }
    }

    /// Soft measurement outcome probabilities, loosely `Tr(E_k rho)`.
    pub fn measure_soft(&self, reg: &crate::QubitRegister) -> Vec<f64> {
        let n = reg.amplitudes.len() as f64;
        self.operators
            .iter()
            .map(|op| {
                let mut proj = Complex64::default();
                for (i, amp) in reg.amplitudes.iter().enumerate() {
                    if let Some(o) = op.get(i) {
                        proj += o.conj() * amp;
                    }
                }
                proj.norm_sqr() / n
            })
            .collect()
    }
}

pub fn povm_measure_soft(povm: &POVM, reg: &crate::QubitRegister) -> Vec<f64> {
    povm.measure_soft(reg)
}
