use num_complex::Complex64;

#[derive(Debug, Clone, Copy)]
pub struct PartialTrace;

impl PartialTrace {
    pub fn trace_out_b(tensor_state: &[Complex64], dim_a: usize, dim_b: usize) -> Vec<Complex64> {
        let mut rho_a = vec![Complex64::default(); dim_a * dim_a];
        for i in 0..dim_a {
            for j in 0..dim_a {
                let mut sum = Complex64::default();
                for k in 0..dim_b {
                    sum += tensor_state[i * dim_b + k] * tensor_state[j * dim_b + k].conj();
                }
                rho_a[i * dim_a + j] = sum;
            }
        }
        rho_a
    }

    pub fn partial_trace_b(tensor_state: &[Complex64], dim_a: usize, dim_b: usize) -> Vec<Complex64> {
        Self::trace_out_b(tensor_state, dim_a, dim_b)
    }
}
