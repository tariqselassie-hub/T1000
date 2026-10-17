use super::QubitRegister;

pub fn constructive_score(a: &QubitRegister, b: &QubitRegister) -> f64 {
    joint_probability_positive(a, b)
}

pub fn destructive_score(a: &QubitRegister, b: &QubitRegister) -> f64 {
    joint_probability_negative(a, b)
}

pub fn net_interference(a: &QubitRegister, b: &QubitRegister) -> f64 {
    constructive_score(a, b) - destructive_score(a, b)
}

fn joint_probability_positive(a: &QubitRegister, b: &QubitRegister) -> f64 {
    let min_basis = a.basis.min(b.basis);
    (0..min_basis)
        .map(|i| {
            let p = a.probability(i) * b.probability(i);
            if (a.amplitudes[i] * b.amplitudes[i]).re.is_sign_positive() {
                p
            } else {
                0.0
            }
        })
        .sum::<f64>()
        .sqrt()
}

fn joint_probability_negative(a: &QubitRegister, b: &QubitRegister) -> f64 {
    let min_basis = a.basis.min(b.basis);
    (0..min_basis)
        .map(|i| {
            let p = a.probability(i) * b.probability(i);
            if (a.amplitudes[i] * b.amplitudes[i]).re.is_sign_negative() {
                p
            } else {
                0.0
            }
        })
        .sum::<f64>()
        .sqrt()
}
