use num_complex::Complex64;
use t1000_core::*;

fn main() {
    println!("=== quantum_system_demo ===");

    let reg = QubitRegister::new(16, 42);
    let other = QubitRegister::new(8, 7);
    println!("register basis=16 entropy={:.4}", reg.entropy());

    let joint = reg.tensor_product(&other);
    println!("tensor basis={} (expected 128)", joint.basis);

    let phases: Vec<f64> = (0..joint.basis)
        .map(|i| i as f64 * std::f64::consts::PI / joint.basis as f64)
        .collect();
    let mut phased = joint.clone();
    PhaseGate::new(phases).apply(&mut phased);
    println!(
        "phase_gate max_amp_imag={:.4}",
        phased.amplitudes.iter().map(|a| a.im).fold(0.0, f64::max)
    );

    let energies = vec![0.3; 16];
    let ham = Hamiltonian::new(16, energies);
    let mut evolved = reg.clone();
    hamiltonian_time_evolve(&ham, &mut evolved, 1.0);
    println!(
        "hamiltonian evolve norm_sq_sum={:.6}",
        evolved.amplitudes.iter().map(|a| a.norm_sqr()).sum::<f64>()
    );

    let perm = Permutation::new((0..16).rev().collect());
    let mut permuted = reg.clone();
    permutation_apply(&perm, &mut permuted);
    println!(
        "permutation reversed_amp[0]={:.6}",
        permuted.amplitudes[0].re
    );

    let tensor_state: Vec<_> = joint.amplitudes.clone();
    let rho_a = PartialTrace::trace_out_b(&tensor_state, 16, 8);
    println!("partial_trace rho_a trace_0_0={:.6}", rho_a[0].re);

    let povm_ops = vec![vec![Complex64::new(1.0, 0.0); 16]];
    let povm = POVM::new(povm_ops);
    let probs = povm_measure_soft(&povm, &reg);
    println!("povm p0={:.6}", probs.first().copied().unwrap_or(0.0));

    println!("=== DONE ===");
}
