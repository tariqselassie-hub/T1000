use t1000_core::*;

fn main() {
    println!("=== quantum_demo ===");

    let reg = QubitRegister::new(16, 42);
    println!("register basis=16 entropy={:.4}", reg.entropy());

    let other = QubitRegister::new(8, 7);
    let joint = reg.tensor_product(&other);
    println!("tensor basis={} (expected 128)", joint.basis);

    let proj = Projector::new((0..16).collect());
    let collapsed = collapse_with_seed(&reg, &proj, 42);
    println!("collapse {:?}", collapsed);

    let ch = DecoherenceChannel::new(ChannelKind::AmplitudeDamping { gamma: 0.1 }, 1);
    let mut damped = reg.clone();
    let lost = apply(&mut damped, &ch);
    println!(
        "damping lost={:.4} entropy_after={:.4}",
        lost,
        damped.entropy()
    );

    let cc = constructive_score(&reg, &other);
    let cd = destructive_score(&reg, &other);
    println!(
        "interference constructive={:.4} destructive={:.4} net={:.4}",
        cc,
        cd,
        cc - cd
    );

    let a = vec![0u64, 1, 2, 3];
    let b = vec![1u64, 1, 2, 2];
    let et = EntanglementTensor::from_events(&a, &b);
    println!(
        "tensor schmidt_rank={} mutual_info={:.4}",
        et.schmidt_rank(),
        et.mutual_information()
    );
}
