// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! How many gates does the ZINC model need on real qubit connectivities?
//!
//! The circuit is the state preparation of one molecule followed by the depth-4, 10-qubit ansatz
//! (whose entangling strides are 1, 2, 4 and 2). For each topology a placement is searched on a few
//! molecules (best fit) and compared with the naive identity placement, then scored on many
//! molecules. Routing is greedy, so counts are an upper bound; the ranking between topologies is the
//! result. Every routed circuit for the first molecule is verified by simulation.
//!
//! `--dataset qm9` routes QM9 molecules instead.
//!
//! Usage: `zinc_routing_study [--restarts N] [--search-molecules K] [--eval-molecules M] [--depth D]`

use serde::Serialize;
use std::env;
use t1000::routing::{ansatz_program, cost, mean_cost, prep_program, route, search_layout, verify, Item, Topology};
use t1000::state_prep::amplitude_prep;
use t1000::datasets::DataOptions;
use t1000::zinc_data::{InputScaling, Scaler};

const N_QUBITS: usize = 10;

#[derive(Serialize, Clone, Copy)]
struct Stats {
    mean: f64,
    min: usize,
    max: usize,
}

fn stats(v: &[usize]) -> Stats {
    Stats { mean: v.iter().sum::<usize>() as f64 / v.len() as f64, min: *v.iter().min().unwrap(), max: *v.iter().max().unwrap() }
}

#[derive(Serialize)]
struct Placement {
    two_qubit_total: Stats,
    swaps: Stats,
    depth: Stats,
    single_qubit: Stats,
    /// Cost of routing only the ansatz from this placement.
    ansatz_only_two_qubit_total: usize,
}

#[derive(Serialize)]
struct TopologyResult {
    topology: String,
    edges: usize,
    best_layout: Vec<usize>,
    best: Placement,
    naive: Placement,
    /// Circuit fidelity of the best placement for two-qubit errors 1e-3, 5e-3 and 1e-2.
    fidelity: [f64; 3],
    naive_fidelity: [f64; 3],
}

fn programs(depth: usize, molecules: &[Vec<f64>]) -> Vec<Vec<Item>> {
    let ansatz = ansatz_program(N_QUBITS, depth);
    molecules
        .iter()
        .map(|amps| {
            let mut p = prep_program(&amplitude_prep(amps));
            p.extend(ansatz.iter().cloned());
            p
        })
        .collect()
}

fn evaluate(progs: &[Vec<Item>], topo: &Topology, layout: &[usize], depth: usize) -> Placement {
    let costs: Vec<_> = progs.iter().map(|p| cost(&route(p, topo, layout).gates)).collect();
    let pick = |f: fn(&t1000::routing::Cost) -> usize| stats(&costs.iter().map(f).collect::<Vec<_>>());
    let ansatz_only = cost(&route(&ansatz_program(N_QUBITS, depth), topo, layout).gates).two_qubit_total();
    Placement {
        two_qubit_total: pick(|c| c.two_qubit_total()),
        swaps: pick(|c| c.swaps),
        depth: pick(|c| c.depth),
        single_qubit: pick(|c| c.single),
        ansatz_only_two_qubit_total: ansatz_only,
    }
}

fn fidelities(p: &Placement) -> [f64; 3] {
    [1e-3f64, 5e-3, 1e-2].map(|p2| (1.0 - p2 / 10.0).powf(p.single_qubit.mean) * (1.0 - p2).powf(p.two_qubit_total.mean))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let value = |flag: &str, default: usize| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(default);
    let (restarts, search_n, eval_n, depth) = (value("--restarts", 40), value("--search-molecules", 4), value("--eval-molecules", 200), value("--depth", 4));

    let data_options = DataOptions::from_args(&args);
    let data = data_options.load();
    let train = data.train;
    let fit_rows: Vec<&[f64]> = train.iter().map(|m| m.features.as_slice()).collect();
    let scaler = Scaler::fit_rows(InputScaling::MaxAbs, &fit_rows);
    let amps = |idx: usize| -> Vec<f64> {
        let mut a = vec![0.0; 32];
        let f = scaler.apply(&train[idx].features);
        a[..f.len()].copy_from_slice(&f);
        a
    };
    let step = train.len() / eval_n;
    let eval_set: Vec<Vec<f64>> = (0..eval_n).map(|i| amps(i * step)).collect();
    let search_set: Vec<Vec<f64>> = (0..search_n).map(|i| amps(i * step + step / 2)).collect();
    let (eval_progs, search_progs) = (programs(depth, &eval_set), programs(depth, &search_set));

    let topologies = [Topology::all_to_all(N_QUBITS), Topology::line(N_QUBITS), Topology::ring(N_QUBITS), Topology::grid(2, 5), Topology::heavy_hex_fragment()];
    let identity: Vec<usize> = (0..N_QUBITS).collect();
    println!("{}: routing the preparation + depth-{depth} ansatz of {eval_n} molecules ({} layout restarts, searched on {search_n} molecules)\n", data.label, restarts);
    let mut results = Vec::new();
    for topo in &topologies {
        let (layout, searched) = search_layout(&search_progs, topo, restarts, 7);
        let (best, naive) = (evaluate(&eval_progs, topo, &layout, depth), evaluate(&eval_progs, topo, &identity, depth));
        // Verify every routed circuit of the first molecule against the original.
        let dev = verify(&eval_progs[0], &route(&eval_progs[0], topo, &layout), N_QUBITS).max(verify(&eval_progs[0], &route(&eval_progs[0], topo, &identity), N_QUBITS));
        println!("{:<11} best layout {:?}  (search objective {:.1}, verified to {:.1e})", topo.name, layout, searched, dev);
        let _ = mean_cost(&search_progs, topo, &layout);
        results.push(TopologyResult { topology: topo.name.clone(), edges: topo.edges().len(), best_layout: layout, fidelity: fidelities(&best), naive_fidelity: fidelities(&naive), best, naive });
    }

    println!("\nTwo-qubit gates per molecule (CNOT-equivalents, SWAP = 3) and circuit fidelity");
    println!("┌────────────┬───────┬──────────────────────┬────────────────────┬────────────────────┬─────────────────────────┐");
    println!("│ topology   │ edges │ best: total (swaps)  │ naive: total       │ best 2q depth      │ fidelity at 2q error    │");
    println!("│            │       │ mean [min,max]       │ mean               │ mean               │ 1e-3    5e-3    1e-2    │");
    println!("├────────────┼───────┼──────────────────────┼────────────────────┼────────────────────┼─────────────────────────┤");
    for r in &results {
        println!(
            "│ {:<10} │ {:>5} │ {:>6.1} ({:>5.1}) [{:>3},{:>3}] │ {:>18.1} │ {:>18.1} │ {:.3}   {:.3}   {:.3}   │",
            r.topology, r.edges, r.best.two_qubit_total.mean, r.best.swaps.mean, r.best.two_qubit_total.min, r.best.two_qubit_total.max,
            r.naive.two_qubit_total.mean, r.best.depth.mean, r.fidelity[0], r.fidelity[1], r.fidelity[2]
        );
    }
    println!("└────────────┴───────┴──────────────────────┴────────────────────┴────────────────────┴─────────────────────────┘");
    println!("\nPer topology, options for zinc_shot_noise at the typical device (--gate-error 0.005):");
    for r in &results {
        println!("  {:<11} --circuit-1q {:.1} --circuit-2q {:.1}", r.topology, r.best.single_qubit.mean, r.best.two_qubit_total.mean);
    }

    let json = serde_json::json!({ "depth": depth, "restarts": restarts, "search_molecules": search_n, "eval_molecules": eval_n, "results": results });
    let name = if data.label.starts_with("QM9") { "qm9_routing_study" } else { "zinc_routing_study" };
    let _ = t1000::results::write(format!("results/experiments/{name}.json"), serde_json::to_string_pretty(&json).unwrap_or_default());
}
