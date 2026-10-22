// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Routing a circuit onto a device's qubit connectivity.
//!
//! A [`Topology`] lists which physical qubits can interact. [`route`] takes a program over
//! logical qubits and an initial placement, inserts SWAPs so every two-qubit gate acts on
//! neighbours, and returns the physical circuit. Two simplifications keep it small and honest:
//!
//! * the final qubit permutation is **not** undone: the readout relabels the measured bits, so
//!   no restoring SWAPs are needed;
//! * a [`Item::CzSet`] is a set of mutually commuting CZ gates (all CZs of one ansatz layer), which
//!   the router may execute in any order.
//!
//! The router is a greedy one (it picks the SWAP that most reduces the summed distance of the
//! pending gates), so gate counts are an upper bound on what an optimising compiler achieves; the
//! ranking between topologies is the meaningful output. [`search_layout`] looks for the initial
//! placement that minimises the routed cost. Logical and physical qubit counts must be equal.

use crate::compiler::CircuitBuilder;
use crate::state::QuantumState;
use crate::state_prep::PrepGate;

/// A gate of a routable program. Qubit indices are logical in a program and physical in a routed
/// circuit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RGate {
    Ry { qubit: usize, theta: f64 },
    Rz { qubit: usize, theta: f64 },
    Cnot { control: usize, target: usize },
    Cz { a: usize, b: usize },
    Swap { a: usize, b: usize },
}

/// One step of a program.
#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    /// A gate that must keep its place in the order.
    Gate(RGate),
    /// CZ gates that commute with each other and may run in any order.
    CzSet(Vec<(usize, usize)>),
}

/// Qubit connectivity.
#[derive(Debug, Clone)]
pub struct Topology {
    pub name: String,
    pub n: usize,
    edges: Vec<(usize, usize)>,
    dist: Vec<Vec<usize>>,
}

impl Topology {
    pub fn new(name: &str, n: usize, edges: &[(usize, usize)]) -> Self {
        let mut dist = vec![vec![usize::MAX; n]; n];
        let mut adj = vec![Vec::new(); n];
        for &(a, b) in edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        for (s, from_s) in dist.iter_mut().enumerate() {
            from_s[s] = 0;
            let mut queue = std::collections::VecDeque::from([s]);
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if from_s[v] == usize::MAX {
                        from_s[v] = from_s[u] + 1;
                        queue.push_back(v);
                    }
                }
            }
        }
        assert!(dist.iter().flatten().all(|&d| d != usize::MAX), "topology {name} is not connected");
        Self { name: name.to_string(), n, edges: edges.to_vec(), dist }
    }

    pub fn all_to_all(n: usize) -> Self {
        let edges: Vec<_> = (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
        Self::new("all-to-all", n, &edges)
    }

    pub fn line(n: usize) -> Self {
        let edges: Vec<_> = (0..n - 1).map(|a| (a, a + 1)).collect();
        Self::new("line", n, &edges)
    }

    pub fn ring(n: usize) -> Self {
        let edges: Vec<_> = (0..n).map(|a| (a, (a + 1) % n)).collect();
        Self::new("ring", n, &edges)
    }

    /// `rows x cols` square lattice, qubit `r * cols + c`.
    pub fn grid(rows: usize, cols: usize) -> Self {
        let mut edges = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    edges.push((r * cols + c, r * cols + c + 1));
                }
                if r + 1 < rows {
                    edges.push((r * cols + c, (r + 1) * cols + c));
                }
            }
        }
        Self::new("grid", rows * cols, &edges)
    }

    /// A 10-qubit tree cut from the heavy-hex lattice around two adjacent heavy hexagons: a path of
    /// five qubits with a branch of two on the second qubit and a branch of three on the fourth.
    /// Both branch points have degree 3, as in heavy-hex. Real devices are larger and contain
    /// cycles, so this fragment is a conservative choice.
    pub fn heavy_hex_fragment() -> Self {
        // path 0-1-2-3-4, branch 1-5-6, branch 3-7-8-9
        Self::new("heavy-hex", 10, &[(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (3, 7), (7, 8), (8, 9)])
    }

    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        self.dist[a][b] == 1
    }

    pub fn distance(&self, a: usize, b: usize) -> usize {
        self.dist[a][b]
    }

    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }
}

/// A routed circuit on physical qubits.
#[derive(Debug, Clone)]
pub struct Routed {
    pub gates: Vec<RGate>,
    /// `final_map[logical]` is the physical qubit holding that logical qubit at the end.
    pub final_map: Vec<usize>,
}

/// Gate counts and two-qubit depth of a circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cost {
    pub single: usize,
    pub native_two: usize,
    pub swaps: usize,
    /// Depth in two-qubit layers, counting a SWAP as three.
    pub depth: usize,
}

impl Cost {
    /// Two-qubit gates with each SWAP counted as three CNOTs.
    pub fn two_qubit_total(&self) -> usize {
        self.native_two + 3 * self.swaps
    }
}

pub fn cost(gates: &[RGate]) -> Cost {
    let mut c = Cost::default();
    let mut busy = std::collections::HashMap::<usize, usize>::new();
    for g in gates {
        let (a, b, layers) = match *g {
            RGate::Ry { .. } | RGate::Rz { .. } => {
                c.single += 1;
                continue;
            }
            RGate::Cnot { control, target } => {
                c.native_two += 1;
                (control, target, 1)
            }
            RGate::Cz { a, b } => {
                c.native_two += 1;
                (a, b, 1)
            }
            RGate::Swap { a, b } => {
                c.swaps += 1;
                (a, b, 3)
            }
        };
        let start = busy.get(&a).copied().unwrap_or(0).max(busy.get(&b).copied().unwrap_or(0));
        busy.insert(a, start + layers);
        busy.insert(b, start + layers);
        c.depth = c.depth.max(start + layers);
    }
    c
}

struct Mapping {
    phys: Vec<usize>,
    logical: Vec<usize>,
}

impl Mapping {
    fn new(layout: &[usize]) -> Self {
        let mut logical = vec![usize::MAX; layout.len()];
        for (l, &p) in layout.iter().enumerate() {
            assert!(logical[p] == usize::MAX, "layout must be a permutation");
            logical[p] = l;
        }
        Self { phys: layout.to_vec(), logical }
    }

    fn swap(&mut self, p: usize, q: usize) {
        let (lp, lq) = (self.logical[p], self.logical[q]);
        self.logical.swap(p, q);
        self.phys[lp] = q;
        self.phys[lq] = p;
    }
}

/// Weight of the next layer's gates when choosing among SWAPs that all shorten the current layer.
const LOOKAHEAD_WEIGHT: f64 = 0.4;

/// Executes `pairs` (all pending at once) on neighbours, inserting SWAPs, emitting `make(pa, pb)`.
///
/// A SWAP is only taken if it reduces the summed distance of the pending pairs (so the loop always
/// terminates); among those, the one that also leaves the `lookahead` pairs closest wins.
fn route_pairs(
    topo: &Topology,
    map: &mut Mapping,
    mut pairs: Vec<(usize, usize)>,
    lookahead: &[(usize, usize)],
    make: impl Fn(usize, usize) -> RGate,
    out: &mut Vec<RGate>,
) {
    loop {
        pairs.retain(|&(a, b)| {
            let (pa, pb) = (map.phys[a], map.phys[b]);
            if topo.adjacent(pa, pb) {
                out.push(make(pa, pb));
                false
            } else {
                true
            }
        });
        if pairs.is_empty() {
            return;
        }
        let total = |m: &Mapping| -> usize { pairs.iter().map(|&(a, b)| topo.distance(m.phys[a], m.phys[b])).sum() };
        let current = total(map);
        let ahead = |m: &Mapping| -> f64 { lookahead.iter().map(|&(a, b)| topo.distance(m.phys[a], m.phys[b]) as f64).sum() };
        let mut best: Option<((usize, usize), f64)> = None;
        for &(p, q) in topo.edges() {
            let involved = |x: usize| pairs.iter().any(|&(a, b)| map.logical[x] == a || map.logical[x] == b);
            if !(involved(p) || involved(q)) {
                continue;
            }
            map.swap(p, q);
            let (primary, secondary) = (total(map), ahead(map));
            map.swap(p, q);
            if primary < current {
                let score = primary as f64 + LOOKAHEAD_WEIGHT * secondary;
                if best.is_none_or(|(_, s)| score < s) {
                    best = Some(((p, q), score));
                }
            }
        }
        match best {
            Some(((p, q), _)) => {
                map.swap(p, q);
                out.push(RGate::Swap { a: p, b: q });
            }
            _ => {
                // No swap reduces the total: bring the first pair together along a shortest path.
                let (a, b) = pairs[0];
                while !topo.adjacent(map.phys[a], map.phys[b]) {
                    let pa = map.phys[a];
                    let next = topo
                        .edges()
                        .iter()
                        .filter_map(|&(x, y)| if x == pa { Some(y) } else if y == pa { Some(x) } else { None })
                        .min_by_key(|&nb| topo.distance(nb, map.phys[b]))
                        .expect("connected topology");
                    map.swap(pa, next);
                    out.push(RGate::Swap { a: pa, b: next });
                }
            }
        }
    }
}

/// Routes `program` onto `topo` from the placement `layout` (`layout[logical]` = physical).
pub fn route(program: &[Item], topo: &Topology, layout: &[usize]) -> Routed {
    assert_eq!(layout.len(), topo.n, "logical and physical qubit counts must match");
    let mut map = Mapping::new(layout);
    let mut gates = Vec::new();
    for (index, item) in program.iter().enumerate() {
        // The gates of the next two-qubit item, used to break ties between equally good SWAPs.
        let ahead: Vec<(usize, usize)> = program[index + 1..]
            .iter()
            .find_map(|next| match next {
                Item::CzSet(p) => Some(p.clone()),
                Item::Gate(RGate::Cnot { control, target }) => Some(vec![(*control, *target)]),
                Item::Gate(RGate::Cz { a, b }) => Some(vec![(*a, *b)]),
                _ => None,
            })
            .unwrap_or_default();
        match item {
            Item::CzSet(pairs) => route_pairs(topo, &mut map, pairs.clone(), &ahead, |a, b| RGate::Cz { a, b }, &mut gates),
            Item::Gate(RGate::Ry { qubit, theta }) => gates.push(RGate::Ry { qubit: map.phys[*qubit], theta: *theta }),
            Item::Gate(RGate::Rz { qubit, theta }) => gates.push(RGate::Rz { qubit: map.phys[*qubit], theta: *theta }),
            Item::Gate(RGate::Cnot { control, target }) => {
                let (c, t) = (*control, *target);
                // Keep the control/target roles: route the pair, then emit with current positions.
                let mut scratch = Vec::new();
                route_pairs(topo, &mut map, vec![(c, t)], &ahead, |a, b| RGate::Cz { a, b }, &mut scratch);
                let swaps_only = scratch.iter().filter(|g| matches!(g, RGate::Swap { .. }));
                gates.extend(swaps_only);
                gates.push(RGate::Cnot { control: map.phys[c], target: map.phys[t] });
            }
            Item::Gate(RGate::Cz { a, b }) => route_pairs(topo, &mut map, vec![(*a, *b)], &ahead, |x, y| RGate::Cz { a: x, b: y }, &mut gates),
            Item::Gate(RGate::Swap { a, b }) => {
                gates.push(RGate::Swap { a: map.phys[*a], b: map.phys[*b] });
                let (pa, pb) = (map.phys[*a], map.phys[*b]);
                map.swap(pa, pb);
            }
        }
    }
    Routed { gates, final_map: map.phys }
}

/// The state-preparation gates as program items (they act on the lowest logical qubits).
pub fn prep_program(gates: &[PrepGate]) -> Vec<Item> {
    gates
        .iter()
        .map(|g| match *g {
            PrepGate::Ry { qubit, theta } => Item::Gate(RGate::Ry { qubit, theta }),
            PrepGate::Cnot { control, target } => Item::Gate(RGate::Cnot { control, target }),
        })
        .collect()
}

/// The hardware-efficient ansatz as a program: per layer `n` Ry, `n` Rz and the layer's CZ set
/// (`(q, q + 2^(layer mod 4)) mod n`, as in `CircuitAnsatz`). Angles are fixed pseudo-random
/// values: routing cost does not depend on them. The Rademacher phase layer is a diagonal sign
/// pattern implementable as virtual Z rotations, and is not counted.
pub fn ansatz_program(n: usize, depth: usize) -> Vec<Item> {
    let mut rng = Rng(0xA5A5_0000 ^ (n as u64) << 8 ^ depth as u64);
    let mut angle = || (rng.next() % 6283) as f64 / 1000.0;
    let mut items = Vec::new();
    for layer in 0..depth {
        for qubit in 0..n {
            items.push(Item::Gate(RGate::Ry { qubit, theta: angle() }));
        }
        for qubit in 0..n {
            items.push(Item::Gate(RGate::Rz { qubit, theta: angle() }));
        }
        items.push(Item::CzSet(crate::ansatz::entangling_pairs(n, layer).collect()));
    }
    items
}

/// Small deterministic generator for the layout search.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            v.swap(i, (self.next() % (i as u64 + 1)) as usize);
        }
    }
}

/// Mean two-qubit cost of routing every program in `programs` from `layout`.
pub fn mean_cost(programs: &[Vec<Item>], topo: &Topology, layout: &[usize]) -> f64 {
    let total: usize = programs.iter().map(|p| cost(&route(p, topo, layout).gates).two_qubit_total()).sum();
    total as f64 / programs.len() as f64
}

/// Looks for the placement with the lowest mean routed cost over `programs`: the identity layout
/// and `restarts` random layouts, each improved by pairwise exchanges until none helps.
/// Returns the layout and its mean two-qubit cost; never worse than the identity layout.
pub fn search_layout(programs: &[Vec<Item>], topo: &Topology, restarts: usize, seed: u64) -> (Vec<usize>, f64) {
    let n = topo.n;
    let mut rng = Rng(seed);
    let identity: Vec<usize> = (0..n).collect();
    let mut best = (identity.clone(), mean_cost(programs, topo, &identity));
    for r in 0..=restarts {
        let mut layout = identity.clone();
        if r > 0 {
            rng.shuffle(&mut layout);
        }
        let mut here = mean_cost(programs, topo, &layout);
        loop {
            let mut improved = None;
            for i in 0..n {
                for j in i + 1..n {
                    layout.swap(i, j);
                    let c = mean_cost(programs, topo, &layout);
                    layout.swap(i, j);
                    if c < here - 1e-12 && improved.is_none_or(|(_, _, ic)| c < ic) {
                        improved = Some((i, j, c));
                    }
                }
            }
            match improved {
                Some((i, j, c)) => {
                    layout.swap(i, j);
                    here = c;
                }
                None => break,
            }
        }
        if here < best.1 - 1e-12 {
            best = (layout, here);
        }
    }
    best
}

/// Simulates a gate list on `n` qubits from |0...0>.
pub fn simulate(gates: &[RGate], n: usize) -> QuantumState {
    let mut b = CircuitBuilder::new(n);
    for g in gates {
        match *g {
            RGate::Ry { qubit, theta } => b.ry(qubit, theta),
            RGate::Rz { qubit, theta } => b.rz(qubit, theta),
            RGate::Cnot { control, target } => b.cnot(control, target),
            RGate::Cz { a, b: q } => b.cz(a, q),
            RGate::Swap { a, b: q } => b.swap(a, q),
        };
    }
    let mut state = QuantumState::ground(1 << n);
    b.compile().execute(&mut state);
    state
}

/// The gates of a program in order (a `CzSet` in the order listed), for simulation.
pub fn flatten(program: &[Item]) -> Vec<RGate> {
    program
        .iter()
        .flat_map(|item| match item {
            Item::Gate(g) => vec![*g],
            Item::CzSet(pairs) => pairs.iter().map(|&(a, b)| RGate::Cz { a, b }).collect(),
        })
        .collect()
}

/// Largest amplitude difference between the program run on logical qubits and the routed circuit
/// run on physical qubits, after undoing the final qubit permutation.
pub fn verify(program: &[Item], routed: &Routed, n: usize) -> f64 {
    let logical = simulate(&flatten(program), n);
    let physical = simulate(&routed.gates, n);
    let mut worst = 0.0f64;
    for x in 0..1usize << n {
        // Logical basis state x lives at the physical index with bit final_map[l] = bit l of x.
        let y = (0..n).fold(0usize, |acc, l| acc | (((x >> l) & 1) << routed.final_map[l]));
        worst = worst
            .max((logical.real[x] - physical.real[y]).abs())
            .max((logical.imag[x] - physical.imag[y]).abs());
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_program(n: usize, seed: u64) -> Vec<Item> {
        let mut rng = Rng(seed);
        let mut angle = || (rng.next() % 6283) as f64 / 1000.0;
        let mut items = Vec::new();
        for q in 0..n {
            items.push(Item::Gate(RGate::Ry { qubit: q, theta: angle() }));
        }
        for round in 0..3 {
            let stride = 1 + round;
            items.push(Item::CzSet((0..n).map(|q| (q, (q + stride) % n)).filter(|&(a, b)| a != b).collect()));
            for q in 0..n {
                items.push(Item::Gate(RGate::Ry { qubit: q, theta: angle() }));
                items.push(Item::Gate(RGate::Rz { qubit: q, theta: angle() }));
            }
            items.push(Item::Gate(RGate::Cnot { control: round, target: n - 1 - round }));
            items.push(Item::Gate(RGate::Cnot { control: n - 1, target: round + 1 }));
        }
        items
    }

    fn topologies() -> Vec<Topology> {
        vec![Topology::all_to_all(6), Topology::line(6), Topology::ring(6), Topology::grid(2, 3)]
    }

    #[test]
    fn distances_match_the_geometry() {
        let line = Topology::line(5);
        assert_eq!(line.distance(0, 4), 4);
        let ring = Topology::ring(10);
        assert_eq!((ring.distance(0, 5), ring.distance(0, 9)), (5, 1));
        let grid = Topology::grid(2, 5);
        assert_eq!(grid.distance(0, 9), 5);
        let hh = Topology::heavy_hex_fragment();
        assert_eq!((hh.n, hh.edges().len()), (10, 9));
        assert!(Topology::all_to_all(7).distance(2, 6) == 1);
    }

    #[test]
    fn routed_circuits_equal_the_original_up_to_the_final_permutation() {
        for topo in topologies() {
            for seed in 1..=4 {
                let program = random_program(6, seed);
                let mut layout: Vec<usize> = (0..6).collect();
                Rng(seed * 31).shuffle(&mut layout);
                let routed = route(&program, &topo, &layout);
                let dev = verify(&program, &routed, 6);
                assert!(dev < 1e-10, "{} seed {seed}: deviation {dev}", topo.name);
            }
        }
    }

    #[test]
    fn every_routed_two_qubit_gate_acts_on_a_device_edge() {
        for topo in topologies() {
            let routed = route(&random_program(6, 9), &topo, &(0..6).collect::<Vec<_>>());
            for g in &routed.gates {
                let pair = match *g {
                    RGate::Cnot { control, target } => Some((control, target)),
                    RGate::Cz { a, b } | RGate::Swap { a, b } => Some((a, b)),
                    _ => None,
                };
                if let Some((a, b)) = pair {
                    assert!(topo.adjacent(a, b), "{}: gate on non-neighbours {a},{b}", topo.name);
                }
            }
        }
    }

    #[test]
    fn all_to_all_needs_no_swaps_and_a_matching_ring_layout_none_either() {
        let program = random_program(6, 5);
        let free = route(&program, &Topology::all_to_all(6), &(0..6).collect::<Vec<_>>());
        assert_eq!(cost(&free.gates).swaps, 0);
        // One layer of nearest-neighbour CZs on a ring placed identically needs no swaps.
        let ring_layer = vec![Item::CzSet((0..6).map(|q| (q, (q + 1) % 6)).collect())];
        let routed = route(&ring_layer, &Topology::ring(6), &(0..6).collect::<Vec<_>>());
        assert_eq!(cost(&routed.gates).swaps, 0);
    }

    #[test]
    fn the_layout_search_is_never_worse_than_the_identity_layout() {
        let programs = vec![random_program(6, 3), random_program(6, 4)];
        for topo in topologies() {
            let identity = mean_cost(&programs, &topo, &(0..6).collect::<Vec<_>>());
            let (layout, best) = search_layout(&programs, &topo, 12, 7);
            assert!(best <= identity + 1e-9, "{}: {best} > {identity}", topo.name);
            assert!((mean_cost(&programs, &topo, &layout) - best).abs() < 1e-9);
        }
    }

    #[test]
    fn ansatz_program_has_the_expected_gates_and_strides() {
        let program = ansatz_program(10, 4);
        let sets: Vec<&Vec<(usize, usize)>> = program
            .iter()
            .filter_map(|i| if let Item::CzSet(p) = i { Some(p) } else { None })
            .collect();
        assert_eq!(sets.len(), 4);
        assert!(sets.iter().all(|s| s.len() == 10));
        // Strides 1, 2, 4 and 8 (mod 10): pairs (q, q + stride).
        assert_eq!(sets[0][0], (0, 1));
        assert_eq!(sets[1][0], (0, 2));
        assert_eq!(sets[2][0], (0, 4));
        assert_eq!(sets[3][0], (0, 8));
        let singles = program.iter().filter(|i| matches!(i, Item::Gate(_))).count();
        assert_eq!(singles, 4 * 20);
    }

    #[test]
    fn a_routed_ansatz_plus_preparation_matches_the_original_on_ten_qubits() {
        let prep = crate::state_prep::amplitude_prep(&[0.3, 0.1, 0.5, 0.2, 0.05, 0.4, 0.3, 0.1, 0.2, 0.25, 0.15, 0.3, 0.1, 0.35, 0.2, 0.05, 0.4, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut program = prep_program(&prep);
        program.extend(ansatz_program(10, 2));
        for topo in [Topology::line(10), Topology::grid(2, 5), Topology::heavy_hex_fragment()] {
            let (layout, _) = search_layout(std::slice::from_ref(&program), &topo, 2, 3);
            let routed = route(&program, &topo, &layout);
            assert!(verify(&program, &routed, 10) < 1e-9, "{}", topo.name);
        }
    }

    #[test]
    fn cost_counts_swaps_as_three_and_tracks_depth() {
        let gates = [
            RGate::Ry { qubit: 0, theta: 0.1 },
            RGate::Cz { a: 0, b: 1 },
            RGate::Swap { a: 1, b: 2 },
            RGate::Cnot { control: 2, target: 3 },
        ];
        let c = cost(&gates);
        assert_eq!((c.single, c.native_two, c.swaps), (1, 2, 1));
        assert_eq!(c.two_qubit_total(), 5);
        assert_eq!(c.depth, 1 + 3 + 1);
    }
}
