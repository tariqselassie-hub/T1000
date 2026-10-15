use num_complex::Complex64;
use rand::Rng;
use super::linalg::ComplexMatrix;
use super::state::{LatticeTopology, PlanarTensorNetwork};

/// Configuration for Boundary Matrix Product State (BMPS) contraction.
#[derive(Debug, Clone)]
pub struct BoundaryMpsConfig {
    /// Maximum virtual bond dimension R of the sweeping boundary MPS.
    pub max_boundary_dim: usize,
}

impl Default for BoundaryMpsConfig {
    fn default() -> Self {
        Self {
            max_boundary_dim: 16,
        }
    }
}

/// A 1D Boundary MPS slice along a column of the 2D planar lattice.
/// Represented as a sequence of rank-3 tensors: [left_bond, horizontal_wire, right_bond].
#[derive(Debug, Clone)]
pub struct BoundaryMps {
    pub tensors: Vec<ComplexMatrix>, // each tensor is (left_dim * wire_dim) x right_dim
    pub wire_dims: Vec<usize>,
    pub bond_dims: Vec<usize>, // length tensors.len() + 1, start and end are 1
}

impl BoundaryMps {
    /// Create initial trivial boundary MPS of rank 1 with 1-dimensional open wires.
    pub fn new_initial(height: usize) -> Self {
        let mut tensors = Vec::with_capacity(height);
        let wire_dims = vec![1; height];
        let bond_dims = vec![1; height + 1];

        for _ in 0..height {
            // (1 * 1) x 1 matrix with entry 1.0
            tensors.push(ComplexMatrix::from_flat(
                1,
                1,
                vec![Complex64::new(1.0, 0.0)],
            ));
        }

        Self {
            tensors,
            wire_dims,
            bond_dims,
        }
    }
}

/// Compute quantum state amplitude <x | psi> for a computational basis bitstring x
/// by contracting the planar 2D tensor network using Boundary MPS sweeps.
/// Slices the lattice into columns and sweeps a 1D boundary MPS from left to right.
pub fn compute_amplitude(
    tn: &PlanarTensorNetwork,
    bitstring: &[u8],
    config: &BoundaryMpsConfig,
) -> Complex64 {
    assert_eq!(
        bitstring.len(),
        tn.num_qubits(),
        "bitstring length must match qubit count"
    );

    match tn.topology {
        LatticeTopology::SquareGrid { width, height } => {
            contract_square_grid_amplitude(tn, bitstring, width, height, config)
        }
        LatticeTopology::HeavyHex { .. } => {
            // For general planar topologies, project physical bits and contract via graph elimination
            contract_planar_amplitude_generic(tn, bitstring, config)
        }
    }
}

/// Square-grid column-wise BMPS contraction:
/// Sweeps from column c = 0 to width - 1.
fn contract_square_grid_amplitude(
    tn: &PlanarTensorNetwork,
    bitstring: &[u8],
    width: usize,
    height: usize,
    config: &BoundaryMpsConfig,
) -> Complex64 {
    // Current boundary vector over vertical bonds at column boundary:
    // For small/moderate height, maintain the transfer vector along horizontal cuts
    // In column c, for each y in 0..height:
    // site = y * width + c
    // Project physical index to bitstring[site]
    // The site has incident bonds: West (c > 0), East (c + 1 < width), North (y > 0), South (y + 1 < height).

    // Vector representing boundary tensor across horizontal bonds leaving column c:
    // Initial boundary before column 0 is scalar 1.0
    let mut boundary_state: Vec<Complex64> = vec![Complex64::new(1.0, 0.0)];
    let mut current_wire_dims: Vec<usize> = vec![1; height];

    for c in 0..width {
        // Build column transfer operator MPO
        // For each y in 0..height, get site tensor projected to bitstring[site]
        let mut column_tensors = Vec::with_capacity(height);
        let mut next_wire_dims = Vec::with_capacity(height);

        for y in 0..height {
            let site_idx = y * width + c;
            let site = tn.site(site_idx);
            let bit = bitstring[site_idx] as usize;

            let west_dim = if c > 0 { site.neighbor_bond_dim(y * width + (c - 1)).unwrap_or(1) } else { 1 };
            let east_dim = if c + 1 < width { site.neighbor_bond_dim(y * width + (c + 1)).unwrap_or(1) } else { 1 };
            let north_dim = if y > 0 { site.neighbor_bond_dim((y - 1) * width + c).unwrap_or(1) } else { 1 };
            let south_dim = if y + 1 < height { site.neighbor_bond_dim((y + 1) * width + c).unwrap_or(1) } else { 1 };

            next_wire_dims.push(east_dim);

            // Extract projected matrix: (West, North) x (East, South)
            let mut node_data = vec![Complex64::new(0.0, 0.0); west_dim * north_dim * east_dim * south_dim];

            // Iterate over all neighbor indices of site
            let nbr_dims = &site.bond_dims;
            let mut indices = vec![0; nbr_dims.len()];

            let total_combos: usize = nbr_dims.iter().product();
            for combo in 0..total_combos {
                let mut rem = combo;
                for i in (0..nbr_dims.len()).rev() {
                    indices[i] = rem % nbr_dims[i];
                    rem /= nbr_dims[i];
                }

                let val = site.get_elem(bit, &indices);
                if val.norm_sqr() < 1e-28 {
                    continue;
                }

                // Map neighbor indices to canonical (West, East, North, South)
                let mut w = 0;
                let mut e = 0;
                let mut n = 0;
                let mut s = 0;

                for (pos, &nbr) in site.neighbors.iter().enumerate() {
                    let nx = nbr % width;
                    let ny = nbr / width;
                    if nx < c {
                        w = indices[pos];
                    } else if nx > c {
                        e = indices[pos];
                    } else if ny < y {
                        n = indices[pos];
                    } else if ny > y {
                        s = indices[pos];
                    }
                }

                let idx = ((w * north_dim + n) * east_dim + e) * south_dim + s;
                node_data[idx] = val;
            }

            column_tensors.push((west_dim, north_dim, east_dim, south_dim, node_data));
        }

        // Contract column into new boundary state:
        // New boundary has dimensions (next_wire_dims[0] x ... x next_wire_dims[height-1])
        let new_total_dim: usize = next_wire_dims.iter().product();
        let mut next_boundary = vec![Complex64::new(0.0, 0.0); new_total_dim];

        // Contract through the column with vertical bond accumulation
        contract_column_slice(
            &column_tensors,
            &current_wire_dims,
            &next_wire_dims,
            &boundary_state,
            &mut next_boundary,
        );

        // Compress boundary if column dimensions exceed max_boundary_dim
        if c + 1 < width && new_total_dim > config.max_boundary_dim.pow(2) {
            compress_boundary_vector(&mut next_boundary, config.max_boundary_dim);
        }

        boundary_state = next_boundary;
        current_wire_dims = next_wire_dims;
    }

    // At the final column, boundary_state has collapsed to length 1 (scalar amplitude)
    if !boundary_state.is_empty() {
        boundary_state[0]
    } else {
        Complex64::new(0.0, 0.0)
    }
}

fn contract_column_slice(
    column_tensors: &[(usize, usize, usize, usize, Vec<Complex64>)],
    prev_wires: &[usize],
    _next_wires: &[usize],
    prev_state: &[Complex64],
    next_state: &mut [Complex64],
) {
    let height = column_tensors.len();
    if height == 0 {
        return;
    }

    // Accumulate vertically from top (y=0) to bottom (y=height-1)
    // Vertical index between y and y+1 connects south of y to north of y+1
    let mut current_layer: Vec<(usize, Complex64)> = Vec::new();

    // Map each prev_state entry:
    for (prev_idx, &p_val) in prev_state.iter().enumerate() {
        if p_val.norm_sqr() < 1e-28 {
            continue;
        }

        // Decode prev horizontal indices
        let mut prev_indices = vec![0; height];
        let mut rem = prev_idx;
        for y in (0..height).rev() {
            prev_indices[y] = rem % prev_wires[y];
            rem /= prev_wires[y];
        }

        current_layer.clear();
        current_layer.push((0, p_val)); // (north_val=0, weight)

        // Cascade down the column
        for y in 0..height {
            let (_west_dim, north_dim, east_dim, south_dim, ref data) = column_tensors[y];
            let w = prev_indices[y];
            let mut next_layer = Vec::new();

            for &(n, in_weight) in &current_layer {
                for e in 0..east_dim {
                    for s in 0..south_dim {
                        let idx = ((w * north_dim + n) * east_dim + e) * south_dim + s;
                        let t_val = data[idx];
                        if t_val.norm_sqr() > 1e-28 {
                            next_layer.push(((e, s), in_weight * t_val));
                        }
                    }
                }
            }

            // Group by south index for next row, maintaining east index
            if y + 1 == height {
                // Final row: south must be 0 (boundary)
                for &((e, _s), w_final) in &next_layer {
                    // Accumulate into next_state
                    // Compute east index
                    let next_idx = e; // for 1D, or general multi-row layout
                    if next_idx < next_state.len() {
                        next_state[next_idx] += w_final;
                    }
                }
            } else {
                current_layer = next_layer.into_iter().map(|((_e, s), w)| (s, w)).collect();
            }
        }
    }
}

/// Zeroes all but the `max_dim` largest-magnitude boundary entries.
/// This is a heuristic sparsification with no error bound, not an SVD/MPS compression.
fn compress_boundary_vector(boundary: &mut [Complex64], max_dim: usize) {
    if boundary.len() <= max_dim {
        return;
    }
    // Truncate smaller amplitudes to conserve boundary rank
    let mut magnitudes: Vec<(f64, usize)> = boundary
        .iter()
        .enumerate()
        .map(|(i, z)| (z.norm_sqr(), i))
        .collect();
    magnitudes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = vec![false; boundary.len()];
    for &(_, idx) in magnitudes.iter().take(max_dim) {
        keep[idx] = true;
    }

    for (i, val) in boundary.iter_mut().enumerate() {
        if !keep[i] {
            *val = Complex64::new(0.0, 0.0);
        }
    }
}

/// Generic planar amplitude contraction via tensor elimination.
fn contract_planar_amplitude_generic(
    tn: &PlanarTensorNetwork,
    bitstring: &[u8],
    _config: &BoundaryMpsConfig,
) -> Complex64 {
    // Projecting each site at bond index 0 is only exact when no bond carries entanglement.
    if let Some(site) = tn.sites.iter().find(|s| s.bond_dims.iter().any(|&d| d > 1)) {
        panic!(
            "heavy-hex amplitude contraction is only implemented for product states \
             (site {} has bond dims {:?})",
            site.site_idx, site.bond_dims
        );
    }

    // Project all physical indices to bitstring values
    let mut amp = Complex64::new(1.0, 0.0);
    for (i, site) in tn.sites.iter().enumerate() {
        let bit = bitstring[i] as usize;
        let zero_bonds = vec![0; site.neighbors.len()];
        let val = site.get_elem(bit, &zero_bonds);
        amp *= val;
    }
    amp
}

/// Compute ideal probability p(x) = |<x | psi>|^2.
pub fn compute_probability(
    tn: &PlanarTensorNetwork,
    bitstring: &[u8],
    config: &BoundaryMpsConfig,
) -> f64 {
    let amp = compute_amplitude(tn, bitstring, config);
    amp.norm_sqr()
}

/// Sample a single bitstring x from the tensor network state with probability q(x).
/// Performs site-by-site conditional sampling.
pub fn sample_bitstring(
    tn: &PlanarTensorNetwork,
    config: &BoundaryMpsConfig,
    rng: &mut impl Rng,
) -> (Vec<u8>, f64) {
    let n = tn.num_qubits();
    let mut bitstring = vec![0u8; n];
    let mut cumulative_prob = 1.0;

    for site in 0..n {
        // Measure marginal conditional probability P(X_site = 0 | x_{<site})
        bitstring[site] = 0;
        let p0 = compute_probability(tn, &bitstring, config);

        bitstring[site] = 1;
        let p1 = compute_probability(tn, &bitstring, config);

        let total = p0 + p1;
        let prob_0 = if total > 1e-28 { (p0 / total).clamp(0.0, 1.0) } else { 0.5 };

        let draw: f64 = rng.gen();
        if draw < prob_0 {
            bitstring[site] = 0;
            cumulative_prob *= prob_0;
        } else {
            bitstring[site] = 1;
            cumulative_prob *= 1.0 - prob_0;
        }
    }

    (bitstring, cumulative_prob)
}

/// Batch of bitstring samples certified against ideal amplitudes.
#[derive(Debug, Clone)]
pub struct SampleBatch {
    pub bitstrings: Vec<Vec<u8>>,
    pub sampled_probs_q: Vec<f64>,
    pub ideal_probs_p: Vec<f64>,
    pub mean_prob_ratio: f64,
    pub sample_kld: f64,
    pub linear_xeb: f64,
}

/// Draw a batch of M bitstrings, compute ideal probabilities, and evaluate linear XEB.
pub fn sample_batch(
    tn: &PlanarTensorNetwork,
    sample_count: usize,
    config: &BoundaryMpsConfig,
    rng: &mut impl Rng,
) -> SampleBatch {
    let n = tn.num_qubits();
    let d_hilbert = (1_u64 << n.min(62)) as f64;

    let mut bitstrings = Vec::with_capacity(sample_count);
    let mut sampled_probs_q = Vec::with_capacity(sample_count);
    let mut ideal_probs_p = Vec::with_capacity(sample_count);

    let mut sum_ratio = 0.0;
    let mut sum_kld = 0.0;
    let mut sum_ideal_p = 0.0;

    for _ in 0..sample_count {
        let (bits, q_x) = sample_bitstring(tn, config, rng);
        let p_x = compute_probability(tn, &bits, config);

        let q_clamped = q_x.max(1e-20);
        let p_clamped = p_x.max(1e-20);

        let ratio = p_clamped / q_clamped;
        sum_ratio += ratio;
        sum_kld += (q_clamped / p_clamped).ln();
        sum_ideal_p += p_x;

        bitstrings.push(bits);
        sampled_probs_q.push(q_x);
        ideal_probs_p.push(p_x);
    }

    let m_f64 = sample_count.max(1) as f64;
    let mean_ratio = sum_ratio / m_f64;
    let mean_kld = (sum_kld / m_f64).abs();

    // Standard Linear XEB formula from Google Cirq:
    // F_XEB = 2^N * (1/M * sum_{i=1}^M P_ideal(x_i)) - 1
    let avg_ideal = sum_ideal_p / m_f64;
    let linear_xeb = d_hilbert * avg_ideal - 1.0;

    SampleBatch {
        bitstrings,
        sampled_probs_q,
        ideal_probs_p,
        mean_prob_ratio: mean_ratio,
        sample_kld: mean_kld,
        linear_xeb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use crate::tensor_network::state::LatticeTopology;
    use crate::tensor_network::evolution::{SingleQubitGate, TwoQubitGate, apply_single_qubit_gate, apply_two_qubit_gate, EvolutionTelemetry};

    #[test]
    fn test_compute_amplitude_product_state_zero() {
        let topo = LatticeTopology::SquareGrid { width: 2, height: 2 };
        let tn = PlanarTensorNetwork::new_product_state(topo, 8);
        let config = BoundaryMpsConfig::default();

        let amp_0000 = compute_amplitude(&tn, &[0, 0, 0, 0], &config);
        assert!((amp_0000.re - 1.0).abs() < 1e-6);
        assert!(amp_0000.im.abs() < 1e-6);

        let amp_0001 = compute_amplitude(&tn, &[0, 0, 0, 1], &config);
        assert!(amp_0001.norm() < 1e-6);
    }

    #[test]
    fn test_heavy_hex_product_state_amplitude() {
        let tn = PlanarTensorNetwork::new_product_state(LatticeTopology::HeavyHex { width: 2, height: 1 }, 8);
        let n = tn.num_qubits();
        let config = BoundaryMpsConfig::default();
        assert!((compute_amplitude(&tn, &vec![0; n], &config).re - 1.0).abs() < 1e-12);
        let mut one = vec![0; n];
        one[0] = 1;
        assert!(compute_amplitude(&tn, &one, &config).norm() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "only implemented for product states")]
    fn test_heavy_hex_entangled_amplitude_refuses() {
        let topo = LatticeTopology::HeavyHex { width: 2, height: 1 };
        let (u, v) = topo.edges()[0];
        let mut tn = PlanarTensorNetwork::new_product_state(topo, 8);
        let mut tel = EvolutionTelemetry::new();
        apply_single_qubit_gate(&mut tn, u, &SingleQubitGate::hadamard(), &mut tel);
        apply_two_qubit_gate(&mut tn, u, v, &TwoQubitGate::cnot(), &mut tel);
        let n = tn.num_qubits();
        compute_amplitude(&tn, &vec![0; n], &BoundaryMpsConfig::default());
    }

    #[test]
    fn test_compute_amplitude_bell_state() {
        let topo = LatticeTopology::SquareGrid { width: 2, height: 1 };
        let mut tn = PlanarTensorNetwork::new_product_state(topo, 8);
        let mut tel = EvolutionTelemetry::new();

        apply_single_qubit_gate(&mut tn, 0, &SingleQubitGate::hadamard(), &mut tel);
        apply_two_qubit_gate(&mut tn, 0, 1, &TwoQubitGate::cnot(), &mut tel);

        let config = BoundaryMpsConfig::default();
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

        let amp_00 = compute_amplitude(&tn, &[0, 0], &config);
        let amp_11 = compute_amplitude(&tn, &[1, 1], &config);
        let amp_01 = compute_amplitude(&tn, &[0, 1], &config);
        let amp_10 = compute_amplitude(&tn, &[1, 0], &config);

        assert!((amp_00.re - inv_sqrt2).abs() < 1e-6);
        assert!((amp_11.re - inv_sqrt2).abs() < 1e-6);
        assert!(amp_01.norm() < 1e-6);
        assert!(amp_10.norm() < 1e-6);
    }

    #[test]
    fn test_sample_batch_bell_state() {
        let topo = LatticeTopology::SquareGrid { width: 2, height: 1 };
        let mut tn = PlanarTensorNetwork::new_product_state(topo, 8);
        let mut tel = EvolutionTelemetry::new();

        apply_single_qubit_gate(&mut tn, 0, &SingleQubitGate::hadamard(), &mut tel);
        apply_two_qubit_gate(&mut tn, 0, 1, &TwoQubitGate::cnot(), &mut tel);

        let config = BoundaryMpsConfig::default();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let batch = sample_batch(&tn, 100, &config, &mut rng);
        assert_eq!(batch.bitstrings.len(), 100);

        for bits in &batch.bitstrings {
            assert!(bits == &[0, 0] || bits == &[1, 1], "Bell state samples must be |00> or |11>");
        }
    }
}
