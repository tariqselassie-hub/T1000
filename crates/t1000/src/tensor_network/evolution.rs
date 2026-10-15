use num_complex::Complex64;
use super::linalg::{ComplexMatrix, svd};
use super::state::PlanarTensorNetwork;

/// Gate error and circuit fidelity telemetry.
#[derive(Debug, Clone, Default)]
pub struct EvolutionTelemetry {
    pub single_qubit_gates: usize,
    pub two_qubit_gates: usize,
    pub cumulative_fidelity: f64,
    pub max_bond_dim_reached: usize,
    pub total_truncation_error: f64,
}

impl EvolutionTelemetry {
    pub fn new() -> Self {
        Self {
            single_qubit_gates: 0,
            two_qubit_gates: 0,
            cumulative_fidelity: 1.0,
            max_bond_dim_reached: 1,
            total_truncation_error: 0.0,
        }
    }
}

/// Common single-qubit quantum gates.
pub struct SingleQubitGate;

impl SingleQubitGate {
    pub fn pauli_x() -> ComplexMatrix {
        ComplexMatrix::from_flat(
            2,
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
    }

    pub fn pauli_y() -> ComplexMatrix {
        ComplexMatrix::from_flat(
            2,
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        )
    }

    pub fn pauli_z() -> ComplexMatrix {
        ComplexMatrix::from_flat(
            2,
            2,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        )
    }

    pub fn hadamard() -> ComplexMatrix {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        ComplexMatrix::from_flat(
            2,
            2,
            vec![
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        )
    }

    pub fn rx(theta: f64) -> ComplexMatrix {
        let c = (theta * 0.5).cos();
        let s = (theta * 0.5).sin();
        ComplexMatrix::from_flat(
            2,
            2,
            vec![
                Complex64::new(c, 0.0),
                Complex64::new(0.0, -s),
                Complex64::new(0.0, -s),
                Complex64::new(c, 0.0),
            ],
        )
    }

    pub fn rz(theta: f64) -> ComplexMatrix {
        let half = theta * 0.5;
        ComplexMatrix::from_flat(
            2,
            2,
            vec![
                Complex64::from_polar(1.0, -half),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::from_polar(1.0, half),
            ],
        )
    }

    /// Google Cirq PhasedXZGate(axis_phase, x_exponent, z_exponent).
    pub fn phased_xz(axis_phase_rad: f64, x_rad: f64, z_rad: f64) -> ComplexMatrix {
        let rz_pre = Self::rz(-axis_phase_rad);
        let rx_op = Self::rx(x_rad);
        let rz_post = Self::rz(axis_phase_rad + z_rad);
        rz_post.matmul(&rx_op).matmul(&rz_pre)
    }
}

/// Common two-qubit quantum gates (4x4 matrices in row-major computational basis |00>, |01>, |10>, |11>).
pub struct TwoQubitGate;

impl TwoQubitGate {
    pub fn cz() -> ComplexMatrix {
        let mut m = ComplexMatrix::identity(4);
        m[(3, 3)] = Complex64::new(-1.0, 0.0);
        m
    }

    pub fn cnot() -> ComplexMatrix {
        let mut m = ComplexMatrix::identity(4);
        m[(2, 2)] = Complex64::new(0.0, 0.0);
        m[(2, 3)] = Complex64::new(1.0, 0.0);
        m[(3, 3)] = Complex64::new(0.0, 0.0);
        m[(3, 2)] = Complex64::new(1.0, 0.0);
        m
    }

    /// Square root of iSWAP (Google Sycamore entangling gate).
    pub fn sqrt_iswap() -> ComplexMatrix {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let mut m = ComplexMatrix::identity(4);
        m[(1, 1)] = Complex64::new(inv_sqrt2, 0.0);
        m[(1, 2)] = Complex64::new(0.0, inv_sqrt2);
        m[(2, 1)] = Complex64::new(0.0, inv_sqrt2);
        m[(2, 2)] = Complex64::new(inv_sqrt2, 0.0);
        m
    }
}

/// Apply a single-qubit gate G in-place to site `target_site`.
/// Complexity: O(chi^d), exact with zero truncation.
pub fn apply_single_qubit_gate(
    tn: &mut PlanarTensorNetwork,
    target_site: usize,
    gate: &ComplexMatrix,
    telemetry: &mut EvolutionTelemetry,
) {
    assert_eq!(gate.rows, 2);
    assert_eq!(gate.cols, 2);

    let site = tn.site_mut(target_site);
    let phys_stride = site.strides[0];
    let num_virtual_states = site.data.len() / 2;

    for v_idx in 0..num_virtual_states {
        let psi0 = site.data[v_idx];
        let psi1 = site.data[v_idx + phys_stride];

        let new_psi0 = gate[(0, 0)] * psi0 + gate[(0, 1)] * psi1;
        let new_psi1 = gate[(1, 0)] * psi0 + gate[(1, 1)] * psi1;

        site.data[v_idx] = new_psi0;
        site.data[v_idx + phys_stride] = new_psi1;
    }

    telemetry.single_qubit_gates += 1;
}

/// Apply a two-qubit gate G to adjacent sites `(u, v)` with SVD bond truncation.
/// Truncates the bond between u and v to at most `max_bond_dim` using SVD.
/// Returns the gate truncation error epsilon_i.
pub fn apply_two_qubit_gate(
    tn: &mut PlanarTensorNetwork,
    u: usize,
    v: usize,
    gate: &ComplexMatrix,
    telemetry: &mut EvolutionTelemetry,
) -> f64 {
    assert_eq!(gate.rows, 4);
    assert_eq!(gate.cols, 4);

    let pos_in_u = tn
        .site(u)
        .neighbor_bond_pos(v)
        .expect("u and v must be adjacent");
    let pos_in_v = tn
        .site(v)
        .neighbor_bond_pos(u)
        .expect("v and u must be adjacent");

    let site_u = tn.site(u).clone();
    let site_v = tn.site(v).clone();

    let cur_bond_dim = site_u.bond_dims[pos_in_u];
    assert_eq!(cur_bond_dim, site_v.bond_dims[pos_in_v]);

    // Calculate left virtual dimension D_u (excluding bond to v)
    let d_u_others: usize = site_u
        .bond_dims
        .iter()
        .enumerate()
        .filter(|&(p, _)| p != pos_in_u)
        .map(|(_, &dim)| dim)
        .product();

    // Calculate right virtual dimension D_v (excluding bond to u)
    let d_v_others: usize = site_v
        .bond_dims
        .iter()
        .enumerate()
        .filter(|&(p, _)| p != pos_in_v)
        .map(|(_, &dim)| dim)
        .product();

    // Row dimension of theta: 2 (phys u) * D_u
    let left_rows = 2 * d_u_others;
    // Col dimension of theta: 2 (phys v) * D_v
    let right_cols = 2 * d_v_others;

    // Build contracted composite matrix Theta: left_rows x right_cols
    // Theta[(su, i_others), (sv, j_others)] = sum_b T_u(su, b, i_others) * T_v(sv, b, j_others)
    let mut theta = ComplexMatrix::new(left_rows, right_cols);

    // Map other bond index combinations for u and v
    let u_combos = enumerate_other_bond_indices(&site_u, pos_in_u);
    let v_combos = enumerate_other_bond_indices(&site_v, pos_in_v);

    for (i_idx, u_idx_slice) in u_combos.iter().enumerate() {
        for (j_idx, v_idx_slice) in v_combos.iter().enumerate() {
            for b in 0..cur_bond_dim {
                let mut full_u = u_idx_slice.clone();
                full_u.insert(pos_in_u, b);

                let mut full_v = v_idx_slice.clone();
                full_v.insert(pos_in_v, b);

                for su in 0..2 {
                    let u_val = site_u.get_elem(su, &full_u);
                    if u_val.norm_sqr() < 1e-28 {
                        continue;
                    }
                    let row = su * d_u_others + i_idx;

                    for sv in 0..2 {
                        let v_val = site_v.get_elem(sv, &full_v);
                        if v_val.norm_sqr() < 1e-28 {
                            continue;
                        }
                        let col = sv * d_v_others + j_idx;

                        theta[(row, col)] += u_val * v_val;
                    }
                }
            }
        }
    }

    // Apply two-qubit gate G:
    // Theta'_[(s'u, i), (s'v, j)] = sum_{su, sv} G_{(s'u, s'v), (su, sv)} * Theta[(su, i), (sv, j)]
    let mut gated_theta = ComplexMatrix::new(left_rows, right_cols);

    for i_idx in 0..d_u_others {
        for j_idx in 0..d_v_others {
            for s_prime_u in 0..2 {
                for s_prime_v in 0..2 {
                    let gate_row = s_prime_u * 2 + s_prime_v;
                    let out_row = s_prime_u * d_u_others + i_idx;
                    let out_col = s_prime_v * d_v_others + j_idx;

                    let mut sum = Complex64::new(0.0, 0.0);
                    for su in 0..2 {
                        let in_row = su * d_u_others + i_idx;
                        for sv in 0..2 {
                            let in_col = sv * d_v_others + j_idx;
                            let gate_col = su * 2 + sv;
                            let g_val = gate[(gate_row, gate_col)];
                            if g_val.norm_sqr() > 1e-28 {
                                sum += g_val * theta[(in_row, in_col)];
                            }
                        }
                    }
                    gated_theta[(out_row, out_col)] = sum;
                }
            }
        }
    }

    // Perform SVD with truncation to max_bond_dim
    let svd_res = svd(&gated_theta, Some(tn.max_bond_dim));
    let new_bond_dim = svd_res.s.len();

    // Symmetrically absorb singular values: sqrt(S) into left, sqrt(S) into right
    let mut sqrt_s = Vec::with_capacity(new_bond_dim);
    for &sigma in &svd_res.s {
        sqrt_s.push(sigma.max(0.0).sqrt());
    }

    // Reconstruct Site U tensor with new bond dimension
    let mut new_bond_dims_u = site_u.bond_dims.clone();
    new_bond_dims_u[pos_in_u] = new_bond_dim;
    let mut new_u_site = super::state::SiteTensor {
        site_idx: u,
        physical_dim: 2,
        neighbors: site_u.neighbors.clone(),
        bond_dims: new_bond_dims_u,
        strides: Vec::new(),
        data: Vec::new(),
    };
    new_u_site.recompute_strides();
    new_u_site.data = vec![Complex64::new(0.0, 0.0); 2 * new_u_site.strides[0]];

    for (i_idx, u_idx_slice) in u_combos.iter().enumerate() {
        for su in 0..2 {
            let row = su * d_u_others + i_idx;
            for (new_b, &s_val) in sqrt_s.iter().enumerate().take(new_bond_dim) {
                let mut full_u = u_idx_slice.clone();
                full_u.insert(pos_in_u, new_b);
                let val = svd_res.u[(row, new_b)] * s_val;
                new_u_site.set_elem(su, &full_u, val);
            }
        }
    }

    // Reconstruct Site V tensor with new bond dimension
    let mut new_bond_dims_v = site_v.bond_dims.clone();
    new_bond_dims_v[pos_in_v] = new_bond_dim;
    let mut new_v_site = super::state::SiteTensor {
        site_idx: v,
        physical_dim: 2,
        neighbors: site_v.neighbors.clone(),
        bond_dims: new_bond_dims_v,
        strides: Vec::new(),
        data: Vec::new(),
    };
    new_v_site.recompute_strides();
    new_v_site.data = vec![Complex64::new(0.0, 0.0); 2 * new_v_site.strides[0]];

    for (j_idx, v_idx_slice) in v_combos.iter().enumerate() {
        for sv in 0..2 {
            let col = sv * d_v_others + j_idx;
            for (new_b, &s_val) in sqrt_s.iter().enumerate().take(new_bond_dim) {
                let mut full_v = v_idx_slice.clone();
                full_v.insert(pos_in_v, new_b);
                let val = svd_res.v_dag[(new_b, col)] * s_val;
                new_v_site.set_elem(sv, &full_v, val);
            }
        }
    }

    // Update state and bond
    *tn.site_mut(u) = new_u_site;
    *tn.site_mut(v) = new_v_site;

    let bond = tn.bond_mut(u, v).expect("bond must exist");
    bond.weights = svd_res.s;
    bond.normalize();

    // Telemetry update
    let err = svd_res.discarded_weight;
    let gate_fid = (1.0 - err).max(0.0);
    telemetry.two_qubit_gates += 1;
    telemetry.cumulative_fidelity *= gate_fid;
    telemetry.total_truncation_error += err;
    if new_bond_dim > telemetry.max_bond_dim_reached {
        telemetry.max_bond_dim_reached = new_bond_dim;
    }

    err
}

fn enumerate_other_bond_indices(
    site: &super::state::SiteTensor,
    excluded_pos: usize,
) -> Vec<Vec<usize>> {
    let other_dims: Vec<usize> = site
        .bond_dims
        .iter()
        .enumerate()
        .filter(|&(p, _)| p != excluded_pos)
        .map(|(_, &dim)| dim)
        .collect();

    let mut results = Vec::new();
    let mut current = vec![0; other_dims.len()];
    build_index_combos(&other_dims, 0, &mut current, &mut results);
    results
}

fn build_index_combos(
    dims: &[usize],
    depth: usize,
    current: &mut [usize],
    out: &mut Vec<Vec<usize>>,
) {
    if depth == dims.len() {
        out.push(current.to_vec());
        return;
    }
    for i in 0..dims[depth] {
        current[depth] = i;
        build_index_combos(dims, depth + 1, current, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_network::state::LatticeTopology;

    #[test]
    fn test_apply_single_qubit_hadamard() {
        let topo = LatticeTopology::SquareGrid { width: 2, height: 2 };
        let mut tn = PlanarTensorNetwork::new_product_state(topo, 8);
        let mut tel = EvolutionTelemetry::new();

        let h = SingleQubitGate::hadamard();
        apply_single_qubit_gate(&mut tn, 0, &h, &mut tel);

        assert_eq!(tel.single_qubit_gates, 1);
        let s0 = tn.site(0);
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        assert!((s0.get_elem(0, &[0, 0]).re - inv_sqrt2).abs() < 1e-6);
        assert!((s0.get_elem(1, &[0, 0]).re - inv_sqrt2).abs() < 1e-6);
    }

    #[test]
    fn test_create_bell_pair_with_cnot() {
        let topo = LatticeTopology::SquareGrid { width: 2, height: 1 };
        let mut tn = PlanarTensorNetwork::new_product_state(topo, 8);
        let mut tel = EvolutionTelemetry::new();

        // Apply H to qubit 0
        apply_single_qubit_gate(&mut tn, 0, &SingleQubitGate::hadamard(), &mut tel);

        // Apply CNOT between 0 and 1
        let cnot = TwoQubitGate::cnot();
        let err = apply_two_qubit_gate(&mut tn, 0, 1, &cnot, &mut tel);

        assert_eq!(tel.two_qubit_gates, 1);
        assert!(err < 1e-12, "Bell pair creation must have zero truncation error");
        assert_eq!(tn.site(0).neighbor_bond_dim(1), Some(2));
        assert_eq!(tn.site(1).neighbor_bond_dim(0), Some(2));
        assert_eq!(tn.bond(0, 1).unwrap().dim(), 2);
        assert!(tn.validate_integrity().is_ok());

        // Entanglement entropy of maximally entangled Bell pair should be ln(2)
        let entropy = tn.bond(0, 1).unwrap().entanglement_entropy();
        let expected = 2.0_f64.ln();
        assert!((entropy - expected).abs() < 1e-6, "Bell pair entropy was {entropy}, expected {expected}");
    }
}
