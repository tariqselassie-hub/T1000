use std::collections::HashMap;
use num_complex::Complex64;
use super::linalg::{ComplexMatrix, svd};
use super::state::PlanarTensorNetwork;

/// Container for directed message matrices M_{u, v} on each directed edge (u, v).
#[derive(Debug, Clone)]
pub struct BpMessageSet {
    pub messages: HashMap<(usize, usize), ComplexMatrix>,
}

impl BpMessageSet {
    pub fn new_identity(tn: &PlanarTensorNetwork) -> Self {
        let mut messages = HashMap::new();
        for u in 0..tn.num_qubits() {
            let site = tn.site(u);
            for (idx, &v) in site.neighbors.iter().enumerate() {
                let dim = site.bond_dims[idx];
                messages.insert((u, v), ComplexMatrix::identity(dim));
            }
        }
        Self { messages }
    }

    pub fn get(&self, u: usize, v: usize) -> Option<&ComplexMatrix> {
        self.messages.get(&(u, v))
    }

    pub fn get_mut(&mut self, u: usize, v: usize) -> Option<&mut ComplexMatrix> {
        self.messages.get_mut(&(u, v))
    }
}

/// Configuration parameters for Belief Propagation.
#[derive(Debug, Clone)]
pub struct BpConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub damping: f64, // Damping factor in [0, 1), 0 = no damping
}

impl Default for BpConfig {
    fn default() -> Self {
        Self {
            max_iter: 30,
            tol: 1e-8,
            damping: 0.1,
        }
    }
}

/// Compute matrix square root M^{1/2} and inverse square root M^{-1/2} for Hermitian PSD matrix.
pub fn matrix_sqrt_and_inv_sqrt(m: &ComplexMatrix) -> (ComplexMatrix, ComplexMatrix) {
    let svd_res = svd(m, None);
    let k = svd_res.s.len();
    let mut sqrt_diag = ComplexMatrix::new(k, k);
    let mut inv_sqrt_diag = ComplexMatrix::new(k, k);

    for i in 0..k {
        let s = svd_res.s[i].max(0.0);
        let s_sqrt = s.sqrt();
        sqrt_diag[(i, i)] = Complex64::new(s_sqrt, 0.0);

        let inv_s_sqrt = if s_sqrt > 1e-12 { 1.0 / s_sqrt } else { 0.0 };
        inv_sqrt_diag[(i, i)] = Complex64::new(inv_s_sqrt, 0.0);
    }

    // M^{1/2} = U * diag(S^{1/2}) * U^dag
    let m_sqrt = svd_res.u.matmul(&sqrt_diag).matmul(&svd_res.u.conjugate_transpose());
    // M^{-1/2} = U * diag(S^{-1/2}) * U^dag
    let m_inv_sqrt = svd_res.u.matmul(&inv_sqrt_diag).matmul(&svd_res.u.conjugate_transpose());

    (m_sqrt, m_inv_sqrt)
}

/// Update a single directed message M_{v, v_i} according to Eq. (4) of Tindall & Fishman (2025):
/// M_{v, v_i} = sum_{s, {b_j, b'_j}_{j != i}} T_v(s, b_i, {b_j}) * T_v^*(s, b'_i, {b'_j}) * prod_{j != i} [M_{v_j, v}]_{b_j, b'_j}
/// Uses factorized sequential contraction M = \tilde{T} * \tilde{T}^dag in O(chi^{d+1}) operations.
pub fn compute_outgoing_message(
    tn: &PlanarTensorNetwork,
    messages: &BpMessageSet,
    v: usize,
    target_nbr: usize,
) -> ComplexMatrix {
    let site = tn.site(v);
    let target_pos = site
        .neighbor_bond_pos(target_nbr)
        .expect("target_nbr must be a neighbor of v");
    let target_dim = site.bond_dims[target_pos];

    // Sequentially contract incoming message square roots S = M^{1/2} onto each other bond
    let mut contracted_data = site.data.clone();

    for (pos, &nbr) in site.neighbors.iter().enumerate() {
        if pos == target_pos {
            continue;
        }
        let msg = messages.get(nbr, v).expect("incoming message must exist");
        let (sqrt_msg, _) = matrix_sqrt_and_inv_sqrt(msg);
        contracted_data = contract_bond_with_matrix(site, pos, &sqrt_msg, &contracted_data);
    }

    // Now M_{target_b, target_b'} = sum_{s, others} \tilde{T}(s, target_b, others) * \tilde{T}^*(s, target_b', others)
    let mut out = ComplexMatrix::new(target_dim, target_dim);
    let other_dims: Vec<usize> = site
        .bond_dims
        .iter()
        .enumerate()
        .filter(|&(p, _)| p != target_pos)
        .map(|(_, &dim)| dim)
        .collect();

    let mut other_combos = Vec::new();
    let mut cur_combo = vec![0; other_dims.len()];
    build_index_combos(&other_dims, 0, &mut cur_combo, &mut other_combos);

    let mut full_indices = vec![0; site.neighbors.len()];

    for s in 0..site.physical_dim {
        for combo in &other_combos {
            // Map combo into full_indices
            let mut combo_idx = 0;
            for (pos, full_idx) in full_indices.iter_mut().enumerate() {
                if pos == target_pos {
                    continue;
                }
                *full_idx = combo[combo_idx];
                combo_idx += 1;
            }

            // Extract row vector for each target_b
            let mut target_vals = Vec::with_capacity(target_dim);
            for target_b in 0..target_dim {
                full_indices[target_pos] = target_b;
                let idx = site.flat_index(s, &full_indices);
                target_vals.push(contracted_data[idx]);
            }

            // Outer product update: out[b, b'] += target_vals[b] * target_vals[b'].conj()
            for b in 0..target_dim {
                let vb = target_vals[b];
                if vb.norm_sqr() < 1e-28 {
                    continue;
                }
                for b_prime in 0..target_dim {
                    out[(b, b_prime)] += vb * target_vals[b_prime].conj();
                }
            }
        }
    }

    // Normalize message by trace: Tr(M) = 1
    let mut tr = 0.0;
    for i in 0..target_dim {
        tr += out[(i, i)].re;
    }
    if tr > 1e-28 {
        let inv_tr = 1.0 / tr;
        for elem in &mut out.data {
            *elem *= inv_tr;
        }
    }

    out
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

/// Run Belief Propagation to compute fixed-point message tensors on the norm network.
/// Returns (converged_messages, final_delta_c, iterations_executed).
pub fn run_belief_propagation(
    tn: &PlanarTensorNetwork,
    config: &BpConfig,
) -> (BpMessageSet, f64, usize) {
    let mut messages = BpMessageSet::new_identity(tn);
    let edges = tn.topology.edges();
    let num_directed_edges = edges.len() * 2;
    let mut final_delta = 0.0;
    let mut iters_done = 0;

    for _iter in 0..config.max_iter {
        iters_done += 1;
        let mut delta_sum = 0.0;
        let old_messages = messages.clone();

        // Perform hybrid edge updates
        for &(u, v) in &edges {
            // Forward update M_{u, v}
            let new_m_uv = compute_outgoing_message(tn, &old_messages, u, v);
            let prev_m_uv = old_messages.get(u, v).unwrap();
            let diff_uv = new_m_uv.data
                .iter()
                .zip(&prev_m_uv.data)
                .map(|(a, b)| (a - b).norm())
                .sum::<f64>();
            delta_sum += diff_uv;

            // Apply damping: M = (1 - damping) * new + damping * old
            let damped_uv = if config.damping > 0.0 {
                let mut d = ComplexMatrix::new(new_m_uv.rows, new_m_uv.cols);
                let w_new = 1.0 - config.damping;
                for i in 0..d.data.len() {
                    d.data[i] = new_m_uv.data[i] * w_new + prev_m_uv.data[i] * config.damping;
                }
                d
            } else {
                new_m_uv
            };
            *messages.get_mut(u, v).unwrap() = damped_uv;

            // Backward update M_{v, u}
            let new_m_vu = compute_outgoing_message(tn, &old_messages, v, u);
            let prev_m_vu = old_messages.get(v, u).unwrap();
            let diff_vu = new_m_vu.data
                .iter()
                .zip(&prev_m_vu.data)
                .map(|(a, b)| (a - b).norm())
                .sum::<f64>();
            delta_sum += diff_vu;

            let damped_vu = if config.damping > 0.0 {
                let mut d = ComplexMatrix::new(new_m_vu.rows, new_m_vu.cols);
                let w_new = 1.0 - config.damping;
                for i in 0..d.data.len() {
                    d.data[i] = new_m_vu.data[i] * w_new + prev_m_vu.data[i] * config.damping;
                }
                d
            } else {
                new_m_vu
            };
            *messages.get_mut(v, u).unwrap() = damped_vu;
        }

        final_delta = if num_directed_edges > 0 {
            delta_sum / (num_directed_edges as f64)
        } else {
            0.0
        };

        if final_delta < config.tol {
            break;
        }
    }

    (messages, final_delta, iters_done)
}

/// Bring a PlanarTensorNetwork into the Vidal gauge using converged BP message tensors.
/// Executes the single terminal gauge transformation from Tindall & Fishman (2025).
pub fn gauge_into_vidal(tn: &mut PlanarTensorNetwork, messages: &BpMessageSet) {
    let edges = tn.topology.edges();

    // Map to hold edge transformation matrices X_{u, v} and X_{v, u}
    let mut edge_transforms: HashMap<(usize, usize), ComplexMatrix> = HashMap::new();

    for &(u, v) in &edges {
        let m_uv = messages.get(u, v).expect("message (u, v) must exist");
        let m_vu = messages.get(v, u).expect("message (v, u) must exist");

        let (sqrt_uv, inv_sqrt_uv) = matrix_sqrt_and_inv_sqrt(m_uv);
        let (sqrt_vu, inv_sqrt_vu) = matrix_sqrt_and_inv_sqrt(m_vu);

        // M_{u, v}^{1/2} * M_{v, u}^{1/2} ~ U * Lambda * V^dag
        let prod = sqrt_uv.matmul(&sqrt_vu);
        let svd_res = svd(&prod, None);

        // Store new singular values as bond tensor weights
        let bond = tn.bond_mut(u, v).expect("bond must exist");
        bond.weights = svd_res.s.clone();
        bond.normalize();

        // X_{u, v} = M_{u, v}^{-1/2} * U
        let x_uv = inv_sqrt_uv.matmul(&svd_res.u);
        // X_{v, u} = M_{v, u}^{-1/2} * (V^dag)^dag = M_{v, u}^{-1/2} * V
        let x_vu = inv_sqrt_vu.matmul(&svd_res.v_dag.conjugate_transpose());

        edge_transforms.insert((u, v), x_uv);
        edge_transforms.insert((v, u), x_vu);
    }

    // Transform each site tensor: T_v -> Gamma_v by contracting incident bonds with X_{v, v_i}
    let n = tn.num_qubits();
    for site_idx in 0..n {
        let site = tn.site(site_idx).clone();
        let mut transformed_data = site.data.clone();

        for (bond_pos, &nbr) in site.neighbors.iter().enumerate() {
            if let Some(x_mat) = edge_transforms.get(&(site_idx, nbr)) {
                let dim = site.bond_dims[bond_pos];
                if dim > 1 && x_mat.rows == dim && x_mat.cols == dim {
                    // Contract index bond_pos with x_mat
                    transformed_data = contract_bond_with_matrix(
                        &site,
                        bond_pos,
                        x_mat,
                        &transformed_data,
                    );
                }
            }
        }

        tn.site_mut(site_idx).data = transformed_data;
    }
}

fn contract_bond_with_matrix(
    site: &super::state::SiteTensor,
    bond_pos: usize,
    mat: &ComplexMatrix,
    current_data: &[Complex64],
) -> Vec<Complex64> {
    let d = site.neighbors.len();
    let dim = site.bond_dims[bond_pos];
    let mut new_data = vec![Complex64::new(0.0, 0.0); current_data.len()];

    let mut indices = vec![0; d];
    let total_outer = site.physical_dim;

    // Enumerate all slices
    for s in 0..total_outer {
        enumerate_bond_contraction(
            site,
            bond_pos,
            dim,
            mat,
            s,
            0,
            &mut indices,
            current_data,
            &mut new_data,
        );
    }

    new_data
}

#[allow(clippy::too_many_arguments)]
fn enumerate_bond_contraction(
    site: &super::state::SiteTensor,
    target_pos: usize,
    dim: usize,
    mat: &ComplexMatrix,
    s: usize,
    cur_pos: usize,
    indices: &mut [usize],
    in_data: &[Complex64],
    out_data: &mut [Complex64],
) {
    if cur_pos == site.neighbors.len() {
        // Contract target bond
        for new_b in 0..dim {
            indices[target_pos] = new_b;
            let out_idx = site.flat_index(s, indices);
            let mut sum = Complex64::new(0.0, 0.0);
            for old_b in 0..dim {
                indices[target_pos] = old_b;
                let in_idx = site.flat_index(s, indices);
                sum += in_data[in_idx] * mat[(old_b, new_b)];
            }
            indices[target_pos] = new_b;
            out_data[out_idx] = sum;
        }
        return;
    }

    if cur_pos == target_pos {
        enumerate_bond_contraction(
            site,
            target_pos,
            dim,
            mat,
            s,
            cur_pos + 1,
            indices,
            in_data,
            out_data,
        );
    } else {
        let b_dim = site.bond_dims[cur_pos];
        for b in 0..b_dim {
            indices[cur_pos] = b;
            enumerate_bond_contraction(
                site,
                target_pos,
                dim,
                mat,
                s,
                cur_pos + 1,
                indices,
                in_data,
                out_data,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_network::state::LatticeTopology;

    #[test]
    fn test_bp_message_set_initialization() {
        let topo = LatticeTopology::SquareGrid { width: 3, height: 3 };
        let tn = PlanarTensorNetwork::new_product_state(topo, 16);
        let msgs = BpMessageSet::new_identity(&tn);
        assert_eq!(msgs.messages.len(), 24); // 12 edges * 2 directed
        for msg in msgs.messages.values() {
            assert_eq!(msg.rows, 1);
            assert_eq!(msg.cols, 1);
            assert_eq!(msg[(0, 0)], Complex64::new(1.0, 0.0));
        }
    }

    #[test]
    fn test_matrix_sqrt_and_inv_sqrt() {
        let mut m = ComplexMatrix::new(2, 2);
        m[(0, 0)] = Complex64::new(4.0, 0.0);
        m[(1, 1)] = Complex64::new(9.0, 0.0);

        let (sqrt_m, inv_sqrt_m) = matrix_sqrt_and_inv_sqrt(&m);
        assert!((sqrt_m[(0, 0)].re - 2.0).abs() < 1e-6);
        assert!((sqrt_m[(1, 1)].re - 3.0).abs() < 1e-6);
        assert!((inv_sqrt_m[(0, 0)].re - 0.5).abs() < 1e-6);
        assert!((inv_sqrt_m[(1, 1)].re - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_run_bp_on_product_state() {
        let topo = LatticeTopology::SquareGrid { width: 4, height: 4 };
        let mut tn = PlanarTensorNetwork::new_product_state(topo, 16);
        let config = BpConfig {
            max_iter: 10,
            tol: 1e-8,
            damping: 0.0,
        };

        let (msgs, delta, iters) = run_belief_propagation(&tn, &config);
        assert!(iters <= 2, "Product state BP must converge in <= 2 iterations (took {iters})");
        assert!(delta < 1e-7);

        // Terminal Vidal gauge transformation
        gauge_into_vidal(&mut tn, &msgs);
        assert!(tn.validate_integrity().is_ok());
    }
}
