// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! State-vector simulation: one-shot `run_circuit` and the stateful `State`, with Pauli-string
//! expectation values, seeded shot sampling and the engine's device-noise models.

use crate::MAX_QUBITS;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use t1000::compiler::{CircuitBuilder, Unitary2x2};
use t1000::device_noise::{apply_readout, depolarize, ReadoutError};
use t1000::QuantumState;

/// A gate as the Python layer passes it: `(name, qubits, angle)`.
pub type Gate = (String, Vec<usize>, Option<f64>);

/// Named single-qubit gates as 2x2 unitaries.
fn single_qubit(name: &str, theta: Option<f64>) -> PyResult<Unitary2x2> {
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let (o, z, i) = (c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0));
    let angle = || theta.ok_or_else(|| PyValueError::new_err(format!("{name} needs an angle")));
    Ok(match name {
        "H" => Unitary2x2::hadamard(),
        "X" => Unitary2x2::new(z, o, o, z),
        "Y" => Unitary2x2::new(z, -i, i, z),
        "Z" => Unitary2x2::new(o, z, z, -o),
        "S" => Unitary2x2::new(o, z, z, i),
        "T" => Unitary2x2::new(
            o,
            z,
            z,
            Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4),
        ),
        "RX" => Unitary2x2::rx(angle()?),
        "RY" => Unitary2x2::ry(angle()?),
        "RZ" => Unitary2x2::rz(angle()?),
        _ => return Err(PyValueError::new_err(format!("unknown gate {name}"))),
    })
}

/// Builds the circuit and counts its one- and two-qubit gates.
///
/// Names: H X Y Z S T RX RY RZ (one qubit), CX/CNOT CZ SWAP CP (two qubits; CP takes an angle).
fn build(n_qubits: usize, gates: &[Gate]) -> PyResult<(CircuitBuilder, usize, usize)> {
    let mut builder = CircuitBuilder::new(n_qubits);
    let (mut n1, mut n2) = (0, 0);
    for (k, (raw, qubits, theta)) in gates.iter().enumerate() {
        let name = raw.to_ascii_uppercase();
        if let Some(&q) = qubits.iter().find(|&&q| q >= n_qubits) {
            return Err(PyValueError::new_err(format!(
                "gate {k} ({name}): qubit {q} out of range"
            )));
        }
        let two = |builder: &mut CircuitBuilder| -> PyResult<()> {
            let &[a, b] = qubits.as_slice() else {
                unreachable!()
            };
            match name.as_str() {
                "CX" | "CNOT" => builder.cnot(a, b),
                "CZ" => builder.cz(a, b),
                "SWAP" => builder.swap(a, b),
                _ => {
                    let t = theta.ok_or_else(|| {
                        PyValueError::new_err(format!("gate {k} (CP) needs an angle"))
                    })?;
                    builder.controlled_phase(a, b, t)
                }
            };
            Ok(())
        };
        match (name.as_str(), qubits.as_slice()) {
            ("CX" | "CNOT" | "CZ" | "SWAP" | "CP", &[a, b]) if a != b => {
                two(&mut builder)?;
                n2 += 1;
            }
            (_, &[q]) => {
                builder.unitary_1q(q, single_qubit(&name, *theta)?);
                n1 += 1;
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "gate {k} ({name}): wrong number of qubits {qubits:?} (two-qubit gates need two different qubits)"
                )))
            }
        }
    }
    Ok((builder, n1, n2))
}

fn check_qubits(n_qubits: usize) -> PyResult<()> {
    if n_qubits == 0 || n_qubits > MAX_QUBITS {
        return Err(PyValueError::new_err(format!(
            "n_qubits must be 1 to {MAX_QUBITS}, got {n_qubits}"
        )));
    }
    Ok(())
}

/// Applies `gates` to |0...0> on `n_qubits` qubits and returns the state vector.
/// Qubit 0 is the least significant bit of the index.
#[pyfunction]
pub fn run_circuit<'py>(
    py: Python<'py>,
    n_qubits: usize,
    gates: Vec<Gate>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    check_qubits(n_qubits)?;
    let (builder, _, _) = build(n_qubits, &gates)?;
    let amplitudes = py.detach(move || {
        let mut state = QuantumState::ground(1 << n_qubits);
        builder.execute(&mut state);
        state.to_complex_vec()
    });
    Ok(amplitudes.into_pyarray(py))
}

/// A Pauli string as bit masks: X (or Y) flips, Z (or Y) phases.
#[derive(Debug, Clone, Copy)]
struct Pauli {
    x: usize,
    z: usize,
}

/// Parses `"X0 Z1 Y3"` (qubits not listed carry the identity; `""` or `"I"` is the identity).
fn parse_pauli(label: &str, n_qubits: usize) -> PyResult<Pauli> {
    let mut p = Pauli { x: 0, z: 0 };
    let bad = |m: String| PyValueError::new_err(format!("Pauli string {label:?}: {m}"));
    for token in label.split_whitespace() {
        if token.eq_ignore_ascii_case("I") {
            continue;
        }
        let (op, rest) = token.split_at(1);
        let q: usize = rest
            .parse()
            .map_err(|_| bad(format!("bad term {token:?}, expected e.g. X0")))?;
        if q >= n_qubits {
            return Err(bad(format!("qubit {q} out of range")));
        }
        let bit = 1usize << q;
        if (p.x | p.z) & bit != 0 {
            return Err(bad(format!("qubit {q} appears twice")));
        }
        match op.to_ascii_uppercase().as_str() {
            "X" => p.x |= bit,
            "Z" => p.z |= bit,
            "Y" => {
                p.x |= bit;
                p.z |= bit;
            }
            _ => return Err(bad(format!("unknown Pauli {op:?}"))),
        }
    }
    Ok(p)
}

/// `<psi|P|psi>` with `P` acting as `P|i> = phase(i) |i ^ x>`: each Z or Y on a set bit gives -1,
/// each Y gives a further `i` (`Y = i X Z`).
fn pauli_expectation(state: &QuantumState, p: Pauli) -> f64 {
    let y_phase = match (p.x & p.z).count_ones() % 4 {
        0 => Complex64::new(1.0, 0.0),
        1 => Complex64::new(0.0, 1.0),
        2 => Complex64::new(-1.0, 0.0),
        _ => Complex64::new(0.0, -1.0),
    };
    let mut acc = Complex64::new(0.0, 0.0);
    for i in 0..state.dim {
        let sign = if (i & p.z).count_ones().is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        let j = i ^ p.x;
        let a_i = Complex64::new(state.real[i], state.imag[i]);
        let a_j = Complex64::new(state.real[j], -state.imag[j]);
        acc += a_j * a_i * sign;
    }
    (acc * y_phase).re
}

/// A state vector that gates are applied to in place.
///
/// Noise uses the engine's device models (`t1000::device_noise`): a global depolarizing
/// approximation with fidelity `(1 - p1)^n1 (1 - p2)^n2` from the gates applied so far, and
/// independent per-qubit readout flips. Both act on measured quantities, not on the state.
#[pyclass(module = "t1000._t1000")]
#[derive(Debug, Clone)]
pub struct State {
    n_qubits: usize,
    state: QuantumState,
    n1: usize,
    n2: usize,
}

impl State {
    fn fidelity(&self, p1: f64, p2: f64) -> PyResult<f64> {
        if !(0.0..=1.0).contains(&p1) || !(0.0..=1.0).contains(&p2) {
            return Err(PyValueError::new_err("p1 and p2 must lie in [0, 1]"));
        }
        Ok((1.0 - p1).powi(self.n1 as i32) * (1.0 - p2).powi(self.n2 as i32))
    }

    fn noisy_probabilities(
        &self,
        p1: f64,
        p2: f64,
        readout: Option<(f64, f64)>,
    ) -> PyResult<Vec<f64>> {
        let mut p = self.state.probabilities();
        if p1 > 0.0 || p2 > 0.0 {
            depolarize(&mut p, self.fidelity(p1, p2)?);
        }
        if let Some((p01, p10)) = readout {
            if !(0.0..=0.5).contains(&p01) || !(0.0..=0.5).contains(&p10) {
                return Err(PyValueError::new_err(
                    "readout flip probabilities must lie in [0, 0.5]",
                ));
            }
            apply_readout(&mut p, self.n_qubits, ReadoutError { p01, p10 });
        }
        Ok(p)
    }
}

#[pymethods]
impl State {
    /// `n_qubits` qubits in the computational basis state `basis_state` (default |0...0>).
    #[new]
    #[pyo3(signature = (n_qubits, basis_state=0))]
    fn new(n_qubits: usize, basis_state: usize) -> PyResult<Self> {
        check_qubits(n_qubits)?;
        if basis_state >= 1 << n_qubits {
            return Err(PyValueError::new_err(format!(
                "basis_state {basis_state} needs more than {n_qubits} qubits"
            )));
        }
        let mut state = QuantumState::zero(1 << n_qubits);
        state.real[basis_state] = 1.0;
        Ok(State {
            n_qubits,
            state,
            n1: 0,
            n2: 0,
        })
    }

    /// A state from complex amplitudes (length a power of two, normalized to 1e-9).
    #[staticmethod]
    fn from_amplitudes(amplitudes: PyReadonlyArray1<'_, Complex64>) -> PyResult<Self> {
        let a = amplitudes.as_array();
        let n_qubits = a.len().trailing_zeros() as usize;
        if !a.len().is_power_of_two() || n_qubits == 0 {
            return Err(PyValueError::new_err("need 2^n amplitudes with n >= 1"));
        }
        check_qubits(n_qubits)?;
        let norm: f64 = a.iter().map(|c| c.norm_sqr()).sum();
        if (norm - 1.0).abs() > 1e-9 {
            return Err(PyValueError::new_err(format!(
                "amplitudes are not normalized (norm^2 = {norm})"
            )));
        }
        let state = QuantumState {
            dim: a.len(),
            real: a.iter().map(|c| c.re).collect(),
            imag: a.iter().map(|c| c.im).collect(),
        };
        Ok(State {
            n_qubits,
            state,
            n1: 0,
            n2: 0,
        })
    }

    /// Applies gates in order (same names as `run_circuit`).
    fn apply(&mut self, py: Python<'_>, gates: Vec<Gate>) -> PyResult<()> {
        let (builder, n1, n2) = build(self.n_qubits, &gates)?;
        let state = &mut self.state;
        py.detach(|| {
            builder.execute(state);
        });
        self.n1 += n1;
        self.n2 += n2;
        Ok(())
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    /// `(one_qubit_gates, two_qubit_gates)` applied so far.
    #[getter]
    fn gate_counts(&self) -> (usize, usize) {
        (self.n1, self.n2)
    }

    #[getter]
    fn amplitudes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<Complex64>> {
        self.state.to_complex_vec().into_pyarray(py)
    }

    /// Outcome probabilities, optionally with depolarizing gate error and readout error.
    #[pyo3(signature = (p1=0.0, p2=0.0, readout=None))]
    fn probabilities<'py>(
        &self,
        py: Python<'py>,
        p1: f64,
        p2: f64,
        readout: Option<(f64, f64)>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.noisy_probabilities(p1, p2, readout)?.into_pyarray(py))
    }

    /// `<P>` for a Pauli string such as `"X0 Z1 Y3"`. With gate error, the depolarizing model
    /// scales every non-identity expectation by the circuit fidelity.
    #[pyo3(signature = (pauli, p1=0.0, p2=0.0))]
    fn expectation(&self, pauli: &str, p1: f64, p2: f64) -> PyResult<f64> {
        let p = parse_pauli(pauli, self.n_qubits)?;
        let f = if p.x | p.z == 0 {
            1.0
        } else {
            self.fidelity(p1, p2)?
        };
        Ok(f * pauli_expectation(&self.state, p))
    }

    /// `constant + sum coeff_k <P_k>` over `terms = [(coeff, "X0 Z1"), ...]`, e.g. a Hamiltonian.
    #[pyo3(signature = (terms, constant=0.0, p1=0.0, p2=0.0))]
    fn expectation_sum(
        &self,
        py: Python<'_>,
        terms: Vec<(f64, String)>,
        constant: f64,
        p1: f64,
        p2: f64,
    ) -> PyResult<f64> {
        let paulis = terms
            .iter()
            .map(|(c, l)| parse_pauli(l, self.n_qubits).map(|p| (*c, p)))
            .collect::<PyResult<Vec<_>>>()?;
        let f = self.fidelity(p1, p2)?;
        let state = &self.state;
        let sum = py.detach(|| {
            paulis
                .iter()
                .map(|&(c, p)| {
                    let scale = if p.x | p.z == 0 { 1.0 } else { f };
                    c * scale * pauli_expectation(state, p)
                })
                .sum::<f64>()
        });
        Ok(constant + sum)
    }

    /// `shots` measurement outcomes (basis-state indices), reproducible for a given `seed`.
    #[pyo3(signature = (shots, seed=0, p1=0.0, p2=0.0, readout=None))]
    fn sample<'py>(
        &self,
        py: Python<'py>,
        shots: usize,
        seed: u64,
        p1: f64,
        p2: f64,
        readout: Option<(f64, f64)>,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let p = self.noisy_probabilities(p1, p2, readout)?;
        let out = py.detach(move || {
            let mut cdf = Vec::with_capacity(p.len());
            let mut acc = 0.0;
            for v in &p {
                acc += v.max(0.0);
                cdf.push(acc);
            }
            let total = acc;
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            (0..shots)
                .map(|_| {
                    let u: f64 = rng.gen::<f64>() * total;
                    cdf.partition_point(|&c| c <= u).min(cdf.len() - 1) as i64
                })
                .collect::<Vec<i64>>()
        });
        Ok(out.into_pyarray(py))
    }

    fn copy(&self) -> State {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "State(n_qubits={}, gates={}+{})",
            self.n_qubits, self.n1, self.n2
        )
    }
}
