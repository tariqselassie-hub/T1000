// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Native module `t1000._t1000`: thin wrappers over the T-1000 engine. The Python package
//! `t1000` (python/t1000/__init__.py) adds the user-facing API on top.

use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use t1000::compiler::{CircuitBuilder, Unitary2x2};
use t1000::vqc_classifier::{train_vqc, TrainedVqc, VqcConfig};
use t1000::vqe::basis::Atom;
use t1000::vqe::driver::{solve, VqeConfig};
use t1000::vqe::molecules::{by_formula, catalog, Molecule};
use t1000::QuantumState;

/// Largest register `run_circuit` accepts: 2^26 amplitudes of two f64 each is 1 GiB.
const MAX_QUBITS: usize = 26;

/// A trained variational quantum circuit classifier.
#[pyclass(module = "t1000._t1000")]
#[derive(Debug, Clone)]
struct VqcModel {
    model: TrainedVqc,
    n_qubits: usize,
}

#[pymethods]
impl VqcModel {
    /// Readout <Z> in [-1, 1] for each row of `x`; its sign is the predicted class.
    fn decision_function<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let rows = rows_of(&x, self.n_qubits)?;
        let model = self.model.clone();
        let out = py.detach(move || model.decisions(&rows));
        Ok(out.into_pyarray(py))
    }

    /// Trained circuit weights.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.model.weights.clone().into_pyarray(py)
    }

    /// Mean 0.5 (y_pred - y)^2 over the training rows at the last epoch.
    #[getter]
    fn final_loss(&self) -> f64 {
        self.model.final_loss
    }

    #[getter]
    fn n_qubits(&self) -> usize {
        self.n_qubits
    }

    fn __repr__(&self) -> String {
        format!(
            "VqcModel(n_qubits={}, parameters={}, final_loss={:.4})",
            self.n_qubits,
            self.model.parameter_count(),
            self.model.final_loss
        )
    }
}

/// Copies a 2-D float array into rows, checking the column count.
fn rows_of(x: &PyReadonlyArray2<'_, f64>, n_cols: usize) -> PyResult<Vec<Vec<f64>>> {
    let a = x.as_array();
    if a.ncols() != n_cols {
        return Err(PyValueError::new_err(format!(
            "expected {n_cols} feature columns (one per qubit), got {}",
            a.ncols()
        )));
    }
    Ok(a.rows().into_iter().map(|r| r.to_vec()).collect())
}

/// Trains a classifier on `x` (n_samples x n_qubits angles) and labels `y` in {+1, -1}.
#[pyfunction]
#[pyo3(signature = (x, y, depth=3, epochs=30, lr=0.05, seed=11))]
fn vqc_train(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    depth: usize,
    epochs: usize,
    lr: f64,
    seed: u64,
) -> PyResult<VqcModel> {
    let n_qubits = x.as_array().ncols();
    if n_qubits == 0 || n_qubits > MAX_QUBITS {
        return Err(PyValueError::new_err(format!(
            "need 1 to {MAX_QUBITS} feature columns, got {n_qubits}"
        )));
    }
    let rows = rows_of(&x, n_qubits)?;
    let labels = y.as_array().to_vec();
    if rows.is_empty() || rows.len() != labels.len() {
        return Err(PyValueError::new_err(format!(
            "x has {} rows but y has {} labels",
            rows.len(),
            labels.len()
        )));
    }
    if labels.iter().any(|&v| v != 1.0 && v != -1.0) {
        return Err(PyValueError::new_err("labels must be +1 or -1"));
    }
    let cfg = VqcConfig {
        n_qubits,
        depth,
        epochs,
        lr,
        seed,
    };
    let model = py.detach(move || train_vqc(&rows, &labels, &cfg));
    Ok(VqcModel { model, n_qubits })
}

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

/// Applies `gates` to |0...0> on `n_qubits` qubits and returns the state vector.
///
/// Each gate is `(name, qubits, angle)`: names H X Y Z S T RX RY RZ (one qubit) and CX/CNOT CZ
/// SWAP CP (two qubits; CP takes an angle). Qubit 0 is the least significant bit of the index.
#[pyfunction]
fn run_circuit<'py>(
    py: Python<'py>,
    n_qubits: usize,
    gates: Vec<(String, Vec<usize>, Option<f64>)>,
) -> PyResult<Bound<'py, PyArray1<Complex64>>> {
    if n_qubits == 0 || n_qubits > MAX_QUBITS {
        return Err(PyValueError::new_err(format!(
            "n_qubits must be 1 to {MAX_QUBITS}, got {n_qubits}"
        )));
    }
    let mut builder = CircuitBuilder::new(n_qubits);
    for (k, (raw, qubits, theta)) in gates.iter().enumerate() {
        let name = raw.to_ascii_uppercase();
        if let Some(&q) = qubits.iter().find(|&&q| q >= n_qubits) {
            return Err(PyValueError::new_err(format!(
                "gate {k} ({name}): qubit {q} out of range"
            )));
        }
        match (name.as_str(), qubits.as_slice()) {
            ("CX" | "CNOT", &[a, b]) if a != b => {
                builder.cnot(a, b);
            }
            ("CZ", &[a, b]) if a != b => {
                builder.cz(a, b);
            }
            ("SWAP", &[a, b]) if a != b => {
                builder.swap(a, b);
            }
            ("CP", &[a, b]) if a != b => {
                let t = theta.ok_or_else(|| PyValueError::new_err(format!("gate {k} (CP) needs an angle")))?;
                builder.controlled_phase(a, b, t);
            }
            (_, &[q]) => {
                builder.unitary_1q(q, single_qubit(&name, *theta)?);
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "gate {k} ({name}): wrong number of qubits {qubits:?} (two-qubit gates need two different qubits)"
                )))
            }
        }
    }
    let amplitudes = py.detach(move || {
        let mut state = QuantumState::ground(1 << n_qubits);
        builder.execute(&mut state);
        state.to_complex_vec()
    });
    Ok(amplitudes.into_pyarray(py))
}

/// Ab initio VQE (STO-3G, RHF, UCCSD) for a catalog formula or explicit atoms; returns JSON.
#[pyfunction]
#[pyo3(signature = (formula=None, atoms=None, charge=0, freeze_core=true, active_orbitals=None))]
fn vqe_solve_json(
    py: Python<'_>,
    formula: Option<String>,
    atoms: Option<Vec<(u32, [f64; 3])>>,
    charge: i32,
    freeze_core: bool,
    active_orbitals: Option<usize>,
) -> PyResult<String> {
    let molecule = match (formula, atoms) {
        (Some(f), None) => by_formula(&f)
            .ok_or_else(|| PyValueError::new_err(format!("{f} is not in the catalog")))?,
        (None, Some(list)) if !list.is_empty() => Molecule {
            name: "custom",
            formula: "custom",
            atoms: list
                .iter()
                .map(|&(z, pos)| Atom::angstrom(z, pos))
                .collect(),
            charge,
        },
        _ => {
            return Err(PyValueError::new_err(
                "pass exactly one of formula= or atoms=",
            ))
        }
    };
    let cfg = VqeConfig {
        freeze_core,
        active_orbitals,
        ..VqeConfig::default()
    };
    let report = py
        .detach(move || solve(&molecule, &cfg))
        .map_err(PyValueError::new_err)?;
    serde_json::to_string(&report).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Formulas of the built-in VQE molecule catalog.
#[pyfunction]
fn vqe_catalog() -> Vec<String> {
    catalog().iter().map(|m| m.formula.to_string()).collect()
}

#[pymodule]
fn _t1000(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<VqcModel>()?;
    m.add_function(wrap_pyfunction!(vqc_train, m)?)?;
    m.add_function(wrap_pyfunction!(run_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(vqe_solve_json, m)?)?;
    m.add_function(wrap_pyfunction!(vqe_catalog, m)?)?;
    Ok(())
}
