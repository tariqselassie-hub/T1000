// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! The variational quantum circuit classifier (`t1000::vqc_classifier`).

use crate::MAX_QUBITS;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use t1000::vqc_classifier::{train_vqc, TrainedVqc, VqcConfig};

/// A trained variational quantum circuit classifier.
#[pyclass(module = "t1000._t1000")]
#[derive(Debug, Clone)]
pub struct VqcModel {
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
pub fn vqc_train(
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
