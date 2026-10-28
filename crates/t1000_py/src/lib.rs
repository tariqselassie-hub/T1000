// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Native module `t1000._t1000`: thin wrappers over the T-1000 engine. The Python package
//! `t1000` (python/t1000/__init__.py) adds the user-facing API on top.

mod chemistry;
mod circuits;
mod vqc;

use pyo3::prelude::*;

/// Largest register accepted: 2^26 amplitudes of two f64 each is 1 GiB.
pub(crate) const MAX_QUBITS: usize = 26;

#[pymodule]
fn _t1000(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<vqc::VqcModel>()?;
    m.add_function(wrap_pyfunction!(vqc::vqc_train, m)?)?;
    m.add_class::<circuits::State>()?;
    m.add_function(wrap_pyfunction!(circuits::run_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(chemistry::vqe_solve_json, m)?)?;
    m.add_function(wrap_pyfunction!(chemistry::vqe_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(chemistry::hamiltonian_json, m)?)?;
    m.add_function(wrap_pyfunction!(chemistry::classical_json, m)?)?;
    m.add_function(wrap_pyfunction!(chemistry::adapt_json, m)?)?;
    m.add_function(wrap_pyfunction!(chemistry::scan_json, m)?)?;
    Ok(())
}
