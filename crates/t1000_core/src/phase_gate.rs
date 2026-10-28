// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use num_complex::Complex64;

/// Applies phase rotation onto basis states without changing amplitudes.
#[derive(Debug, Clone)]
pub struct PhaseGate {
    pub phases: Vec<f64>,
}

impl PhaseGate {
    pub fn new(phases: Vec<f64>) -> Self {
        Self { phases }
    }

    pub fn apply(&self, reg: &mut crate::QubitRegister) {
        assert_eq!(reg.basis, self.phases.len());
        for (i, amp) in reg.amplitudes.iter_mut().enumerate() {
            *amp *= Complex64::from_polar(1.0, self.phases[i]);
        }
    }
}

pub fn phase_gate_apply(gate: &PhaseGate, reg: &mut crate::QubitRegister) {
    gate.apply(reg);
}
