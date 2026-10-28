// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

use num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct Permutation {
    pub map: Vec<usize>,
}

impl Permutation {
    pub fn new(map: Vec<usize>) -> Self {
        Self { map }
    }

    pub fn apply(&self, reg: &mut crate::QubitRegister) {
        assert_eq!(reg.basis, self.map.len());
        let mut new_amps = vec![Complex64::default(); reg.amplitudes.len()];
        for (old_idx, &new_idx) in self.map.iter().enumerate() {
            new_amps[new_idx] = reg.amplitudes[old_idx];
        }
        reg.amplitudes = new_amps;
    }
}

pub fn permutation_apply(perm: &Permutation, reg: &mut crate::QubitRegister) {
    perm.apply(reg);
}
