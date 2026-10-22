// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Threaded holdout benchmark for the four 12-qubit ZINC circuit kernels (full, pruned, fused
//! observable rotation, and fused with qubit 11 skipped): one place that knows how each kernel
//! evolves a molecule, reads out its Z expectations and maps them to (logP, QED, SAS).

use crate::zinc_data::ZincMolecule;
use crate::zinc_training_common::eval_all_z12;
use crate::{CircuitAnsatz, DataEncodingLayer, QuantumState};

const DIM: usize = 1 << 12;

/// The four benchmarked circuit kernels.
#[derive(Debug, Clone, Copy)]
pub enum Kernel {
    /// Mode A: all 111 parameters, 108 gates.
    Full,
    /// Mode B: 99 parameters, 84 gates (terminal Rz and ring CZ stripped).
    Pruned,
    /// Mode C: 99 parameters, evolution stops after layer 2, rotated-observable readout.
    Fused12,
    /// Mode D: 98 parameters, as C with qubit 11 skipped.
    Fused11,
}

impl Kernel {
    /// Number of Z expectation values the readout uses.
    pub fn n_z(self) -> usize {
        match self {
            Kernel::Fused11 => 11,
            _ => 12,
        }
    }

    /// Quantum angles, which precede the readout weights in the parameter vector.
    pub fn n_quantum(self) -> usize {
        match self {
            Kernel::Full => 72,
            Kernel::Pruned | Kernel::Fused12 => 60,
            Kernel::Fused11 => 59,
        }
    }
}

/// The shared circuit and input encoding.
#[derive(Debug)]
pub struct Bench<'a> {
    pub ansatz: &'a CircuitAnsatz,
    pub encoding: &'a DataEncodingLayer,
}

impl Bench<'_> {
    /// Encodes one molecule, runs `kernel` and leaves its Z expectations in `z[..n_z]`.
    pub fn evolve(
        &self,
        kernel: Kernel,
        params: &[f64],
        features: &[f64],
        state: &mut QuantumState,
        z: &mut [f64; 12],
    ) {
        self.encoding.encode_amplitude_into(features, state);
        match kernel {
            Kernel::Full => {
                self.ansatz.forward_in_place(state, &params[..72]);
                eval_all_z12(state, z);
            }
            Kernel::Pruned => {
                self.ansatz.forward_pruned_in_place(state, &params[..60]);
                eval_all_z12(state, z);
            }
            Kernel::Fused12 | Kernel::Fused11 => {
                let n = kernel.n_z();
                self.ansatz.forward_l2_in_place(state, &params[..48]);
                let theta_y = &params[48..48 + n];
                crate::simd::measure_fused_ry_z_readout(
                    &state.real,
                    &state.imag,
                    theta_y,
                    &mut z[..n],
                );
            }
        }
    }

    /// `repeats` passes over one chunk; only the first pass keeps predictions.
    pub fn run_chunk(
        &self,
        kernel: Kernel,
        params: &[f64],
        chunk: &[ZincMolecule],
        repeats: usize,
    ) -> Vec<[f64; 3]> {
        let mut state = QuantumState::new(DIM);
        let mut z = [0.0; 12];
        let mut preds = Vec::with_capacity(chunk.len());
        let w_off = kernel.n_quantum();
        let b_off = w_off + 36;

        for rep in 0..repeats {
            for sample in chunk {
                self.evolve(kernel, params, &sample.features, &mut state, &mut z);
                if rep == 0 {
                    preds.push(readout(params, w_off, b_off, &z[..kernel.n_z()]));
                }
            }
        }
        preds
    }

    /// Splits the holdout set across threads and concatenates the first-pass predictions.
    pub fn run_threads(
        &self,
        kernel: Kernel,
        params: &[f64],
        samples: &[ZincMolecule],
        num_threads: usize,
        repeats: usize,
    ) -> Vec<[f64; 3]> {
        let chunk_size = samples.len().div_ceil(num_threads);
        let sub_chunks: Vec<&[ZincMolecule]> = samples.chunks(chunk_size).collect();

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(sub_chunks.len());
            for chunk in sub_chunks {
                handles.push(s.spawn(move || self.run_chunk(kernel, params, chunk, repeats)));
            }
            let mut all = Vec::with_capacity(samples.len());
            for h in handles {
                all.extend(h.join().unwrap());
            }
            all
        })
    }
}

/// (logP, QED, SAS) from the Z expectations: bias plus the three weight rows
/// (row stride 12, whatever the kernel's width).
pub fn readout(p: &[f64], w_off: usize, b_off: usize, z: &[f64]) -> [f64; 3] {
    let mut p_logp = p[b_off];
    let mut p_qed = p[b_off + 1];
    let mut p_sas = p[b_off + 2];
    for (q, zq) in z.iter().enumerate() {
        p_logp += p[w_off + q] * zq;
        p_qed += p[w_off + 12 + q] * zq;
        p_sas += p[w_off + 2 * 12 + q] * zq;
    }
    [p_logp, p_qed, p_sas]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zinc_training_common::derive_pruned_params;
    use crate::EncodingKind;

    fn molecule(seed: usize) -> ZincMolecule {
        ZincMolecule {
            smiles: String::new(),
            features: (0..19)
                .map(|i| ((i * 7 + seed * 3) % 11) as f64 / 5.0 + 0.1)
                .collect(),
            log_p: 0.0,
            qed: 0.0,
            sas: 0.0,
        }
    }

    /// 111 parameters; qubit 11's readout weights are zero so dropping it changes nothing.
    fn full_params() -> Vec<f64> {
        let mut p: Vec<f64> = (0..111).map(|i| (i as f64 * 0.37).sin()).collect();
        for row in 0..3 {
            p[72 + row * 12 + 11] = 0.0;
        }
        p
    }

    fn predict(
        kernel: Kernel,
        params: &[f64],
        chunk: &[ZincMolecule],
        repeats: usize,
    ) -> Vec<[f64; 3]> {
        let encoding = DataEncodingLayer::new(12, EncodingKind::Amplitude);
        let ansatz = CircuitAnsatz::new(12, 3).with_pi_layer(true);
        Bench {
            ansatz: &ansatz,
            encoding: &encoding,
        }
        .run_chunk(kernel, params, chunk, repeats)
    }

    #[test]
    fn all_four_kernels_agree_on_the_same_model() {
        let molecules: Vec<ZincMolecule> = (0..4).map(molecule).collect();
        let full = full_params();
        let pruned = derive_pruned_params(&full, 12);
        let fused_98 = derive_pruned_params(&full, 11);
        assert_eq!((pruned.len(), fused_98.len()), (99, 98));

        let a = predict(Kernel::Full, &full, &molecules, 1);
        let others = [
            predict(Kernel::Pruned, &pruned, &molecules, 1),
            predict(Kernel::Fused12, &pruned, &molecules, 1),
            predict(Kernel::Fused11, &fused_98, &molecules, 1),
        ];
        for preds in &others {
            assert_eq!(preds.len(), a.len());
            for (x, y) in a.iter().zip(preds) {
                for k in 0..3 {
                    assert!((x[k] - y[k]).abs() < 1e-12, "{x:?} vs {y:?}");
                }
            }
        }
        assert!(
            a.iter().flatten().any(|v| v.abs() > 1e-3),
            "predictions should not be trivial"
        );
    }

    #[test]
    fn only_the_first_pass_keeps_predictions() {
        let molecules: Vec<ZincMolecule> = (0..3).map(molecule).collect();
        let full = full_params();
        assert_eq!(
            predict(Kernel::Full, &full, &molecules, 1),
            predict(Kernel::Full, &full, &molecules, 4)
        );
    }

    #[test]
    fn readout_uses_row_stride_12_for_every_width() {
        let mut p = vec![0.0; 108];
        p[95] = 0.5; // biases at b_off = 59 + 36
        p[96] = -1.0;
        p[97] = 2.0;
        p[59 + 2] = 3.0; // logP weight of qubit 2
        p[59 + 12 + 2] = 5.0; // QED weight of qubit 2
        p[59 + 24 + 2] = 7.0; // SAS weight of qubit 2
        let z = [0.0, 0.0, 2.0];
        assert_eq!(
            readout(&p, 59, 59 + 36, &z),
            [0.5 + 6.0, -1.0 + 10.0, 2.0 + 14.0]
        );
    }

    #[test]
    fn kernel_widths() {
        assert_eq!((Kernel::Full.n_z(), Kernel::Full.n_quantum()), (12, 72));
        assert_eq!((Kernel::Pruned.n_z(), Kernel::Pruned.n_quantum()), (12, 60));
        assert_eq!(
            (Kernel::Fused12.n_z(), Kernel::Fused12.n_quantum()),
            (12, 60)
        );
        assert_eq!(
            (Kernel::Fused11.n_z(), Kernel::Fused11.n_quantum()),
            (11, 59)
        );
    }
}
