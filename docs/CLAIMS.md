# Claims ledger

Where each headline number in the root `README.md` comes from, and how to regenerate it. The
numeric checks are automated: `python scripts/check_claims.py` scans the committed result files
for each README number and exits non-zero if one is missing.

Status legend

| Mark | Meaning |
| :--- | :--- |
| **Reproduced** | The number is present in a committed result file (path and key given). |
| **Re-run** | Not stored as a result file; produced by running the named binary (re-run on 2026-10-23). |
| **Differs** | The README number is not in the committed output; the committed output holds a different value from another run. |
| **Reference** | A tabulated reference value, not computed by the code that reports it. |

Timing figures (latency, throughput) are host-dependent. `results/hosts/<slug>/` holds per-host
runs; the root `results/` files come from the reference host.

## Reproduced from committed results

| README § | Claim | Source (`results/…`) |
| :--- | :--- | :--- |
| 4 | Linear XEB 1.0045 | `experiments/frontier_rcs_results.json` `porter_thomas.exact_xeb` = 1.00449 |
| 4 | KS D = 0.0021 (critical 0.0027) | same file, `porter_thomas.ks_statistic` = 0.002063, `ks_critical_value` = 0.002652 |
| 5 | MNIST 10-qubit VQC 84.50 % | `experiments/mnist_qnn_training.json` `final_test_acc_pct` |
| 5 | State coherence 350.94 | same file, `history[15].mean_coherence_l1` |
| 15 | Path A worst-group accuracy 61.91 %, G2 71.81 % | `experiments/path_a_vs_c_comparison_report.json` |
| 16 | QMUX coarse accuracy 70.69 % | `experiments/ultrametric_tree_sorter_report.json` `test_overall_coarse_accuracy` |
| 17 | Glued trees: classical 49.40 %, quantum peak 54.16 % at depth 10 | `experiments/glued_trees_report.json` `depths[8]` |
| 18 | Valve K=2 peak 77.16 % | `experiments/glued_tree_valve_report.json` `benchmarks_k2[0]` |
| 19 | Cascaded 2×2 peak 60.31 %, contrast 2,489:1 | `experiments/cascaded_tree_valve_report.json` |
| 20 | Topological walk joint accuracy 82.00 %, contrast 2,050:1 | `topological_walk_sorter_telemetry.json` `models[2]` |
| 21 | Nighthawk 20-hop fidelity 0.2119 → 0.6687 | `data/nighthawk_99_plaquette_telemetry.json` |
| 21 | Cepheus corner fidelity 0.8588 → 0.9146 | `data/cepheus_108q_telemetry.json` |
| 22 | Cepheus hardware exit probability 0.652 (652/1000), predictive accuracy 84.43 %, simulator 0.7722 | `experiments/cepheus_glued_trees_30q/cepheus_hardware_results.json` (job `ef387a4a-…`, raw counts in `cepheus_physical_1000shots_raw.json`); `data/cepheus_glued_trees_telemetry.json` |
| 27 | QEC threshold 8.14 %, 41.67× suppression at p = 0.001, 20.72 M decodes/s | `data/vnni_qec_threshold_report.json` |
| 28 | Coulomb eigenspectrum 14,352.5 mol/s; QM9 ingestion 4,056.4 mol/s | `data/molecular_eigensystem_report.json` |
| 29 | Ab initio VQE: H₂ FCI energy -1.13727; CH₄ error 0.1853 mHa; BeH₂ (largest) 0.3681 mHa | `hosts/unknown-cpu/data/vqe_molecules_report.json` `fci_energy`, `vqe_error_vs_fci` (full table in `docs/vqe.md`) |
| 29 | Noisy VQE: naive UCCSD CNOT counts 1,376 (LiH) and 36,480 (CH₄) | `hosts/unknown-cpu/data/vqe_noise_report.json` `resources[*].cnot` (full table in `docs/vqe.md`) |
| M158 | Szegedy walk sink probability 0.7351 | `data/directed_heptal_quantum_walk_report.json` |
| 4 | Noisy-RCS fidelity gain +5,147 % (14Q) and +13,543 % (16Q) | `rcs_physical_noise_chaos.json` `depth_sweep[*].fidelity_gain_pct` (see context below) |

## Re-run

| README § | Claim | How |
| :--- | :--- | :--- |
| 14 | CHSH witness S = 2.82842712, \|S − 2√2\| = 2.22e-15 across the cut bridge | `cargo run --release -p t1000 --bin cut_bridge_chsh_bench` reproduces both numbers exactly. The committed `experiments/chsh_qrack_ace_results.json` is a different, shot-based Qrack run (S = 2.8145, 2,048 shots per correlator). |

## Differs from committed output

| README § | README says | Committed output says | Note |
| :--- | :--- | :--- | :--- |
| 5 | 34.38 µs / sample | `experiments/mnist_qnn_training.json` `per_sample_latency_us` = 30.23 | Different runs; latency varies by host and load. Accuracy and coherence match. |
| 14 | 316.50 ms prefix, Δ = 0.0001056229 (36Q checkpointing) | `experiments/cut_bridge_36q_results.json`: total 218.33 ms, covariance delta 0.01255 | The README and `docs/findings/cut_bridge_36q_checkpointing.md` quote one run and the JSON holds another; the two have not been reconciled. |

## Reference values, not computed

* `chemistry::molecules` holds tabulated Hartree-Fock and ground-state energies per molecule (for
  example H₂ -1.116699 and -1.137270 Ha). `JordanWignerHamiltonian::from_molecule` builds its
  one-body and two-body terms from fixed formulas, not molecular integrals, and
  `ground_state_simulation` returns the tabulated energy. README § 3 ("ground-state energies for 9
  molecules") describes these tabulated values, in a basis and geometry that are not stated.
* The same nine molecules are now computed from their geometries in `t1000::vqe` (STO-3G, see
  `docs/vqe.md`). The two sets differ: the tabulated Hartree-Fock energies are -2.9154 (HeH⁺),
  -56.1654 (NH₃) and -40.1432 (CH₄), against -2.8414, -55.4540 and -39.7268 computed in STO-3G
  (the computed values match the published STO-3G energies). H₂ and H₂O are close (-1.116699 against
  -1.116684, -74.9654 against -74.9629). Use the `vqe` figures when a number must be reproducible.

## Context for comparisons

* **§ 5 MNIST.** 84.50 % is measured against the repository's own BNN and HDC baselines (78.6 % to
  81.3 %), not against a standard classical model on the same task.
* **§ 4 noisy RCS.** The two gain figures are the best entries of a depth sweep that also contains
  negative entries (−97.8 %, −72.8 %, −99.4 %, −18.4 %). They are gains relative to a very low
  baseline fidelity.
* **§ 3 ZINC and QM9.** The trained-circuit results are compared with classical ridge baselines in
  `docs/host_tuning_and_result_recording.md` § 5 and § 6: the circuit head matches quadratic ridge,
  re-uploading ties a degree-4 polynomial ridge, and neither shows an advantage over classical
  models. The trained 12-qubit circuit on ZINC has weak holdout correlation (logP Pearson r between
  −0.15 and 0.49 in the committed runs).

## Hardware check of the FPGA design (2026-10-23)

`scripts/fpga/verify_1000_qpu.py` against the `fpga/1000_qpu/build/1000_qpu.bit` bitstream on the
Basys 3 (`COM6`, 3 Mbaud):

| Check | Result |
| :--- | :--- |
| Ping `0xAA` → `0x10`, single layer `0x10` → `0xEE`, recovery after unknown opcodes | Pass (20 of 20). |
| Autonomous layers `0x20` | Acknowledged. Measured 8.48 µs per layer (difference of means, depth 255 vs 1), against the 10.24 µs expected from 1,024 cycles at 100 MHz; the 17 % gap is unexplained. |
| Entropy stream `0x30` | Streams at ~4,200 seeds/s as documented, but **every seed is the same counter `00 01 02 … 1f`**: 1 distinct seed in 128, 31 % ones, 5.0 bits/byte. The README's NIST SP 800-22 statement (§ 6, § 7 of `docs/fpga_architecture.md`) does not hold for this bitstream. |

Cause: `qpu_bram_matrix.v` has no initial contents and `qpu_top.v` has no opcode to load or read
back the statevector, so the layers act on an all-zero state and the entropy register only folds in
the harvest index. The handshake and timing are real; the computation is not yet meaningful.

## Not checked

The README quantities with no committed result file are not covered: FPGA figures (§ 6), the Q-SSM
token rates (§ 12), the MPS zipper and heptal-qudit timings (§§ 10, 11), compile/fusion speedups
(§§ 1, 2), and the speed comparison with Qiskit Aer. These come from benchmark binaries that print
to the terminal; re-run them on the target host to confirm.
