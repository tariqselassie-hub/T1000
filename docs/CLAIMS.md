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
| 5 | MNIST Hybrid QCNN 95.45 % | `experiments/mnist_hybrid_qnn_training.json` `final_test_acc_pct` |
| 5 | Hybrid State coherence 450.69 | same file, `history[15].mean_coherence_l1` |
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

**Legacy design (retired).** The first bitstream (opcodes `0xAA`, `0x10`, `0x20`, `0x30`) was
checked on the Basys 3 (`COM6`, 3 Mbaud): ping and single layer passed 20 of 20, 100-layer loops
were acknowledged at 8.48 µs per layer (expected 10.24 µs from 1,024 cycles at 100 MHz; the gap was
never explained), and the entropy stream `0x30` returned the same counter `00 01 … 1f` in every
seed (1 distinct seed in 128, 31 % ones). That design had no state load or read-back, so its layers
acted on an all-zero state, and the earlier README statements on NIST SP 800-22 compliance, the
169x layer speedup and the 26-qubit scaling did not hold. The entropy audit committed on
2026-10-15 agrees: 1,280,008 bits from 5,000 seeds, 31.3 % ones, 5.0 bits/byte, every NIST test
failed. The design, its scripts and its datasets were removed, and the web dashboard no longer
reports hardware figures it did not measure.

**Current design (`fpga/1000_qpu`, `docs/fpga_architecture.md`).** Evidence in the repository:

| Check | Result |
| :--- | :--- |
| Rust model (`t1000::qpu`), compiler to gates, protocol session | 18 unit tests pass, including the compiled Pauli rotations against the floating-point rotation. |
| Fixed-point precision (`qpu_precision`) | UCCSD energy error against double precision is at most 1.2e-4 mHa at 24 fractional bits for every molecule up to 14 qubits (chemical accuracy: 1.59 mHa); CH4 (16 qubits) does not fit. |
| RTL against the model in Vivado xsim (`sim/run_sim.py`) | PASS: 2,536 of 2,536 reply bytes equal the model's for the 13,577-byte golden stream. |
| Vivado 2026.1 implementation | Slack +0.372 ns at 100 MHz; 1,904 LUTs, 1,614 registers, 32 BRAM36, 40 DSP48E1. |
| Dashboard circuits on the board (`scripts/fpga/verify_dashboard_circuits.py`) | Bell pair, 12-qubit and 14-qubit RY/CZ/T/SWAP circuits (132 gates) run through `web/serve.py`: board norm 1.0, Pauli-Z from the board's state equals the host simulation and the board's `EXPECT_Z` to 1e-3 (agreement to 4 decimals in the first run). |
| Board (`scripts/fpga/verify_1000_qpu.py`, Basys 3 on `COM6`, 3 Mbaud, 2026-10-23) | PASS: the 2,536 reply bytes of the golden stream match the model, and so do the 5,885 reply bytes of the stream extended with the full LiH UCCSD circuit (3,152 gates, 127,095 stimulus bytes). 400 gates at 14 qubits took 121.9 µs each, against a 120.0 µs UART floor and 81.9 µs of fabric time. |

## Experiments from the 2026 literature scan (2026-10-23)

| Claim | Evidence |
| :--- | :--- |
| Qubit-wise commuting measurement groups cut the ideal-device shot budget for 1.59 mHa by 1.0x (H₂), 6.3x to 25.3x (HeH⁺ to CH₄); the grouped estimator reproduces the energy to 1e-13 | `hosts/unknown-cpu/data/vqe_grouping_report.json`; `cargo run --release -p t1000 --bin vqe_grouping`; table in `docs/vqe.md` |
| Quantum fidelity-kernel ridge classifier under nested cross-validation (48 combinations each): 1.3, 9.5 and 3.0 points below RBF on MNIST 1v7, 3v8, 4v9 (240 demo images each); 17.5 points above RBF on a planted quantum-kernel task; chance on shuffled labels | `hosts/unknown-cpu/data/quantum_kernel_bench.json`; `cargo run --release -p t1000 --bin quantum_kernel_bench`; `docs/quantum_kernel_benchmark.md` |
| First-order perturbative amplitudes (PT1) as the UCCSD starting point cut the starting energy error from 9.6-78 mHa (HF) to 0.01-2.7 mHa; six of nine molecules start inside 1.59 mHa; the optimizer saves 0-9 iterations (BH₃ takes 2 more); every start reaches the same energy | `hosts/unknown-cpu/data/vqe_warmstart_report.json`; `cargo run --release -p t1000 --bin vqe_warmstart`; table in `docs/vqe.md` |
| The analytic depolarising energy bias overstates sampled gate-level Pauli noise at small error rates by 1.4x to 1.5x (H₂, HeH⁺) and 5.3x to 5.6x (LiH, H₂O); the tolerable two-qubit error for 1.59 mHa is 1.4x to 5.6x higher than the analytic value (H₂ 3.1e-5, HeH⁺ 2.2e-5, LiH 2.7e-6, H₂O 2.2e-7) | `hosts/unknown-cpu/data/vqe_noise_check_report.json`; `cargo run --release -p t1000 --bin vqe_noise_check`; table in `docs/vqe.md`; standard errors 1 to 3 % of the bias |
| Fixed-point infidelity of the 1000_QPU arithmetic is about 0.07 · gates · 2^n · 4^(-fraction bits), measured on random circuits of 4 to 14 qubits | `hosts/unknown-cpu/data/qpu_scaling_report.json`; `cargo run --release -p t1000 --bin qpu_scaling`; fit over 51 cells, individual cells 0.01 to 0.09 |

## README audit (2026-10-23)

Every benchmark binary named in the README that needs no external data was rebuilt and re-run on the
reference host (Core Ultra 7 265, 20 threads) and compared with the README. Timings vary by 10 to 30 %
between runs, so a re-measured value within that band counts as reproduced. Column "Action" says what
the README now says.

| README § | Claim as it was | Re-run result | Action |
| :--- | :--- | :--- | :--- |
| 1 | Fused diagonal kernel 1.93x faster; adjoint pass 0.1002 s to 0.0921 s | Rademacher layer 1.74x to 1.84x (18Q); whole-pass effect not re-run | Qualified |
| 2 | Up to 1.9x faster than Qiskit Aer | No reproduction path: `frontier_rcs` refers to `scripts/aer_rcs_benchmark.py`, which is not in the repository. Against the scalar reference: 11.5x to 14.0x | Removed; scalar-reference speedup stated |
| 3 | 9 molecules' ground-state energies from Jordan-Wigner Hamiltonians | Tabulated reference values and fixed formulas, not integrals (see "Reference values") | Rewritten; points to § 29 |
| 3 | ZINC 250k in 1.17 s (213,351 molecules/s) | Needs the ZINC 250k table (not shipped) | Marked not re-run |
| 4 | Adjoint 1,296 parameters in 0.250 s | 0.113 s to 0.118 s; 47x to 61x faster than parameter shift at 12 qubits, max difference 3e-16 | Corrected |
| 4 | Dense coherence to 26 qubits, 64 qubits by tensor-network belief propagation, heptal 7^9 | Dense stage reproduces (26Q in 5.8 s). The tensor-network and heptal stages did not finish within 150 s | Qualified (approximate, not reproduced) |
| 4 | Noisy RCS fidelity gains +5,147 % and +13,543 % | Baseline fidelity 0.0004, stabilised 0.021 at 14Q; gains are relative to a near-zero baseline | Rewritten with absolute values |
| 5 | MNIST VQC "surpasses classical" baselines | Only the repository's BNN/HDC baselines. On the same split: logistic regression 87.0 %, 3-nearest-neighbours 92.35 %, VQC 84.50 % | Rewritten |
| 7, 27 | VNNI syndrome 26.11 ns, 38.30 M/s; commutation 34.81 M/s; decode 48.2 ns, 20.72 M/s | 27.0 ns, 37.0 M/s; 33.9 M/s; 41 to 58 ns, 21.1 M/s | Reproduced within host variance |
| 27 | "Fault-tolerant threshold" 8.14 % | Number reproduces (8.14 %, 41.67x at p = 0.001, P_L = 24 p^2). It is the crossing with nine unencoded qubits (about 6p); against one qubit (2p/3) it is near 5 %. Code-capacity noise, one code distance | Renamed pseudo-threshold, scope stated |
| 10 | 200 qubits in 105.8 us | 22.1 us (window 3), 47.2 us (window 4). The benchmark measures anti-correlation of neighbours, not entanglement | Corrected |
| 11 | 200 qudits in 7.58 ms | 4.88 ms; only a 49-amplitude window is simulated | Corrected |
| 12 | 66.6 to 95.8 ns/token; 100 % accuracy | 64 to 84 ns (11.9 to 15.6 M tokens/s). 100 % is a synthetic coherent-versus-chaotic task; the quench variant gives 70.8 % to 100 % | Corrected |
| 14 | 36Q checkpoint 316.50 ms, delta 0.0001056229, 1.608 s | 318.46 ms, 0.0001056229, 1.607 s: reproduces (the committed JSON is a different run) | Reproduced; "Differs" entry removed |
| 14 | CHSH S = 2.82842712, deviation 2.22e-15 | Reproduces (kernel 29.6 ms against 35.86 ms) | Reproduced |
| 14 | iGPU 2.13x faster than CPU at N >= 22 | 1.8x at 22 qubits (9.5 ms against 17.4 ms); slower than the CPU below 20 qubits | Corrected |
| 17 to 19 | Glued trees 49.40 % / 54.16 %; valve 77.16 %, contrast about 1,700:1; cascaded 60.31 %, 2,489:1 | All reproduce. The classical baseline is a random walker capped at 10,000 steps; "43.1x speedup" has no stated definition | Interpretation corrected; speedup figure removed |
| 21 | Nighthawk 0.2119 to 0.6687; Cepheus corner 0.8588 to 0.9146 | Reproduce; they are noise-model estimates with assumed error rates | Labelled as modelled |
| 22 | "Empirical quantum advantage > 39x"; "84.43 % predictive accuracy" | The recorded numbers are real (652/1000 on `q7`, job `ef387a4a-...`). The eight wire marginals sum to 3.49 (0.478, 0.427, 0.378, 0.240, 0.269, 0.382, 0.663, 0.652), so the measured state is nearly random, not a single walker; `q7` (65.2 %) is no higher than `q6` (66.3 %). The 1.67 % comparison is a single-walker quantity | Reworded: no advantage or ballistic transport is claimed |
| 23, 24 | QGNN "barren-plateau resilience", variance 4.2931e-2 at 14Q | Computed, but identical at 4 to 14 qubits (local observable, shared weights) and smaller than the hardware-efficient baseline's 0.08 to 0.16 at every size | Reworded; not evidence of resilience |
| 24 | Adjoint "153.6x speedup over parameter shift" | The benchmark computed it as `2 * parameters * 0.8` and then crashed on a wrong workspace size. Fixed: measured 47x to 61x. The chart that hard-coded the curves was removed | Corrected; chart deleted |
| 26 | Lattice surgery truth table, Bell fidelity 1.0, 15-to-1 distillation 4.365e-6 at 0.5 % | Reproduce (23.4 ms against 25.15 ms; epsilon_out 4.3653e-6) | Reproduced |
| 28 | Eigensystem 14,352.5 mol/s, QM9 4,056.4 mol/s | The benchmark evaluates 0 molecules without the roboBohr and QM9 files | Marked not reproducible from the repository |
| 25 | Conway QCA | `qgol_sim` runs; its numbers were not compared | Not checked |

Unit tests: 394 library tests pass (`cargo test --release --workspace`).

## Lab bench (2026-10-23)

The dashboard was rebuilt as a bench (`web/index.html`); the old decks moved to `web/archive.html`. Two old
endpoints returned results that were not computed (`/api/canonical/run`: hard-coded Bernstein-Vazirani
probabilities and a uniform Grover distribution; `/api/smiles/evaluate`: invented logP, QED and SAS formulas)
and were removed.

| Component | Check |
| :--- | :--- |
| Number theory (`web/lab_nt.js`) | `node web/test_lab_nt.js`: 12 test groups against known values (for example 600851475143 = 71 x 839 x 1471 x 6857, 2^64 + 1 = 274177 x 67280421310721, order of 2 mod 9 = 6, Pisano periods 60, 24 and 16, chi-square tail values), Euler's criterion for every odd prime below 100, and the digit-root facts over 3,000 integers. |
| Geometry (`web/lab_geom.js`) | `node web/test_lab_geom.js`: the Hopf map returns the Bloch vector, every point of a fibre maps to one base point, two fibres have linking number 1 (numerically 1.00 +/- 0.05), polytope vertex and edge counts (5, 16, 8, 24 vertices; 10, 32, 24, 96 edges), V - E + F - C = 0, 4D rotations are orthogonal. |
| Algorithms and probability (`web/lab_api.py`) | `python web/test_lab_api.py`: 33 checks. Grover equals sin^2((2k+1) theta) at every step (to 1e-15); Bernstein-Vazirani and Deutsch-Jozsa are certain; phase estimation is exact for representable phases and within 1/2^t otherwise with probability at least 4/pi^2; the FFT construction of order finding equals an explicit QFT matrix; 7 mod 15 has four equal peaks at multiples of 64 and yields the factors 3 and 5; quantum walk return probability is cos^2(t) on two nodes and the classical one (1 + exp(-2t))/2; p-values of repeated null experiments are uniform. |

Limits stated in the interface: the order-finding simulation builds the state after the modular
exponentiations directly (equal to the ideal circuit's output, not a gate-level multiplier circuit); walks and
algorithms run on the host in exact arithmetic, only the circuit composer can use the board (up to 14 qubits).

## Not checked

* Claims whose data files are not shipped: ZINC 250k screening speeds (§ 3), the QM9 and roboBohr
  throughputs (§ 28), the Waterbirds sorter results (§§ 15, 16; they match the committed result JSON, but the
  runs were not repeated), and the QM9 training monographs.
* § 8 (canonical algorithms), § 9 (scheduler), § 13 (Clifford+T): covered only by unit tests.
* § 25 (Conway QCA) numbers and § 26 timings.
* The findings documents under `docs/findings/`: 26 now carry an audit note at the top (second pass,
  2026-10-23) and the QGNN one was rewritten. The remaining ones (Hopf S4, directed heptal walk, the QCA
  documents, ZINC and QM9 screening, plaquette zipper) were not re-run or re-read for wording.
* The chart scripts under `scripts/charting/` read result files and were not re-checked individually, except
  that two were removed because their curves were hard-coded: `chart_full_qnn_suite.py` (adjoint scaling) and
  `chart_exascale_showdown.py` (a desktop-against-supercomputer comparison with typed-in numbers that compared
  a dense simulation with a sliding-window method that does not represent the same state).
