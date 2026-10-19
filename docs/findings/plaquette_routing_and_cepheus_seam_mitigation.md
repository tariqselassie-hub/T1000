# T1000-FINDING-2026-011: 2D Plaquette Routing, Rigetti Cepheus-1-108Q Seam Mitigation & Monolithic IBM Nighthawk R2 Showdown

**Author:** Terrence A. Jones SR  
**Date:** 2026-10-18  
**Registry ID:** `T1000-FINDING-2026-011`  
**Subsystems:** `t1000::plaquette::routing`, `t1000::density_matrix`, `t1000::teleportation`, `t1000::pauli_frame`  
**License:** MIT OR Apache-2.0  

---

## 1. Executive Summary & Problem Formulation

Two-dimensional planar superconducting quantum architectures face a fundamental routing bottleneck: entangling diagonally separated qubits ($Q_1$ at top-left to $Q_4$ at bottom-right of a $2 \times 2$ plaquette) requires multi-hop gate chains. 

In standard compilation stacks, diagonal routing is executed via sequential SWAP cascades:
$$Q_1 \xrightarrow{\text{SWAP}} Q_2 \xrightarrow{\text{SWAP}} Q_4$$
This incurs 6 to 9 noisy two-qubit gates (3 CNOTs per SWAP) and prolonged idle decoherence on spectators. Across a large processor grid—such as the 120-qubit **IBM Quantum Nighthawk R2** ($12 \times 10$ monolithic grid) or the 108-qubit **Rigetti Cepheus-1-108Q** ($12 \times 9$ modular array of twelve $3 \times 3$ chiplets)—multi-hop sequential SWAP paths degrade Bell pair fidelity below the classical threshold ($F < 0.50$).

This finding formulates, implements, benchmarks, and visualizes an exact mitigation:
1. **$1 + (2,3) = 4$ Dual-Bridge Routing:** Utilizing both intermediate vertices ($Q_2$ and $Q_3$) as parallel spatial mediators.
2. **Measurement-Based Plaquette Teleportation:** Entangling all four bounding edges with 1 CNOT each, measuring the bridge qubit ($Q_2$) in the $X$-basis to teleport the Bell state directly across the diagonal.
3. **Active Software Pauli-Frame Recovery:** Resolving the $X$-basis syndrome $s_2 \in \{0, 1\}$. Rather than post-selecting out the $s_2 = 1$ ($|\Phi^-\rangle$) subspace (which halves throughput to $48.4\%$), a classical software phase frame $Z_1$ is applied with **0 gate delay**, restoring $|\Phi^+\rangle$ and lifting acceptance to **$96.8\%$** on monolithic lattices.
4. **Inter-Die Seam Penalty Mitigation on Cepheus-1-108Q:** On modular multi-chiplet lattices where inter-die bump bonds exhibit elevated error rates ($p_{\text{seam}} = 0.0220$ vs $p_{\text{intra}} = 0.0080$), constant-depth 4-edge teleportation insulates the circuit from cascading gate errors, providing a **$+5.58\%$ absolute purification boost** at 4-chiplet corner intersections.

---

## 2. Plaquette Hamiltonian & Quantum Evolution Protocols

Consider an elementary $2 \times 2$ plaquette with qubits $\{Q_1(0), Q_2(1), Q_3(2), Q_4(3)\}$ arranged on a 2D square grid:
```text
      (Edge 1,2)
  Q1 (0) ─────── Q2 (1)
    │              │
    │ (Edge 1,3)   │ (Edge 2,4)
    │              │
  Q3 (2) ─────── Q4 (3)
      (Edge 3,4)
```

The system is evolved under full open-system density matrix dynamics ($\rho \in \mathbb{C}^{16 \times 16}$):
$$\rho_{k+1} = \mathcal{E}_{\text{idle}} \circ \mathcal{E}_{\text{depol}} \circ \mathcal{U}_g(\rho_k)$$
where each CNOT on edge $(u, v)$ is followed by single-qubit depolarizing channels of strength $p_{uv}$, and inactive qubits undergo idle dephasing $p_{\text{idle}}$.

### Protocol Suite (1 through 6)

1. **Protocol 1: Sequential SWAP ($1 \to 2 \to 4$)**
   - Applies $H_0 \to \text{CNOT}(0, 1)$, followed by a 3-CNOT bidirectional SWAP between $Q_1$ and $Q_3$ (or $Q_2$ and $Q_4$).
   - Yields $F = 0.9253$ on single plaquettes, collapsing to $F = 0.2119$ across 20 diagonal hops.
2. **Protocol 2: Dual-Path Raw ($1 \to 2$ and $1 \to 3$, No Post-Selection)**
   - Distributes entanglement symmetrically across both upper and lower boundaries without uncomputing mediators.
   - Yields $F = 0.9297$ ($0\%$ post-selection rejection).
3. **Protocol 3: Dual-Path + Syndrome Check**
   - Entangles $1 \to 2$ and $1 \to 3$, checks parity, and projects on $s_2 = 0, s_3 = 0$.
   - Yields $F = 0.9605$ with $94.8\%$ acceptance.
4. **Protocol 4: True Dual-Stabilizer ($Q_2 = X_1 X_4, Q_3 = Z_1 Z_4$)**
   - Interleaved schedule measuring $X_1 X_4$ on $Q_2$ and $Z_1 Z_4$ on $Q_3$.
   - Yields $F = 0.9605$ with $48.4\%$ acceptance (due to uncompensated $s_2 = 1$ projection).
5. **Protocol 5: Symmetric Dual-Bridge (Uncompute + Parity Verification)**
   - Uncomputes $Q_2$ via $Q_1$, verifies $Z$-parity on $Q_3$.
   - Yields $F = 0.9606$ with $94.8\%$ acceptance.
6. **Protocol 6: 4-Edge Plaquette Teleportation + Active Pauli Frame**
   - Applies exactly 1 CNOT per lattice edge: $(1, 0), (1, 3), (0, 2), (3, 2)$, followed by $H_1$ to close the $X$-basis check.
   - Active Pauli frame decoding:
     $$s_2 = 0 \implies |\Phi^+\rangle, \quad s_2 = 1 \implies |\Phi^-\rangle \xrightarrow{\text{Classical } Z_1} |\Phi^+\rangle$$
   - Yields $F = 0.9606$ while recovering the full **$96.8\%$ valid subspace** at zero gate delay.

---

## 3. Physical Origin of the $\sim 0.9605$ Fidelity Ceiling

Across Protocols 3, 4, 5, and 6, the post-selected Bell fidelity strictly saturates at $0.9605\text{--}0.9606$ under $p_{\text{gate}} = 0.010, p_{\text{idle}} = 0.005$.

The underlying physical mechanics:
1. **$Z$-Parity Syndrome Blindness:** Checking parity $s_3 = 0$ in the computational ($Z$) basis detects $100\%$ of single bit-flip ($X$) and bit-phase ($Y$) errors on mediators, but is completely blind to phase drift ($Z$-errors) on diagonal targets $(Q_1, Q_4)$.
2. **Eigenstate Projection in Protocol 4:** When $Q_1, Q_4$ start in $|00\rangle$, the system is an eigenstate of $Z_1 Z_4 = +1$ (so $Q_3$ reads 0). Measuring $X_1 X_4$ projects the state into $|\Phi^+\rangle$ ($s_2 = 0$) or $|\Phi^-\rangle$ ($s_2 = 1$) with equal $50/50$ probability.
3. **Classical Pauli Frame Recovery:** In Protocol 6, acknowledging that $|\Phi^-\rangle = (\mathbb{I} \otimes Z)|\Phi^+\rangle$ allows the quantum compiler to update its classical Pauli frame table in $O(1)$ CPU time. No physical pulse is scheduled; subsequent measurement bases are inverted in software. This doubles usable throughput ($48.4\% \to 96.8\%$) without decreasing state fidelity.

---

## 4. Multi-Chiplet Architecture Analysis: Rigetti Cepheus-1-108Q

Rigetti Cepheus-1-108Q scales to 108 qubits by tiling twelve $3 \times 3$ integrated dies across a $4 \times 3$ grid:

```text
┌───────────────┬───────────────┬───────────────┬───────────────┐
│ Die 0 (3x3)   │ Die 1 (3x3)   │ Die 2 (3x3)   │ Die 3 (3x3)   │
├───────────────┼───────────────┼───────────────┼───────────────┤
│ Die 4 (3x3)   │ Die 5 (3x3)   │ Die 6 (3x3)   │ Die 7 (3x3)   │
├───────────────┼───────────────┼───────────────┼───────────────┤
│ Die 8 (3x3)   │ Die 9 (3x3)   │ Die 10 (3x3)  │ Die 11 (3x3)  │
└───────────────┴───────────────┴───────────────┴───────────────┘
```

### Coupler and Plaquette Partitioning

* **Total Qubits:** $12 \times 9 = 108$
* **Total Couplers:** $195$
  * Intra-Die Couplers ($p_{\text{intra}} = 0.0080$): $144$
  * Inter-Die Seam Couplers ($p_{\text{seam}} = 0.0220$): $51$ ($27$ vertical, $24$ horizontal)
* **Total $2 \times 2$ Plaquettes:** $11 \times 8 = 88$
  * **Intra-Die Plaquettes:** $48$ tiles completely contained within individual $3 \times 3$ dies.
  * **Seam-Straddling Plaquettes:** $34$ tiles crossing a single inter-die boundary.
  * **4-Chiplet Corner Plaquettes:** $6$ tiles ($P_{24}, P_{27}, P_{30}, P_{57}, P_{60}, P_{63}$) centered on 4-way die intersections where all 4 edges cross inter-die seams.

---

## 5. Empirical Showdown: Monolithic vs. Modular Architecture

The benchmarks were executed using `crates/t1000/src/bin/benchmarks/nighthawk_99_plaquette_bench.rs` and `crates/t1000/src/bin/benchmarks/cepheus_108q_bench.rs`:

```text
┌───────────────────────────────────┬──────────────┬──────────────────┬──────────────────┐
│ Architecture / Plaquette Zone     │ SWAP (1>2>4) │ 1+(2,3)=4 Bridge │ 4-Edge Teleport  │
├───────────────────────────────────┼──────────────┼──────────────────┼──────────────────┤
│ Cepheus Intra-Chiplet (48 Tiles)  │ 0.9368       │ 0.9683           │ 0.9683 (Acc 97.4%)│
│ Cepheus Seam-Straddle (34 Tiles)  │ 0.8977       │ 0.9414           │ 0.9414 (Acc 94.6%)│
│ Cepheus 4-Die Corner (6 Tiles)    │ 0.8588       │ 0.9146           │ 0.9146 (Acc 91.8%)│
├───────────────────────────────────┼──────────────┼──────────────────┼──────────────────┤
│ CEPHEUS-108Q OVERALL (88 Tiles)   │ 0.9164       │ 0.9543           │ 0.9543 (Acc 96.2%)│
│ IBM NIGHTHAWK R2 (Monolithic 99)  │ 0.9253       │ 0.9606           │ 0.9606 (Acc 96.8%)│
└───────────────────────────────────┴──────────────┴──────────────────┴──────────────────┘
```

### Global 20-Hop Cross-Chip Diagonal Routing Showdown

On IBM Nighthawk R2, diagonal traversal from $Q_0(0,0)$ to $Q_{119}(11,9)$ requires crossing 10 diagonal plaquettes (20 rectilinear hops):

| Traversal Strategy | Gate Depth | Idle Latency | Memory Footprint | End-to-End Bell Fidelity | Operational Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Sequential SWAP Cascade** | 60 CNOTs | 60 Idle Cycles | $2.12 \times 10^{25}\text{ PB}$ (Dense OOM) | **0.2119** | **Decoherence Collapse ($< 0.50$)** |
| **4-Edge Plaquette Teleport** | 20 Ticks | 0 Idle Cycles | $384\text{ Bytes}$ (L1 Cache) | **0.6687** | **Preserved ($+45.7\%$ Boost)** |

---

## 6. Verification & Telemetry Deliverables

1. **Standalone Binaries:**
   - `crates/t1000/src/bin/benchmarks/plaquette_1234_bench.rs` (Elementary 4Q density matrix protocols).
   - `crates/t1000/src/bin/benchmarks/nighthawk_99_plaquette_bench.rs` (99-plaquette monolithic sweep).
   - `crates/t1000/src/bin/benchmarks/cepheus_108q_bench.rs` (88-plaquette modular multi-chiplet sweep).
2. **Serialized Telemetry:**
   - `results/data/nighthawk_99_plaquette_telemetry.json` (79.8 KB).
   - `results/data/cepheus_108q_telemetry.json` (52.0 KB).
3. **Publication Figures:**
   - `results/charts/nighthawk_99_plaquette_showdown.png` (1.44 MB 4-panel visual).
   - `results/charts/cepheus_108q_showdown.png` (1.15 MB 4-panel visual).
4. **Live Web Dashboard:**
   - Interactive dual-architecture switcher and plaquette inspector in `web/index.html` and `web/app.js`.
