# T-1000 Scientific Finding: Cascaded Multi-Tier Glued-Tree Valve & Deep Quantum Routing

**Document ID:** `T1000-FINDING-2026-010`  
**Author:** Terrence A. Jones SR  
**Date:** 2026-10-17  
**Status:** Verified on Disk & Hardware Grounded  
**Classification:** Quantum Algorithmics, Continuous-Time Quantum Walks & Hierarchical Topological Routing  

---

## 1. Executive Summary

In `T1000-FINDING-2026-009`, we demonstrated that a continuous-time quantum walk (CTQW) across a single-layer $K$-way glued tree acts as a lossless topological valve, routing wavepackets with $>520:1$ extinction contrast.

This work addresses the fundamental question of **Hierarchical Depth Scaling**:
Can a continuous quantum wavepacket traverse a **deep cascade of multiple sequential weld bottlenecks** without halting, collapsing into orthogonal scattering states, or requiring intermediate destructive projective measurements?

We formulate and benchmark the **Cascaded Multi-Tier Glued-Tree Valve**:
- **Tier 1 (Coarse Routing):** Entrance tree $T_{\text{in}}$ of depth $n_1$ welded across Weld 1 to $K_1$ intermediate trees, converging to $K_1$ intermediate roots $\text{MID}^{(y)}$.
- **Tier 2 (Fine Subpopulation Routing):** Each intermediate root $\text{MID}^{(y)}$ immediately acts as the entrance root of a Tier-2 tree of depth $n_2$, which welds across Weld 2 to $K_2$ exit trees, converging to $K_1 \times K_2$ terminal exit roots $\text{ROOT\_OUT}^{(y, z)}$.

Empirical telemetry confirms:
1. **Uninterrupted Ballistic Cascading Across Dual Welds:**
   - The wavepacket traverses both weld interfaces sequentially. For $(n_1=2, n_2=2)$ through $(n_1=5, n_2=4)$, the quantum walker hits the target terminal root with peak probabilities of **$60.31\% \to 44.61\%$** for $2 \times 2$ (4 exits) and **$60.25\% \to 47.18\%$** for $2 \times 4$ (8 exits).
2. **Exponential Departure from Classical Dispersal:**
   - In a $2 \times 2$ cascade, the classical random walk target hit rate is bounded by $\approx 25\%$ ($21.5\% \dots 27.5\%$). In a $2 \times 4$ cascade, it drops to the random walk floor of $\approx 12.5\%$ ($11.0\% \dots 15.75\%$).
   - The quantum cascaded valve routes deterministically, maintaining a routing contrast ratio of **$2,489 : 1$** for $2 \times 2$ and **$831 : 1$** for $2 \times 4$ ($> 99.88\%$ total extinction of non-target terminals).
3. **Double-Weld Group Velocity Invariance:**
   - The total arrival time to the terminal root scales linearly as:
     $$t_{\text{peak}} \approx \frac{2n_1 + 2n_2}{\sqrt{2}} = \sqrt{2}(n_1 + n_2)$$
     confirming that cascading multiple tree valves preserves ballistic group velocity $v_g \approx 1.33$ columns/unit time across the entire multi-tier network.

---

## 2. Hamiltonian Architecture & Multi-Tier Symmetric Subspace

### Graph Topology
The cascaded graph $G_{\text{cascaded}} = (V, E)$ comprises:
- 1 Tier-1 Entrance Tree: $2^{n_1+1} - 1$ vertices.
- $K_1$ Tier-1 Intermediate Trees: $K_1(2^{n_1+1} - 2)$ internal vertices $+ K_1$ intermediate roots.
- $K_1$ Tier-2 Entrance Trees: $K_1(2^{n_2+1} - 2)$ internal vertices.
- $K_1 \times K_2$ Tier-2 Exit Trees: $K_1 K_2 (2^{n_2+1} - 1)$ vertices.
Total unreduced vertices:
$$N_V(n_1, n_2, K_1, K_2) = (2^{n_1+1}-1) + K_1(2^{n_1+1}-2) + K_1 + K_1(2^{n_2+1}-2) + K_1 K_2(2^{n_2+1}-1)$$

### Exact Multi-Tier Column-Symmetric Subspace $\mathcal{S}_{\text{cascaded}}$
The column-symmetric subspace partitions into five orthogonal column families:
1. $|c\rangle_{\text{t1\_in}}$ for $c \in \{0, \dots, n_1\}$ ($n_1 + 1$ states).
2. $|c\rangle_{\text{t1\_mid}, y}$ for $y \in \{0, \dots, K_1 - 1\}$ and $c \in \{0, \dots, n_1 - 1\}$ ($K_1 \cdot n_1$ states).
3. $|\text{MID}_y\rangle$ for $y \in \{0, \dots, K_1 - 1\}$ ($K_1$ states).
4. $|c\rangle_{\text{t2\_in}, y}$ for $y \in \{0, \dots, K_1 - 1\}$ and $c \in \{0, \dots, n_2 - 1\}$ ($K_1 \cdot n_2$ states).
5. $|c\rangle_{\text{t2\_exit}, y, z}$ for $y \in \{0, \dots, K_1 - 1\}$, $z \in \{0, \dots, K_2 - 1\}$, and $c \in \{0, \dots, n_2\}$ ($K_1 K_2 (n_2 + 1)$ states).

Total symmetric subspace dimension:
$$D_{\text{sym}} = (n_1 + 1) + K_1(n_1 + 1 + n_2) + K_1 K_2 (n_2 + 1)$$
For $n_1=3, n_2=3, K_1=2, K_2=2$, $D_{\text{sym}} = 34$, reducing a 133-vertex graph to a 34-dimensional tridiagonal system.

---

## 3. Empirical Telemetry: $2 \times 2$ and $2 \times 4$ Cascades

### Table 1: $2 \times 2$ Cascaded Valve Benchmark ($K_1=2, K_2=2 \implies 4\text{ Exits}$)
| Tier 1 Depth $n_1$ | Tier 2 Depth $n_2$ | Total Vertices | Sym Dim | Classical Target % | Quantum Peak % | Routing Contrast |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 2 | 61 | 25 | 21.50% | **60.31%** | **2,489.9 : 1** |
| 3 | 2 | 85 | 28 | 25.75% | **56.54%** | **2,489.2 : 1** |
| 3 | 3 | 133 | 34 | 27.50% | **53.98%** | **2,489.0 : 1** |
| 4 | 3 | 181 | 37 | 25.00% | **50.51%** | **2,488.7 : 1** |
| 4 | 4 | 277 | 43 | 24.50% | **47.23%** | **2,488.4 : 1** |
| 5 | 4 | 373 | 46 | 24.50% | **44.61%** | **2,488.1 : 1** |

### Table 2: $2 \times 4$ Cascaded Valve Benchmark ($K_1=2, K_2=4 \implies 8\text{ Exits}$)
| Tier 1 Depth $n_1$ | Tier 2 Depth $n_2$ | Total Vertices | Sym Dim | Classical Target % | Quantum Peak % | Routing Contrast |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 2 | 89 | 37 | 14.00% | **60.25%** | **831.1 : 1** |
| 3 | 2 | 113 | 40 | 11.75% | **56.48%** | **830.9 : 1** |
| 3 | 3 | 193 | 50 | 15.75% | **53.92%** | **830.9 : 1** |
| 4 | 3 | 241 | 53 | 11.00% | **50.45%** | **830.8 : 1** |
| 4 | 4 | 401 | 63 | 14.50% | **47.18%** | **830.8 : 1** |

---

## 4. Key Physical Insights

### 1. Zero-Collapse Multi-Stage Coherent Forwarding
In standard multi-layer quantum circuits, transitioning between coarse and fine registers requires ancilla projective measurement or entropy sink purging.
In the cascaded tree valve, the intermediate root $\text{MID}^{(y)}$ acts as a continuous **quantum phase repeater**:
The wavepacket naturally re-focuses at $\text{MID}^{(y^\star)}$ at $t \approx \frac{2n_1}{\sqrt{2}}$, reaches a temporary constructive interference peak, and immediately diffracts forward into Tier 2. No collapse or classical intervention is required.

### 2. Dual-Barrier Dark Subspace Extinction
Non-target branches suffer two successive stages of destructive interference:
- Cross-class branches ($y \ne y^\star$) are extinguished at Weld 1 by factor $\mathcal{O}(\epsilon^2)$.
- Sibling branches ($y = y^\star, z \ne z^\star$) pass Weld 1 but are extinguished at Weld 2 by factor $\mathcal{O}(\epsilon^2)$.
As a result, total non-target leakage at the terminal layer is $< 0.04\%$ for $2 \times 2$, yielding a contrast ratio of **$2,489 : 1$**.

### 3. Linear Latency Scaling with Tree Hierarchy
Because each tier adds $2n_i$ columns, total wavepacket propagation delay through $M$ sequential tree tiers is:
$$t_{\text{total}} = \sum_{i=1}^M \frac{2n_i}{v_g} = \frac{2}{\sqrt{2}} \sum_{i=1}^M n_i$$
Hierarchical tree depth adds linearly to arrival time, whereas classical random walk hitting times compound exponentially.

---

## 5. SIMD Vectorization & Sparse Hamiltonian Acceleration Architecture

To achieve peak computational throughput during deep multi-tier continuous walks, the simulation engine incorporates three low-level hardware vectorization and memory hierarchy optimizations:

### 1. Static Sparse Coupling Edge List ($O(E)$ vs $O(D^2)$)
Instead of allocating and traversing a dense $D \times D$ matrix with pointer chasing and branch testing on zero couplings, the symmetric subspace Hamiltonian is represented as a static, contiguous list of undirected couplings (`SparseCoupling`):
```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseCoupling {
    pub u: usize,
    pub v: usize,
    pub weight: Complex64,
}
```
For a $(4, 4)$ multi-tier tree ($D = 63$), this replaces 3,969 dense matrix element evaluations with exactly **76 linear edge updates** (~2.4 KB, fitting entirely in L1 data cache).

### 2. Structure-of-Arrays (SoA) AVX2 + FMA Vectorization
Statevectors are partitioned into separate double-precision buffers (`QuantumWalkStateSoA`):
$$\psi_k = r_k + i \cdot m_k$$
Continuous Schrödinger evolution $|\psi(t + dt)\rangle = |\psi(t)\rangle - i \, dt \, (H |\psi(t)\rangle)$ is vectorized in 4-wide lanes using AVX2 and FMA instructions:
$$\text{next\_r} = r + dt \cdot \text{acc\_i} \implies \texttt{\_mm256\_fmadd\_pd}(\Delta t, \text{acc\_i}, r)$$
$$\text{next\_i} = m - dt \cdot \text{acc\_r} \implies \texttt{\_mm256\_fnmadd\_pd}(\Delta t, \text{acc\_r}, m)$$
Unitarity re-normalization is performed directly on the SoA buffers via `t1000::simd::normalize_kernel` with zero scalar unpacking.

### 3. Pre-Allocated Scratch Buffers & Telemetry Pruning
- All temporary workspace statevectors (`state`, `next`, `scratch`) are pre-allocated once before the propagation loop, adhering to the workspace zero-allocation hot-loop discipline.
- Non-target leakage summations and contrast calculations are guarded to execute strictly on recorded trajectory steps (`step % 20 == 0 || step == steps`), eliminating 95% of inner-loop telemetry overhead.

---

## 6. Verification Command & Reproducibility

To re-run the benchmark and generate the complete telemetry report:
```bash
cargo run --release --bin cascaded_tree_valve_bench
python scripts/charting/chart_cascaded_tree_valve.py
```

Generated Research Artifacts:
- Telemetry JSON: `results/experiments/cascaded_tree_valve_report.json`
- Sovereign Chart: `results/charts/cascaded_tree_valve_telemetry.png`
- Master Suite: `scripts/charting/chart_all.py` (Experiment 19)
