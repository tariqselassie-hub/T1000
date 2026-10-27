# Benchmark III — DFTHC-SOS Electronic Structure Block Encoding
## Hamiltonian Factorization, QROM Bottlenecks & Subtree Pruning

SPDX-License-Identifier: MIT OR Apache-2.0  
Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

---

### 1. Problem Specification

Benchmark III evaluates the block encoding of the FeMo-co active space Hamiltonian ($N = 76$ spatial orbitals, $\eta = 113$ electrons) in the Double Factorized Tensor Hypercontraction (DF-THC) representation:

$$H = \sum_{p,q} h_{pq}^{(1)} E_{pq} + \frac{1}{2} \sum_{p,q,r,s} h_{pqrs}^{(2)} E_{pq} E_{rs}$$

#### The Optimization Scope
The block encoding consists of:
1. `Outer`: Multiplexed state preparation ($846$ Toffolis).
2. `Inner`: State preparation over two-body basis vectors ($9,461$ Toffolis).
3. `Givens Rotations`: $2(N - 1)(2\beta - 1) = 3,450$ Toffolis.
4. **`ANGLE_k` Circuits (The Bottleneck):** A family of $K = \lceil 75 / J \rceil$ circuits loading $75$ Givens rotation angle words for each vector address $a \in [0, 3975]$.

---

### 2. The 79,480 Toffoli Bottleneck

In the reference baseline ($J = 8$, $K = 10$ circuits):

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DFTHC-SOS TOFFOLI COST BREAKDOWN                     │
├─────────────────────────────────────────┬───────────────┬──────────────┤
│ Subsystem Component                     │ Toffoli Count │ Percentage   │
├─────────────────────────────────────────┼───────────────┼──────────────┤
│ 10 ANGLE_k QROM Circuits (Forward Pass) │ 39,740        │ 42.6%        │
│ 10 ANGLE_k QROM Circuits (Reverse Pass) │ 39,740        │ 42.6%        │
│ Inner State Preparation                 │ 9,461         │ 10.1%        │
│ Givens Rotations (2 passes)             │ 3,450         │ 3.7%         │
│ Outer State Preparation                 │ 846           │ 0.9%         │
├─────────────────────────────────────────┼───────────────┼──────────────┤
│ Total Toffoli Cost (T_total)            │ 93,237        │ 100.0%       │
└─────────────────────────────────────────┴───────────────┴──────────────┘
```

**Over $85.2\%$ of all Toffolis ($79,480$ of $93,237$) come strictly from the $10$ `ANGLE_k` QROM tables.**

---

### 3. The 20 mHa Frobenius Barrier & Precision Bounds

Using the Docker testbed on the reference integrals (`FeMoCo76o113e.fcidump`), we evaluated the DF-THC metrics across precision levels:

```
┌────────────────────────────────────────────────────────────────────────┐
│                  PRECISION TRUNCATION METRIC PROFILE                   │
├─────────┬──────────┬──────────┬──────────┬─────────┬─────────┬─────────┤
│ brot    │ error_1  │ error_2  │ Lambda   │ E_gap   │ lambda_e│ Status  │
├─────────┼──────────┼──────────┼──────────┼─────────┼─────────┼─────────┤
│ 12 (ref)│ 0.003185 │ 0.018640 │ 308.4559 │ 3.9655  │ 49.3014 │ PASS    │
│ 11      │ 0.076909 │ 0.029724 │ 308.4721 │ 3.9817  │ 49.4025 │ FAIL    │
└─────────┴──────────┴──────────┴──────────┴─────────┴─────────┴─────────┘
```

#### Key Findings:
- At `brot = 12`, `error_2 = 0.018640`, leaving an extremely narrow safety margin ($0.020 - 0.01864 = 0.00136\text{ Ha}$) below the $20\text{ mHa}$ threshold.
- Naive truncation to `brot = 11` causes `error_1` ($0.0769$) and `error_2` ($0.0297$) to fail immediately.
- Therefore, `brot = 12` is non-negotiable without re-optimizing the factorization matrices $U, W$.

---

### 4. Circuit Batching Optimization ($J = 9$ vs $J = 8$)

The number of circuits in the family is $K = \lceil 75 / J \rceil$:
- **Baseline ($J = 8$):** $K = \lceil 75 / 8 \rceil = 10$ circuits.
- **Optimized ($J = 9$):** $K = \lceil 75 / 9 \rceil = 9$ circuits.

Moving to $J = 9$ completely eliminates 1 of the 10 QROM circuits:
$$\Delta T = 2 \times 3,974 = \mathbf{7,948\text{ Toffolis saved}}\ (8.5\%\text{ total Toffoli reduction})$$

#### Meeting the 128-Qubit Cap (`QUBIT_CAP`)
In the reference $J = 9$ circuit:
- Selection: 12 qubits ($a < 3,976$).
- Words: $9 \times 12 = 108$ qubits.
- Unconstrained Ancillas: $11$ qubits $\implies 12 + 108 + 11 = 131\text{ qubits}$ (violates 128 cap).

**The 8-Ancilla Architecture:**
By constraining the segment tree to 8 shared ancillas:
$$\text{Total Qubits} = 12 + 108 + 8 = \mathbf{128\text{ qubits}} \le 128$$
The circuit meets the strict verifier cap with zero overhead.

#### Net Score Impact:
- Baseline ($J = 8$): $T = 93,237$, $Q = 730 \implies T \times Q = 68,063,010$.
- Optimized ($J = 9$): $T = 85,289$, $Q = 742 \implies T \times Q = 63,284,438$.
- **Speedup Factor:** $68,063,010 / 63,284,438 = \mathbf{1.0755\times}$ speedup on Benchmark III.

---

### 5. The $\lambda_{\text{eff}}$ Mathematical Spectrum Floor

The overall score multiplier is:
$$\lambda_{\text{eff}} = \sqrt{E_{\text{gap}}(2\Lambda - E_{\text{gap}})}$$

Under the strict acceptance thresholds ($\Lambda > 49.0\text{ Ha}$, $E_{\text{gap}} > 3.914898\text{ Ha}$):
$$\lambda_{\text{eff}}^{\text{floor}} = \sqrt{3.914898 \times (2 \times 49.0 - 3.914898)} = \mathbf{19.192019}$$

The baseline achieves $\lambda_{\text{eff}} = 49.301369$.
Optimizing the electronic structure factorization towards the theoretical floor represents a **$2.56\times$ speedup ceiling** on Benchmark III.
