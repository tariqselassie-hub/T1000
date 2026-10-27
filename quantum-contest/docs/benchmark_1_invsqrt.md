# Benchmark I — Coulomb Potential Phasing (1 / sqrt(x))
## Mathematical Specification, Error Bounds & Register Synthesis

SPDX-License-Identifier: MIT OR Apache-2.0  
Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

---

### 1. Problem Formulation

Benchmark I evaluates the unitary operation:
$$|x\rangle|0\dots 0\rangle \longrightarrow \exp\left( \frac{2\pi i}{\sqrt{x}} \right) |x\rangle|0\dots 0\rangle$$
for every integer momentum index $x \in [1, x_{\text{max}}]$, where:
$$n_p = 10 \implies x_{\text{max}} = 3(2^{n_p} - 2)^2 = 3,133,452$$
The input register $x$ consists of $w = 2n_p + 2 = 22$ qubits in little-endian order.

#### Tolerance Contract
The simulated phase $\theta(x)$ (expressed in turns $\in [0, 1)$) must satisfy:
$$\left| \theta(x) - \frac{1}{\sqrt{x}} \right| \le \text{tol}(x) = \frac{2^{-24}}{\sqrt{x}} \pmod 1$$
Phases are evaluated in 64-bit fixed point (units of $2^{-64}$ turns).

---

### 2. Algorithmic Architecture

The standard synthesis pipeline decomposes into three sequential reversible stages:

1. **Normalization:**
   - Counts leading zero bit pairs $s$ of $x$ into a unary register.
   - Rotates $x$ in-place to $x_{\text{norm}} = x \cdot 2^{2s}$, ensuring $M = x_{\text{norm}} / 2^{2n_p} \in [1, 4)$.
   - Decomposes $1/\sqrt{x} = 2^{s - n_p} / \sqrt{M}$.

2. **Polynomial Initial Guess (Horner's Rule):**
   - Segments $[1, 4)$ into four intervals: $[1, 1.5)$, $[1.5, 2)$, $[2, 3)$, $[3, 4)$ chosen by the top three bits of $x_{\text{norm}}$.
   - Cubic polynomial evaluation over $\delta = M - c_{\text{interval}}$:
     $$r_1 = c_2 - T(c_3 \delta)$$
     $$r_2 = c_1 - T(r_1 \delta)$$
     $$y_0 = c_0 - T(r_2 \delta) \approx \frac{1}{\sqrt{M}}$$

3. **Newton-Raphson Iteration:**
   - Evaluates:
     $$p = T\left(y_0 \left(3 - T\left(M \cdot T\left(y_0^2\right)\right)\right)\right) = \frac{2}{\sqrt{M}}$$
   - De-normalizes via unary-controlled shift: $y_1 = p \cdot 2^s$, yielding fractional phase bits.

4. **Phase Kickback & Reversible Uncomputation:**
   - Rotates phase via $Z_{\text{POW}}$ on individual bits of $y_1$.
   - Executes stages 1–3 in reverse order to return all scratch ancillas to $|0\rangle$.

---

### 3. Precision Optimization & Performance Ledger

```
┌────────────────────────────────────────────────────────────────────────┐
│                   BENCHMARK I PERFORMANCE PROGRESSION                  │
├───────────────────┬────────────┬──────────────┬─────────────┬──────────┤
│ Configuration     │ Qubits (Q) │ Toffoli-Eq   │ Score (Q*C) │ Speedup  │
├───────────────────┼────────────┼──────────────┼─────────────┼──────────┤
│ V1 (Reference)    │ 260        │ 7,950.2      │ 2,067,055   │ 1.0000x  │
│ V7 (Certified V2) │ 258        │ 7,671.5      │ 1,979,259   │ 1.0444x  │
│ V8 (Cand5 Tight)  │ 255        │ 7,470.4      │ 1,904,940   │ 1.0851x  │
│ Target (Quad-B1)  │ 238        │ 6,770.0      │ 1,611,260   │ 1.2829x  │
└───────────────────┴────────────┴──────────────┴─────────────┴──────────┘
```

#### Cand5 Parameter Specifications
- `delta_bits = 12` (was 14 in baseline, 13 in V7)
- `coef_bits = 17` (optimal trade-off)
- `square_bits = 27` (was 28 in baseline)
- `product_bits = 27` (was 28 in baseline, 28 in V7)
- `result_bits = 28` (was 29 in baseline)

---

### 4. Boundary Verification & Safe Modular Arithmetic

#### Grader Boundary Test: $x = 2,684,415$
- At $x = 2,684,415$:
  $$\sqrt{x} = 1638.418445 \implies \text{tol}(x) = \frac{2^{-24}}{1638.418445} \approx 3.63806 \times 10^{-11}\text{ turns}$$
- Performance comparison:
  - `result_bits = 27`: Error $= 3.899 \times 10^{-11}$ (FAIL, $+7.2\%$ over threshold).
  - `result_bits = 28` (V7): Error $= 2.444 \times 10^{-11}$ (PASS, $33\%$ safety margin).
  - `Cand5 Tight` (V8): Error $= \mathbf{1.026 \times 10^{-12}}$ turns (**PASS, $35\times$ safety margin**).

#### Modular Unsigned Distance Formula
To avoid NumPy 2.x `RuntimeWarning: overflow encountered in scalar subtract` on `uint64` phase values:
```python
raw_diff = int(th) - int(tgt[0])
mod_diff = ((raw_diff + (1 << 63)) % (1 << 64)) - (1 << 63)
err = abs(mod_diff)
```

---

### 5. Quadratic (Degree-2) Polynomial Discovery

By partitioning $M \in [1, 2)$ into $K$ uniform sub-intervals, a lower-degree polynomial meets the strict $2^{-24}$ Newton-Raphson convergence bound:

1. **4 Sub-intervals ($h = 0.25$):**
   - Intervals: $[1.00, 1.25)$, $[1.25, 1.50)$, $[1.50, 1.75)$, $[1.75, 2.00)$.
   - Interval selection: top two fractional bits of $x_{\text{norm}}$ ($b_{width-3}, b_{width-4}$).
   - Minimax Quadratic Error: $\max_{\text{NR}} = \mathbf{4.481 \times 10^{-8} < 2^{-24} = 5.960 \times 10^{-8}}$ (PASS).
2. **8 Sub-intervals ($h = 0.125$):**
   - Minimax Quadratic Error: $\max_{\text{NR}} = \mathbf{9.732 \times 10^{-10} \ll 2^{-24}}$ ($60\times$ below tolerance).

#### Architectural Benefits of Degree-2:
- **Eradicates $c_3$ permanently:** Cuts 17 physical qubits from the circuit ($255 \to 238$ qubits).
- **Eliminates Horner Step 1:** Horner evaluation simplifies from 3 steps to 2:
  $$r_1 = c_1 - T(c_2 \delta)$$
  $$y_0 = c_0 - T(r_1 \delta)$$
  Saves $\sim 700$ Toffoli gates across forward and adjoint sweeps.
- **Predicted Score:** $238 \times 6,770 = \mathbf{1,611,260}$ ($1.2829\times$ speedup on Benchmark I).
