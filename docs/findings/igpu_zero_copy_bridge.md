<!--
SPDX-License-Identifier: MIT OR Apache-2.0
Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
-->

# Zero-Copy Intel Xe iGPU Co-Processor Bridge & UMA Quantum Simulation

> **Audit note (2026-10-23).** Re-measured on 2026-10-23: 9.5 ms on the iGPU against 17.4 ms on the CPU at 22 qubits (1.8x). The 2.13x in this document is from an earlier run. Below 20 qubits the iGPU is slower than the CPU (16 qubits: 8.2 ms against 0.06 ms).


## 1. Executive Summary

This report documents the architectural design, zero-copy memory mechanics, and empirical scaling of the **Zero-Copy Intel Xe Integrated GPU (iGPU) Co-Processor Bridge** (`t1000::gpu::IGpuBridge`) implemented in `T-1000`.

Engineered for dense quantum statevector evolution exceeding CPU cache capacities ($N \ge 22$, $\ge 64\text{ MB}$ statevectors), the bridge leverages the Unified Memory Architecture (UMA) shared between the Intel Arrow Lake CPU host and Intel Xe Graphics execution units (EUs). By binding Rust `QuantumState` Structure-of-Arrays (SOA) memory pointers directly to OpenCL buffers via `CL_MEM_USE_HOST_PTR`, physical PCIe data transfers are completely eliminated.

Empirical benchmarks demonstrate:
* **$2.13\times$ Compute Speedup at $N=22$ ($64\text{ MB}$, $4,194,304$ amplitudes):** State evolution completes in **$8.186\text{ ms}$** on the iGPU versus **$16.392\text{ ms}$** on the CPU AVX2+FMA engine.
* **Exact Numerical Invariant:** Verified cross-backend statevector fidelity of **$1.0000000$** ($|1.0 - \mathcal{F}| < 10^{-14}$) between CPU AVX2 and iGPU OpenCL executions.
* **Zero External Toolchain Dependencies:** Pure dynamic loading of `OpenCL.dll` at runtime via standard Win32 API calls (`LoadLibraryA`, `GetProcAddress`), requiring zero C compilers or third-party OpenCL SDK installations.

---

## 2. Microarchitectural Architecture & UMA Pipeline

### 2.1 Unified Memory Architecture (UMA) on Arrow Lake
Modern Intel Core Ultra processors integrate the CPU cores and Xe Graphics engine onto the same silicon package, sharing a high-bandwidth unified LPDDR5x/DDR5 memory bus. Unlike discrete GPUs connected over PCIe x16 buses that suffer from high transfer latency ($10\text{--}50\,\mu\text{s}$ transfer initiation and bandwidth caps), the Intel Xe iGPU reads directly from the same physical RAM pages allocated by the Rust standard allocator.

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Physical DDR5 Memory                │
│    ┌───────────────────────────────────────────────────┐    │
│    │  t1000::QuantumState                              │    │
│    │  real: [f64; 2^N] (64-byte aligned L2/L3 blocks)  │    │
│    │  imag: [f64; 2^N] (64-byte aligned L2/L3 blocks)  │    │
│    └─────────────────────────┬─────────────────────────┘    │
└──────────────────────────────┼──────────────────────────────┘
                               │ CL_MEM_USE_HOST_PTR (Zero-Copy)
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
┌───────────────────────┐             ┌───────────────────────┐
│ CPU Host (Lion Cove)  │             │ Intel(R) Xe Graphics  │
│ 256-bit AVX2+FMA Core │             │ OpenCL Execution Units│
└───────────────────────┘             └───────────────────────┘
```

### 2.2 Host Cache Flushing & Consistency Protocol
When `CL_MEM_USE_HOST_PTR` is utilized on Intel UMA platforms, the GPU execution units write directly to shared memory through write-combining (WC) buffers. If the host CPU immediately reads from its L1/L2/L3 cache lines without cache invalidation, cache incoherency causes amplitude divergence ($\mathcal{F} \approx 0.72$).

To guarantee deterministic, bit-for-bit numerical reproducibility across backends, `t1000::gpu` executes a synchronous read buffer synchronization step after kernel completion:
```rust
(self.clEnqueueReadBuffer)(
    self.queue,
    d_real,
    CL_TRUE, // Blocking read ensures CPU cacheline invalidation
    0,
    byte_size,
    state.real.as_mut_ptr() as *mut c_void,
    0,
    null(),
    null_mut(),
);
```
This forces the Intel memory controller to flush GPU write caches into DDR5 host-visible lines, ensuring exact $\mathcal{F} = 1.0000000$ fidelity.

---

## 3. High-Performance OpenCL Kernel Design

The rotation kernel (`apply_ry_kernel`) applies a single-qubit $R_y(\theta)$ gate to qubit $q$ across all $2^N$ amplitudes in parallel:
$$R_y(\theta) = \begin{bmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{bmatrix}$$

```c
__kernel void apply_ry_kernel(
    __global double* real,
    __global double* imag,
    const ulong half_dim,
    const ulong target_mask,
    const double cos_half,
    const double sin_half)
{
    ulong tid = get_global_id(0);
    if (tid >= half_dim) return;

    // Bit-expansion inserting target qubit bit '0' and '1'
    ulong i0 = (tid & ~(target_mask - 1UL)) << 1 | (tid & (target_mask - 1UL));
    ulong i1 = i0 | target_mask;

    double r0 = real[i0];
    double r1 = real[i1];
    double m0 = imag[i0];
    double m1 = imag[i1];

    real[i0] = cos_half * r0 - sin_half * r1;
    real[i1] = sin_half * r0 + cos_half * r1;
    imag[i0] = cos_half * m0 - sin_half * m1;
    imag[i1] = sin_half * m0 + cos_half * m1;
}
```

Every OpenCL work-item updates two amplitudes ($i_0$ and $i_1$) simultaneously, maximizing memory coalescing across 64-byte DDR5 burst transactions.

---

## 4. Empirical Scaling & Benchmark Results

Measurements taken on **Intel(R) Core(TM) Ultra 7 265** with integrated **Intel(R) Graphics** (`igpu_bridge_bench`):

| Qubits ($N$) | Hilbert Dim ($2^N$) | State Memory (SOA) | CPU AVX2 (ms) | iGPU UMA (ms) | Speedup Factor | Verification Fidelity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **16** | $65,536$ | $1.00\text{ MB}$ | **$0.031\text{ ms}$** | $1.973\text{ ms}$ | $0.02\times$ (L3 bound) | $1.0000000$ |
| **18** | $262,144$ | $4.00\text{ MB}$ | $0.825\text{ ms}$ | **$0.665\text{ ms}$** | **$1.24\times$** | $1.0000000$ |
| **20** | $1,048,576$ | $16.00\text{ MB}$ | $3.259\text{ ms}$ | **$1.956\text{ ms}$** | **$1.67\times$** | $1.0000000$ |
| **22** | $4,194,304$ | $64.00\text{ MB}$ | $16.392\text{ ms}$ | **$8.186\text{ ms}$** | **$2.13\times$** | **$1.0000000$** |

### 4.1 Crossover Analysis
* **L3 Cache Regime ($N \le 16$, $\le 1\text{ MB}$):** The CPU AVX2 engine dominates because the entire statevector fits effortlessly into the 30 MB L3 cache of the Core Ultra 7 265, avoiding memory bus roundtrips entirely. The iGPU overhead is dominated by OpenCL command queue dispatch ($~1.9\text{ ms}$).
* **UMA Memory-Bound Regime ($N \ge 18$, $\ge 4\text{ MB}$):** As statevector sizes surpass L2 cache lines and spill toward DDR5 RAM, the massively parallel execution units of the Intel Xe iGPU saturate DDR5 memory bandwidth significantly more effectively than scalar or AVX2 CPU threads, yielding a clean **$2.13\times$ speedup at $22$ qubits**.

---

## 5. Architectural Integration

The bridge is exposed through:
1. **Low-Level Native Bridge:** [`t1000::gpu::IGpuBridge`](file:///c:/Users/zoddj/zenith_workspace_divizion/T1000/crates/t1000/src/gpu/mod.rs) for explicit zero-copy rotation passes (`apply_ry_zero_copy`).
2. **Master Engine Manifest:** [`t1000::engine::T1000Engine::system_manifest()`](file:///c:/Users/zoddj/zenith_workspace_divizion/T1000/crates/t1000/src/engine.rs) exposing:
   * `manifest.igpu_bridge_available`: Active detection of OpenCL runtime and device.
   * `manifest.igpu_device_name`: String identifier (`Some("Intel(R) Graphics")`).
