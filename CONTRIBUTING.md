# Contributing to T-1000

Thank you for your interest in contributing to **T-1000: Quantum Neural Network & Mimetic Simulation Engine**.

This repository is an ultra-high-performance Variational Quantum Circuit (VQC) and Quantum Neural Network engine engineered in pure Rust. To preserve extreme SIMD throughput, cache efficiency, and scientific reproducibility, all contributions must strictly conform to the engineering standards detailed below.

---

## 1. Code of Conduct & Communication Style

- Maintain an objective, factual, and technical focus in all pull requests, code reviews, and issue discussions.
- Follow the Deen OS Sovereign Visual Specification for all charts, diagrams, and telemetry.
- **Zero Emojis:** Do not use emojis in source code, commit messages, terminal banners, pull requests, or markdown documentation.

---

## 2. Mandatory Source File Headers

Every newly created source file (`.rs`, `.py`, `.sh`, `.sql`) must include the standard dual-licensing SPDX header and copyright attribution at the top:

```rust
// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.
```

---

## 3. Engineering & Architectural Standards

All Rust code in the workspace must conform to the following bounds:

| Scope | Maximum Limit | Engineering Policy |
| :--- | :--- | :--- |
| **Source File / Module** | **$\le 800$ LoC** | Files exceeding 800 lines must be systematically refactored into focused submodules. |
| **Function / Method** | **$\le 80$ LoC** | Decompose monolithic logic into single-purpose helper functions with explicit interfaces. |
| **SIMD & Mathematical Kernels** | **$\le 40$ LoC** | Inner butterfly passes and amplitude rotations must be inlined (`#[inline(always)]`), zero-allocation, and AVX2-friendly. |
| **Benchmark & Experiment CLIs** | **$\le 500$ LoC** | Binaries in `src/bin/` orchestrate benchmarks and output formatted telemetry. |

### Hot-Loop Discipline
- **Zero Allocations in Inner Quantum Loops:** Never call `vec![]`, `to_string()`, `format!()`, or dynamic dispatch inside statevector evolution passes, butterfly stages, or expectation reductions.
- **Pre-allocate Scratch Memory:** Allocate buffers once before running iterative loops and reuse them across iterations.

---

## 4. Development Workflow & Verification

Before submitting a pull request, ensure your branch passes all workspace verification checks:

```bash
# 1. Format verification
cargo fmt --all -- --check

# 2. Strict clippy hygiene check (zero warnings tolerated)
cargo clippy --workspace --all-targets -- -D warnings

# 3. Fast workspace compilation check
cargo check --workspace

# 4. Full test suite execution across all workspace members
cargo test --workspace

# 5. Continuous state dynamics and toolchain benchmark verification
cargo run --release -p t1000 --bin continuous_dynamics_bench
cargo run --release -p t1000 --bin deen_toolchain -- bench-sidecar 10000
```

---

## 5. Result Recording & Telemetry

Benchmark and experiment binaries must write outputs through `t1000::results::{write, create, write_scored, write_if_better}`, never direct `std::fs::write`.
- Default execution writes to scratch: `target/bench-scratch/<host>/`.
- Pass `--record` (or set `T1000_RECORD=1`) to commit results to canonical paths or `results/hosts/<host>/`.

---

## 6. Pull Request Process

1. Fork the canonical repository: `https://github.com/tariqselassie-hub/T1000`.
2. Create a focused topic branch: `git checkout -b feature/your-subsystem-name`.
3. Commit with concise, descriptive commit messages adhering to Conventional Commits:
   - `feat(simd): add AVX2 fused diagonal rotation sweep`
   - `perf(compiler): optimize LIFO wire recycling hot-cache reuse`
   - `fix(qec): correct bit-sliced stabilizer popcount syndrome accumulation`
4. Verify all tests pass locally.
5. Submit your pull request against the `master` branch.
