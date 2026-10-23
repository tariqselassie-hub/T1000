# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

# T-1000 Sovereign Quantum Dashboard & Lego-Block Circuit Composer

This document specifies the architecture, operational protocols, visual telemetry standards, and execution bridges for the **T-1000 Sovereign Quantum Dashboard** and **Lego-Block Circuit Composer**.

---

## 1. System Overview & Dual-Mission Cockpit

The T-1000 web dashboard provides an empirical interface connecting molecular latent space exploration with physical quantum circuit execution. Built in vanilla HTML5, WebGL (Three.js), and pure CSS/JavaScript, it communicates locally with `web/serve.py` over high-speed HTTP/JSON REST APIs and the FPGA board (`COM6` @ 3.0 MBaud).

### Sovereign 4-Deck Workspace Layout

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          T-1000 SOVEREIGN QUANTUM WORKSPACE & TELEMETRY DECK                           │
├──────────────────────────┬──────────────────────────┬──────────────────────────┬───────────────────────┤
│ [1] MOLECULAR & CTQW     │ [2] CIRCUIT COMPOSER     │ [3] FAULT-TOLERANT QEC   │ [4] 36Q VIRTUAL BRIDGE│
│     (3D MANIFOLD & WALK) │     (12Q LEGO CANVAS)    │     (STEANE W7 & MAGIC)  │     (CUT-BRIDGE & PLAQ│
└──────────────────────────┴──────────────────────────┴──────────────────────────┴───────────────────────┘
```

| Deck | Workspace Scope | Key Subsystems & Telemetry Readouts |
| :--- | :--- | :--- |
| **Deck 1** | 3D Molecular Manifold & CTQW Screener | Force-directed distance-geometry relaxation, CPK atomic spheres, electron orbital density mesh, Coulomb Laplacian eigenvalues, ballistic hitting times $t_{\text{hit}}$, Zeno sentry fidelity ($99.9999\%$), and `→ TO COMPOSER` circuit export. |
| **Deck 2** | Lego-Block Quantum Circuit Composer | Visual quantum wire grid ($q_0 \dots q_{11}$), time layers ($L_0 \dots L_{15}$), modular snap-in gate blocks, floating parameter slider HUD ($\theta \in [0, 2\pi]$), Auto-DD scheduler, and a CPU/FPGA target selector (the FPGA target runs the circuit on the board). |
| **Deck 3** | Fault-Tolerant QEC & Magic State Deck | 18-Qubit Transversal Steane-Wheel $W_7$ ($7\text{ data} + 11\text{ in-register ancillas}$), Bravyi-Kitaev 15-to-1 magic state distillation factory ($P_e \le 35 p^3$), Rotated $[[9, 1, 3]]$ Surface Code, and AVX-512 VNNI parity engine. |
| **Deck 4** | 36Q Virtual Cut-Bridge & Silicon Matrix | $131,072\times$ L3 cache compression ($1.00\text{ TB} \to 8\text{ MB}$), mid-circuit checkpointing, Intel Xe Graphics OpenCL UMA bridge, PC vs Artix-7 FPGA 26Q scaling table, IBM Nighthawk 99-plaquette dual-graph heatmap, and Ultrametric Tree QMUX. |

---

## 2. Deck 1: 3D Quantum Molecular Manifold

### Distance-Geometry Relaxation & Orbital Shells
Deck 1 resolves chemical SMILES strings into topological graphs and computes 3D Euclidean conformations using Hooke's spring attraction and Coulomb electrostatic repulsion across 80 relaxation steps.

- **Atomic & Molecular Rendering:**
  - Standard CPK atomic spheres scaled by covalent radius.
  - Covalent cylindrical bonds oriented along inter-atomic vectors.
  - Translucent electron density cloud mesh (wireframe and volumetric modes).
  - Multi-shell atomic models for ground-state elements $Z=1\dots10$ with core $1s$, valence $2s$, and orthogonal $2p_x, 2p_y, 2p_z$ probability dumbbell lobes.
- **Latent-to-Circuit Bridge:**
  - Clicking `→ TO COMPOSER` extracts the active candidate's 19-dimensional descriptors, encodes them into parameterized rotation angles $R_y(x_i)$, and instantly loads the corresponding Variational Quantum Circuit into Deck 2.

---

## 3. Deck 2: Lego-Block Quantum Circuit Composer

### Modular Gate Palette Governance
Gate blocks conform strictly to the Deen OS Sovereign Visual Specification with zero prohibited tints:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               LEGO GATE PALETTE                                  │
├─────────────────────┬─────────────────────────────────────┬──────────────────────┤
│ Gate Family         │ Visual Appearance                   │ Mathematical Action  │
├─────────────────────┼─────────────────────────────────────┼──────────────────────┤
│ Rotations (Ry,Rx,Rz)│ Laser Emerald Green (#00e676)       │ exp(-i θ σ / 2)      │
│ Clifford / Pauli    │ Polar Slate White (#f8fafc)         │ H, X, Z, S, T        │
│ Entangling (CZ, CX) │ Radiant Amber Gold (#ffb300)        │ Controlled unitary   │
│ Readout (Mz, Del)   │ Crimson Red (#ef4444)               │ Projective POVM      │
└─────────────────────┴─────────────────────────────────────┴──────────────────────┘
```

### Parameter Tuning HUD & Live Tweaking
- **Inline Slider Dial:** Clicking any parameterized gate opens a floating control panel displaying the exact rotation angle in radians and multiples of $\pi$ ($-\pi \le \theta \le +\pi$).
- **Quick-Snap Fractions:** Dedicated instant buttons for $-\pi, -\pi/2, -\pi/4, 0, +\pi/4, +\pi/2, +\pi$.
- **Real-Time Hilbert Update:** Moving the slider triggers instant statevector re-evaluation and updates Pauli-Z expectations and basis amplitude distributions without page reloads.

### 1-Click Experimental Presets
1. **Lead #109 VQC Feature Map:** 10-qubit molecular feature map encoding bioactivity, lipophilicity, and polar surface area with alternating $CZ$ ring couplers.
2. **GHZ / Bell 3-Qubit Entangler:** Generates maximally entangled Greenberger-Horne-Zeilinger state $(|000\rangle + |111\rangle)/\sqrt{2}$.
3. **Frontier RCS Chaos Layer:** Depth-6 pseudo-random Haar unitary passes measuring Porter-Thomas chaos convergence.
4. **Hardware-Efficient Ansatz (HEA 4Q):** Alternating single-qubit rotations with entangling gates for variational ground-state discovery.
5. **Quantum Fourier Transform (QFT 4Q):** Complete 4-wire reversible phase transformation with SWAP reversal.
6. **Zeno Perimeter Sentry:** Rapid projective measurement loops verifying interaction-free quantum confinement.

### Cohesion & Robustness Engine (Active Stabilization)
To prevent phase decoherence and numerical gauge drift across deep circuits ($D \ge 36$, $N \ge 14$), Deck 2 provides active cohesion protocols:

1. **Quantum Zeno Sentry (Subspace Pinning):**
   - Applies frequent null-result projective sentries on target reference/boundary subspaces.
   - Quadratically suppresses phase drift: $P_{\text{surv}}(\tau) = 1 - (\Delta H \cdot \tau)^2 + \mathcal{O}(\tau^4) \to 1.000$.
2. **Dynamical Decoupling (XY4 Echoes):**
   - Interleaves symmetrized $\pi$-pulse trains ($X_\pi - Y_\pi - X_\pi - Y_\pi \equiv I$) on idle qubits to cancel low-frequency non-Markovian dephasing noise.
3. **Zero-Noise Extrapolation (ZNE):**
   - Computes expectation values across unitary folding scale factors $c \in [1, 2, 3]$ and performs Richardson polynomial extrapolation back to the ideal zero-noise limit ($c \to 0$).
4. **Symplectic Manifold Re-Projection:**
   - Enforces strict norm and phase continuity on the $S^{2^{N+1}-1}$ complex projective manifold, eliminating floating-point rounding creep.

---

## 4. Execution targets

`POST /api/run_circuit` with `target: "cpu"` simulates the circuit on the host (NumPy statevector in
`web/serve.py`).

With `target: "fpga"` the server compiles the circuit to the 1000_QPU protocol
(`web/qpu_board.py`: RX, RY, RZ, H, X, Z, S, T as 2x2 gates with 25-bit coefficients; CZ and CX as
controlled gates; SWAP as three CNOTs), resets the board to |0...0>, streams the gates, reads the
full statevector back and computes the observables from the board's amplitudes. The response is
labelled `PHYSICAL_SILICON_EXECUTED` and carries `board_norm_squared`, the board's own `EXPECT_Z`
per qubit (`board_expect_z`), `max_abs_pauli_z_error_vs_host` (the host simulation of the same
circuit), and the measured `wall_time_ms` and `gate_phase_us`. If the board is absent, busy or runs
another bitstream, the result is the host simulation labelled `NOT_EXECUTED_ON_HARDWARE` with the
reason in `hardware_error`. No cycle count, DSP figure, fidelity or attestation hash is reported:
none is measured.

Time per gate is limited by the 36-byte gate frame on the 3 Mbaud UART (about 120 µs), not by the
fabric. Up to 14 qubits; the arithmetic is specified in [`fpga_architecture.md`](fpga_architecture.md).
`python scripts/fpga/verify_dashboard_circuits.py` runs three circuits on the board through the
server and checks them against the host.

---

## 5. REST API Specifications (`serve.py`)

The backend server in [web/serve.py](file:///c:/Users/zoddj/zenith_workspace_divizion/T1000/web/serve.py) provides endpoints for hardware telemetry and simulation:

### `GET /api/status`
Returns host environment details and physical `COM6` serial port handshake status:
```json
{
  "engine": "T-1000 Sovereign VQC & Mimetic Simulation Engine",
  "cpu_simd": "AVX2 + FMA Separated Real/Imaginary SOA",
  "target_silicon": "Xilinx Artix-7 XC7A35T-1CPG236C (Basys 3)",
  "fpga_status": {
    "detected": true,
    "online": true,
    "port": "COM6",
    "baud": 3000000,
    "latency_us": 1042.6,
    "status": "AUTHENTICATED_1000_QPU"
  }
}
```

### `POST /api/run_circuit`
Executes an arbitrary circuit payload on the requested hardware target:
- **Request Body:**
  ```json
  {
    "qubits": 6,
    "layers": [
      { "gates": [{ "type": "RY", "target": 0, "theta": 1.5708 }] },
      { "gates": [{ "type": "CZ", "target": 1, "control": 0 }] }
    ],
    "target": "fpga"
  }
  ```
- **Response Telemetry (`fpga` target, executed on the board):**
  ```json
  {
    "target_executed": "Xilinx Artix-7 XC7A35T (1000_QPU, executed on the board)",
    "hardware_mode": "PHYSICAL_SILICON_EXECUTED",
    "board_gates_sent": 132,
    "board_norm_squared": 1.0,
    "board_expect_z": [0.8791, 0.6904, 0.0256],
    "max_abs_pauli_z_error_vs_host": 0.0,
    "wall_time_ms": 168.243,
    "gate_phase_us": 17299.5,
    "pauli_z": [0.8791, 0.6904, 0.0256],
    "top_states": [ { "index": 0, "basis": "|000000⟩", "prob": 0.5, "real": 0.7071, "imag": 0.0 } ]
  }
  ```

---

## 6. How to Launch & Operate

### Primary Startup Commands
```bash
# Standard interactive launch (automatically launches local browser)
python web/serve.py

# Headless daemon mode (ideal for background execution, remote servers, or CI)
python web/serve.py --no-browser

# Launch with explicit serial port, baud rate, and HTTP port overrides
T1000_QPU_PORT=COM6 T1000_QPU_BAUD=3000000 PORT=8000 python web/serve.py --no-browser
```

### Operational URLs & Endpoints
- **Main Dashboard Deck:** `http://localhost:8000/index.html`
  - **[Deck 1] Molecular & CTQW:** 3D force-directed molecule viewer, SMILES featurizer, and Continuous Topological Walk ballistic hitting telemetry.
  - **[Deck 2] Circuit Composer:** 12-wire quantum wire grid ($q_0 \dots q_{11}$), gate palette, parameter HUD, Auto-DD scheduler, and a CPU/FPGA target selector (the FPGA target runs the circuit on the board).
  - **[Deck 3] Fault-Tolerant QEC:** 18-Qubit Transversal Steane-Wheel $W_7$, Bravyi-Kitaev 15-to-1 magic state distillation factory, and Rotated Surface Code.
  - **[Deck 4] 36Q Virtual Bridge:** 36-Qubit Virtual Cut-Bridge ($131,072\times$ L3 compression), Intel Xe Graphics OpenCL bridge, and IBM Nighthawk 99-plaquette heatmap.
- **Quantum Life (Fractional Fermi QCA Simulator):** `http://localhost:8000/qgol.html`
- **REST Health Status:** `http://localhost:8000/api/status`
- **Unified Telemetry Manifest:** `http://localhost:8000/api/telemetry_data`

