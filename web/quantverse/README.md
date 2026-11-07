<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# QUANTVERSE: Sovereign 3D Virtual Universe & Quantum Telemetry Observatory

`quantverse` is an interactive WebGL observatory dashboard built with Vite and Three.js, visualizing the quantum mathematical discoveries of `T-1000`.

---

## 1. System Architecture

The dashboard is decoupled into two independent 3D rendering pipelines to preserve clear visual separation:

1. **Main Cosmic System Viewport (`#canvas-container`):**
   - **Photonic Bloom & Post-Processing:** `EffectComposer` with `UnrealBloomPass` (strength: 1.25, radius: 0.55, threshold: 0.16) and `ACESFilmicToneMapping`.
   - **FLRW Expanding Spacetime & Cosmic Web:** 10,000 starfield particles with Bogoliubov spectrum $n_k$, interconnected with 3D large-scale structure filaments and 280 streaming flux particles.
   - **Expanding Hubble Horizon:** Geodesic boundary $R_H(t)$ and equatorial reticle ring breathing with cosmic scale factor $a(t)$.
   - **Bogoliubov Vacuum Ripples:** 3 concentric expanding shockwaves visualizing particle pair creation.
   - **Relativistic Kerr Gravitational Lensing:** Deep Obsidian (`#05080d`) core, Laser Emerald (`#00ff88`) grid shell, unstable photon sphere orbit at $r=13.8$, primary equatorial accretion disk, and vertical warped accretion halo looping over the poles.
   - **Gyroscopic Stabilizers:** Dual precessing counter-rotating rings based on the 8-node throwing jack stabilizer model.
   - **Tactical Reticle Overlay:** CRT scanlines and center targeting crosshairs.
2. **Dedicated ER=EPR Bridge Tactical HUD (`#bridge-canvas`):**
   - Isolated WebGL sub-renderer located directly inside the right HUD panel.
   - Renders the $K=6$ cut-bridge tensor bisection between Fragment A (18Q) and Fragment B (18Q).
   - Visualizes 6 helical spline flux tubes with 300 streaming quasiprobability particles carrying entanglement flux $\Delta = +0.1246327$.
3. **Phase-Lock Plateau Waveform Monitor (`#wave-canvas`):**
   - 2D canvas displaying the 24-step 100.0000% Phase-Lock Coherence Plateau ($j = 26 \to 3$) discovered during 4-digit prime factoring.

---

## 2. Visual Palette & Aesthetics

Conforms strictly to the Deen OS Sovereign Visual Specification:
- **Root Canvas:** Deep Obsidian (`#05080d`)
- **HUD Reticle Cards:** Void Charcoal (`#0f172a` / `rgba(15, 23, 42, 0.85)`)
- **Quantum Telemetry / Wavefunctions:** Laser Emerald Green (`#00ff88`)
- **Hardware / Registers / Accretion:** Radiant Amber / Gold (`#ffd700`)
- **Decoherence Alerts:** Crimson Red (`#ef4444`)
- **Grid Reticles & Dividers:** Muted Slate Grid (`#1e293b` / `#334155`)

---

## 3. Local Execution & Security Governance

To guarantee security, the development server is bound strictly to `127.0.0.1` (loopback only) with zero LAN or WAN broadcasting.

```powershell
# Navigate to web directory
cd web/quantverse

# Install dependencies (Node 20+)
npm install

# Start local dev server (http://127.0.0.1:5173)
npm run dev

# Compile production bundle
npm run build
```

---

## 4. Legal & Licensing

Dual-licensed under the **MIT License** and the **Apache License, Version 2.0**.  
Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

