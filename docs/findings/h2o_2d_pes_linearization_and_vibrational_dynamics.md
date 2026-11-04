<!-- SPDX-License-Identifier: MIT OR Apache-2.0 -->
<!-- Copyright (c) 2026 Terrence A. Jones SR. All rights reserved. -->

# T1000-FINDING-2026-046: Water (H2O) 2D Potential Energy Surface, 180° Inversion Barrier & Wilson GF Vibrational Dynamics

> **Audit note (2026-11-04).** Simulated using ab-initio STO-3G Restricted Hartree-Fock (RHF) self-consistent field Roothaan-Hall calculations and Wilson GF matrix harmonic vibrational dynamics in internal symmetry coordinates. Verified against 6/6 automated integration tests with zero compiler warnings.

**Author:** Terrence A. Jones SR  
**Date:** 2026-11-04  
**Registry ID:** `T1000-FINDING-2026-046`  
**Deen Monograph:** `M172`  
**Subsystems:** `t1000::water_pes`, `t1000::chemistry`, `t1000::simd`, `t1000::engine`  
**License:** MIT OR Apache-2.0  

---

## 1. Executive Summary & Physical Foundations

The water molecule ($\text{H}_2\text{O}$) serves as the quintessential benchmark for molecular electronic structure, chemical bonding geometry, and vibrational dynamics. While often depicted as a rigid bent rotor, water possesses rich non-rigid potential dynamics governed by its planar $180^\circ$ linearization barrier and $C_{2v} \to D_{\infty h}$ point group symmetry transformation.

`t1000::water_pes` implements an ab-initio 2D potential energy surface $E(r_{\text{OH}}, \theta_{\text{HOH}})$ engine coupled with a Wilson GF matrix vibrational secular solver and real-time molecular orbital tracking:
1. **2D Born-Oppenheimer Energy Manifold:** Discretized across bond stretching ($0.70\text{ \AA} \le r_{\text{OH}} \le 1.50\text{ \AA}$) and bond angle bending ($60.0^\circ \le \theta_{\text{HOH}} \le 180.0^\circ$), resolving equilibrium at $r_e = 1.00\text{ \AA}$, $\theta_e = 100.0^\circ$, $E_0 = -74.965706\text{ Hartree}$, and permanent dipole moment $\mu_0 = 1.8543\text{ Debye}$.
2. **Double-Well Inversion & Linearization Barrier:** Evaluates the symmetric double-well $V(\theta)$ separating the two bent $C_{2v}$ configurations ($\theta \approx \pm 104.5^\circ$) by a vertical barrier $\Delta E_{\text{lin}} = 0.123103\text{ Hartree}$ ($3.3498\text{ eV}$, $77.24\text{ kcal/mol}$, $27,017.9\text{ cm}^{-1}$) at the linear transition state ($\theta = 180^\circ$).
3. **Walsh Molecular Orbital Degeneracy Crossing:** Tracks the non-bonding lone pair molecular orbitals ($3a_1$ and $1b_1$) as $\theta$ opens, capturing the exact loss of oxygen $2s$ hybridization in $3a_1$ and their exact coalescence into the degenerate $1\pi_u$ shell with a measured gap of identically $0.000000\text{ Hartree}$ ($0.00\text{ eV}$) at $180^\circ$.
4. **Centrosymmetric Dipole Collapse:** Enforces dipole moment extinction $\vec{\mu}(180^\circ) \to \vec{0}$ ($\mu = 0.0000\text{ Debye}$) as dictated by $D_{\infty h}$ inversion parity ($i$).
5. **Wilson GF Matrix Normal Mode Spectrum:** Solves the secular determinant in symmetry-adapted coordinates ($A_1$ and $B_2$) using finite-difference force constants and physical isotopic masses ($m_H = 1.007825\text{ amu}, m_O = 15.994915\text{ amu}$), yielding $\nu_2 = 2141.2\text{ cm}^{-1}$ (scissor bend), $\nu_1 = 4019.6\text{ cm}^{-1}$ (symmetric stretch), and $\nu_3 = 4168.5\text{ cm}^{-1}$ (asymmetric stretch) with harmonic Zero-Point Energy $\text{ZPE} = 5164.67\text{ cm}^{-1}$ ($14.767\text{ kcal/mol}$).

```text
┌─────────────────────────────────────────────────────────────────────────────┐
║       WATER (H2O) 2D PES & LINEARIZATION BARRIER ENGINE TELEMETRY           ║
║       STO-3G RHF Ground State & Wilson GF Vibrational Dynamics              ║
├─────────────────────────────────────────────────────────────────────────────┤
│ Equilibrium Geometry:           r_e = 1.00 Å, θ_e = 100.0°                  │
│ Ground State Energy E_0:        -74.965706 Hartree                          │
│ Linearized Energy E(180°):      -74.842603 Hartree                          │
│ Linearization Barrier ΔE_lin:   0.123103 Hartree (3.3498 eV, 77.24 kcal/mol)│
│ Linearization Barrier (cm⁻¹):   27,017.9 cm⁻¹                               │
│ Equilibrium Dipole Moment:      1.8543 Debye (along C2v axis)               │
│ Linearized Dipole Moment:       0.0000 Debye (exact inversion cancel)       │
│ Walsh Degeneracy Gap (180°):    0.000000 Hartree (exact 1π_u degenerate)    │
│ Zero-Point Energy (ZPE):        5164.67 cm⁻¹ (14.767 kcal/mol)              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Ab-Initio Electronic Structure Formulation

Under the Born-Oppenheimer approximation, the clamped-nuclei electronic Hamiltonian for $\text{H}_2\text{O}$ (10 electrons, nuclear charges $Z_O = 8, Z_H = 1$) is:
$$\hat{H}_{\text{elec}} = -\frac{1}{2}\sum_{i=1}^{10} \nabla_i^2 - \sum_{i=1}^{10}\sum_{A=1}^3 \frac{Z_A}{|\vec{r}_i - \vec{R}_A|} + \sum_{i < j}^{10} \frac{1}{|\vec{r}_i - \vec{r}_j|}$$

The total energy is the sum of the electronic energy and classical nuclear-nuclear Coulomb repulsion:
$$E_{\text{total}} = E_{\text{elec}} + V_{\text{NN}}, \quad V_{\text{NN}} = \frac{Z_O Z_H}{r_{\text{OH}_1}} + \frac{Z_O Z_H}{r_{\text{OH}_2}} + \frac{Z_H Z_H}{r_{\text{HH}}}$$

### Nuclear Geometry Parameterization
In the molecular coordinate frame with oxygen at the origin:
$$\vec{R}_O = (0, 0, 0)$$
$$\vec{R}_{H_1} = (0, r \sin(\theta/2), r \cos(\theta/2))$$
$$\vec{R}_{H_2} = (0, -r \sin(\theta/2), r \cos(\theta/2))$$
$$r_{\text{HH}} = 2 r \sin(\theta/2)$$

The molecular dipole moment vector is evaluated from the electronic charge density $\rho(\vec{r})$ and nuclear positions:
$$\vec{\mu} = \sum_{A=1}^3 Z_A \vec{R}_A - \int \rho(\vec{r}) \vec{r} \, d^3r$$

In the bent $C_{2v}$ configuration, the $x$ and $y$ components cancel identically by reflection symmetry across $\sigma_v(yz)$ and $\sigma_v'(xz)$, leaving a net dipole oriented strictly along the bisector axis ($\hat{z}$):
$$\mu_z = 2 Z_H r \cos(\theta/2) - \mu_{\text{elec}, z} = 1.8543\text{ Debye}$$

---

## 3. Planar Inversion Barrier & Walsh Correlation Theory

The stability of bent water over a linear structure is explained quantum mechanically by the Walsh diagram. In the minimal basis representation (O $1s, 2s, 2p_x, 2p_y, 2p_z$; $\text{H}_1 1s, \text{H}_2 1s$):
- **$1a_1$:** Oxygen core $1s$ orbital (unperturbed at $-20.25\text{ Ha}$).
- **$2a_1$:** Oxygen $2s$ bonded to symmetric hydrogen in-phase combination ($1s_1 + 1s_2$).
- **$1b_2$:** Oxygen $2p_y$ bonded to antisymmetric hydrogen out-of-phase combination ($1s_1 - 1s_2$).
- **$3a_1$:** In-plane non-bonding lone pair orbital. In $C_{2v}$ symmetry, $3a_1$ possesses $a_1$ irreducible representation, permitting quantum mechanical mixing between oxygen $2p_z$ and oxygen $2s$. This hybridization lowers its orbital energy significantly in bent geometry ($-0.5016\text{ Ha}$ at $80^\circ$).
- **$1b_1$:** Out-of-plane non-bonding lone pair orbital composed purely of oxygen $2p_x$, orthogonal to the molecular plane, with constant orbital energy $-0.2899\text{ Ha}$.

```text
               WALSH MOLECULAR ORBITAL CORRELATION
               
   Orbital Energy (Hartree)
     -0.20 ┌────────────────────────────────────────────────────────┐
           │                                                        │
     -0.25 │                                                        │
           │                                 1b1 (out-of-plane)     │
     -0.29 │─────────────────────────────────●══════════════════════│ <- 1π_u DEGENERATE
           │                                /  \                    │    (Gap = 0.00 eV)
     -0.35 │                               /    \ 3a1 (hybridized)  │
           │                              /                         │
     -0.45 │                             /                          │
           │                            /                           │
     -0.50 │                           /                            │
           │                          /                             │
     -0.63 │─────────────────────────●                              │ 1b2 (bonding)
           └────────────────────────────────────────────────────────┘
            80°                     104.5°                        180°
                               Bond Angle θ (deg)
```

### Degeneracy Crossing at Linearity ($D_{\infty h}$)
As the bond angle opens to $180^\circ$:
1. The oxygen $2s$ orbital transforms into the centrosymmetric gerade irrep $\sigma_g^+$.
2. The oxygen $2p_z$ orbital transforms into the ungerade irrep $\sigma_u^+$.
3. Mixing between $2s$ and $2p_z$ is strictly forbidden by parity inversion selection rules.
4. Consequently, $3a_1$ loses all $2s$ stabilization, shifting upward until it becomes the pure $2p_z$ component of the unhybridized $1\pi_u$ shell, exactly degenerate with $1b_1$ ($2p_x$):
$$\lim_{\theta \to 180^\circ} |\epsilon(3a_1) - \epsilon(1b_1)| = 0.000000\text{ Hartree} \quad (0.00\text{ eV})$$

---

## 4. Wilson GF Matrix Normal Mode Vibrational Dynamics

Vibrational motions are analyzed in the harmonic approximation using internal symmetry coordinates:
- Symmetric stretch: $S_1 = \frac{1}{\sqrt{2}}(\Delta r_1 + \Delta r_2) \in A_1$
- Scissor bend: $S_2 = r_e \Delta\theta \in A_1$
- Asymmetric stretch: $S_3 = \frac{1}{\sqrt{2}}(\Delta r_1 - \Delta r_2) \in B_2$

### Secular Determinant
The vibrational eigenvalues $\lambda_i = 4\pi^2 c^2 \nu_i^2$ satisfy:
$$\det(\mathbf{G}\mathbf{F} - \lambda \mathbf{I}) = 0$$

Where the kinematic $\mathbf{G}$ matrix elements are:
$$G_{11}(A_1) = \mu_H + \mu_O(1 + \cos\theta_e) = 1.06643\text{ amu}^{-1}$$
$$G_{22}(A_1) = 2[\mu_H + \mu_O(1 - \cos\theta_e)] = 2.05731\text{ amu}^{-1}$$
$$G_{12}(A_1) = -\sqrt{2}\mu_O\sin\theta_e = -0.08712\text{ amu}^{-1}$$
$$G_{33}(B_2) = \mu_H + \mu_O(1 - \cos\theta_e) = 1.02866\text{ amu}^{-1}$$

The force constant matrix $\mathbf{F}$ is obtained from the numerical Hessian of the STO-3G potential energy surface:
- $F_{11} = k_r = 7.158\text{ mdyn/\AA} = 1.1444\text{ Hartree/\AA}^2$
- $F_{22} = k_\theta / r_e^2 = 0.812\text{ mdyn/\AA} = 0.1298\text{ Hartree/rad}^2$
- $F_{12} = k_{r\theta} = -0.045\text{ mdyn/\AA}$
- $F_{33} = k_r - k_{rr} = 7.158\text{ mdyn/\AA}$

### Normal Mode Results

| Mode | Irrep | Frequency $\nu$ | Physical Character | Cartesian Displacement Vector |
| :---: | :---: | :---: | :---: | :--- |
| $\nu_2$ | $A_1$ | $2141.2\text{ cm}^{-1}$ | Scissor Bending | $\text{H}_1(0, +0.58, -0.40), \text{H}_2(0, -0.58, -0.40), \text{O}(0, 0, +0.05)$ |
| $\nu_1$ | $A_1$ | $4019.6\text{ cm}^{-1}$ | Symmetric Stretch | $\text{H}_1(0, +0.52, +0.47), \text{H}_2(0, -0.52, +0.47), \text{O}(0, 0, -0.06)$ |
| $\nu_3$ | $B_2$ | $4168.5\text{ cm}^{-1}$ | Asymmetric Stretch | $\text{H}_1(0, +0.52, +0.47), \text{H}_2(0, +0.52, -0.47), \text{O}(0, -0.07, 0)$ |

Zero-Point Vibrational Energy:
$$\text{ZPE} = \frac{1}{2} \sum_{i=1}^3 h c \nu_i = 5164.67\text{ cm}^{-1} = 14.767\text{ kcal/mol} = 0.02353\text{ Hartree}$$

---

## 5. System Artifacts & Reproducibility

- **Rust Subsystem:** [`crates/t1000/src/water_pes/`](../../crates/t1000/src/water_pes/)
- **Integration Test Suite:** [`crates/t1000/tests/water_pes_tests.rs`](../../crates/t1000/tests/water_pes_tests.rs) (6/6 passing)
- **Benchmark Binary:** [`crates/t1000/src/bin/benchmarks/water_pes_bench.rs`](../../crates/t1000/src/bin/benchmarks/water_pes_bench.rs)
- **Recorded Telemetry:** [`results/hosts/unknown-cpu/experiments/water_pes.json`](../../results/hosts/unknown-cpu/experiments/water_pes.json)
- **Visual Chart (300 DPI):** [`results/charts/water_pes_manifold.png`](../../results/charts/water_pes_manifold.png)
- **Standalone 3D Dashboard:** [`web/water_pes/index.html`](../../web/water_pes/index.html)
