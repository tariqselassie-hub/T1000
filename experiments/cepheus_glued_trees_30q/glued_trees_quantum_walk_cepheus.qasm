OPENQASM 2.0;
include "qelib1.inc";

// ============================================================================
// Deen Research Division · T-1000 Sovereign Quantum Architecture Core
// ALGORITHM: Continuous-Time Quantum Walk (CTQW) on Glued Binary Trees (n = 3)
// THEORETICAL FOUNDATION: Childs et al. (STOC 2003) Exponential Quantum Speedup
// HARDWARE BACKEND: Rigetti Cepheus-1-108Q (Modular 12-Chiplet QPU)
//
// Substrate Mapping:
//   Column-Symmetric Invariant Subspace (8 Columns: q[0] .. q[7])
//   q[0] = Root A (Entrance, Chiplet 0)
//   q[1] = Col 1 (Inside Tree A)
//   q[2] = Col 2 (Inside Tree A)
//   q[3] = Col 3 (Leaves Tree A, Left of Inter-Die Seam, c=2)
//   q[4] = Col 4 (Leaves Tree B, Right of Inter-Die Seam, c=3)
//   q[5] = Col 5 (Inside Tree B)
//   q[6] = Col 6 (Inside Tree B)
//   q[7] = Root B (Terminal Exit Hit Target, Chiplet 1)
//
// The central weld cycle connects q[3] and q[4] directly across the
// vertical inter-die bump-bond seam.
// ============================================================================

qreg q[8];
creg c[8];

// Phase 1: Localize Single Walker Particle at Entrance Root A (q[0])
x q[0];

// ----------------------------------------------------------------------------
// Trotter Step 1: Brickwork XY Hopping Layers
// theta_intra = sqrt(2) * gamma * dt = 0.353553 rad
// theta_weld  = 2.00000 * gamma * dt = 0.500000 rad
// ----------------------------------------------------------------------------

// Layer 1A: Even Bonds (q0-q1, q2-q3, q4-q5, q6-q7)
// Bond (q0, q1)
h q[0]; h q[1];
cx q[0], q[1];
rz(0.353553) q[1];
cx q[0], q[1];
h q[0]; h q[1];
rx(1.5707963) q[0]; rx(1.5707963) q[1];
cx q[0], q[1];
rz(0.353553) q[1];
cx q[0], q[1];
rx(-1.5707963) q[0]; rx(-1.5707963) q[1];

// Bond (q2, q3)
h q[2]; h q[3];
cx q[2], q[3];
rz(0.353553) q[3];
cx q[2], q[3];
h q[2]; h q[3];
rx(1.5707963) q[2]; rx(1.5707963) q[3];
cx q[2], q[3];
rz(0.353553) q[3];
cx q[2], q[3];
rx(-1.5707963) q[2]; rx(-1.5707963) q[3];

// Bond (q4, q5)
h q[4]; h q[5];
cx q[4], q[5];
rz(0.353553) q[5];
cx q[4], q[5];
h q[4]; h q[5];
rx(1.5707963) q[4]; rx(1.5707963) q[5];
cx q[4], q[5];
rz(0.353553) q[5];
cx q[4], q[5];
rx(-1.5707963) q[4]; rx(-1.5707963) q[5];

// Bond (q6, q7)
h q[6]; h q[7];
cx q[6], q[7];
rz(0.353553) q[7];
cx q[6], q[7];
h q[6]; h q[7];
rx(1.5707963) q[6]; rx(1.5707963) q[7];
cx q[6], q[7];
rz(0.353553) q[7];
cx q[6], q[7];
rx(-1.5707963) q[6]; rx(-1.5707963) q[7];

// Layer 1B: Odd Bonds (q1-q2, q3-q4 [WELD SEAM], q5-q6)
// Bond (q1, q2)
h q[1]; h q[2];
cx q[1], q[2];
rz(0.353553) q[2];
cx q[1], q[2];
h q[1]; h q[2];
rx(1.5707963) q[1]; rx(1.5707963) q[2];
cx q[1], q[2];
rz(0.353553) q[2];
cx q[1], q[2];
rx(-1.5707963) q[1]; rx(-1.5707963) q[2];

// WELD SEAM BOND (q3, q4) - Crossing Inter-Die Seam!
h q[3]; h q[4];
cx q[3], q[4];
rz(0.500000) q[4];
cx q[3], q[4];
h q[3]; h q[4];
rx(1.5707963) q[3]; rx(1.5707963) q[4];
cx q[3], q[4];
rz(0.500000) q[4];
cx q[3], q[4];
rx(-1.5707963) q[3]; rx(-1.5707963) q[4];

// Bond (q5, q6)
h q[5]; h q[6];
cx q[5], q[6];
rz(0.353553) q[6];
cx q[5], q[6];
h q[5]; h q[6];
rx(1.5707963) q[5]; rx(1.5707963) q[6];
cx q[5], q[6];
rz(0.353553) q[6];
cx q[5], q[6];
rx(-1.5707963) q[5]; rx(-1.5707963) q[6];

// ----------------------------------------------------------------------------
// Trotter Step 2: Second Ballistic Pass Propelling Wavepacket into Root B
// ----------------------------------------------------------------------------
// Even Bonds
h q[0]; h q[1]; cx q[0], q[1]; rz(0.353553) q[1]; cx q[0], q[1]; h q[0]; h q[1];
rx(1.5707963) q[0]; rx(1.5707963) q[1]; cx q[0], q[1]; rz(0.353553) q[1]; cx q[0], q[1]; rx(-1.5707963) q[0]; rx(-1.5707963) q[1];

h q[2]; h q[3]; cx q[2], q[3]; rz(0.353553) q[3]; cx q[2], q[3]; h q[2]; h q[3];
rx(1.5707963) q[2]; rx(1.5707963) q[3]; cx q[2], q[3]; rz(0.353553) q[3]; cx q[2], q[3]; rx(-1.5707963) q[2]; rx(-1.5707963) q[3];

h q[4]; h q[5]; cx q[4], q[5]; rz(0.353553) q[5]; cx q[4], q[5]; h q[4]; h q[5];
rx(1.5707963) q[4]; rx(1.5707963) q[5]; cx q[4], q[5]; rz(0.353553) q[5]; cx q[4], q[5]; rx(-1.5707963) q[4]; rx(-1.5707963) q[5];

h q[6]; h q[7]; cx q[6], q[7]; rz(0.353553) q[7]; cx q[6], q[7]; h q[6]; h q[7];
rx(1.5707963) q[6]; rx(1.5707963) q[7]; cx q[6], q[7]; rz(0.353553) q[7]; cx q[6], q[7]; rx(-1.5707963) q[6]; rx(-1.5707963) q[7];

// Odd Bonds
h q[1]; h q[2]; cx q[1], q[2]; rz(0.353553) q[2]; cx q[1], q[2]; h q[1]; h q[2];
rx(1.5707963) q[1]; rx(1.5707963) q[2]; cx q[1], q[2]; rz(0.353553) q[2]; cx q[1], q[2]; rx(-1.5707963) q[1]; rx(-1.5707963) q[2];

// WELD SEAM BOND (q3, q4)
h q[3]; h q[4]; cx q[3], q[4]; rz(0.500000) q[4]; cx q[3], q[4]; h q[3]; h q[4];
rx(1.5707963) q[3]; rx(1.5707963) q[4]; cx q[3], q[4]; rz(0.500000) q[4]; cx q[3], q[4]; rx(-1.5707963) q[3]; rx(-1.5707963) q[4];

h q[5]; h q[6]; cx q[5], q[6]; rz(0.353553) q[6]; cx q[5], q[6]; h q[5]; h q[6];
rx(1.5707963) q[5]; rx(1.5707963) q[6]; cx q[5], q[6]; rz(0.353553) q[6]; cx q[5], q[6]; rx(-1.5707963) q[5]; rx(-1.5707963) q[6];

// Phase 3: Projective Measurement Across All Column Registers
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
