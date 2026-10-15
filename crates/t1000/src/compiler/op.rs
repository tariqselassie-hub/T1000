//! Quantum Circuit Operation primitives and 2x2 complex unitary algebra.
//! Supports gate commutativity analysis, qubit dependency footprinting,
//! and single-qubit gate fusion for cache minimization.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// 2x2 Complex Unitary Matrix representing a general single-qubit quantum gate:
/// U = [u00, u01; u10, u11]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Unitary2x2 {
    pub u00_re: f64,
    pub u00_im: f64,
    pub u01_re: f64,
    pub u01_im: f64,
    pub u10_re: f64,
    pub u10_im: f64,
    pub u11_re: f64,
    pub u11_im: f64,
}

impl Unitary2x2 {
    pub const IDENTITY: Self = Self {
        u00_re: 1.0,
        u00_im: 0.0,
        u01_re: 0.0,
        u01_im: 0.0,
        u10_re: 0.0,
        u10_im: 0.0,
        u11_re: 1.0,
        u11_im: 0.0,
    };

    pub fn new(u00: Complex64, u01: Complex64, u10: Complex64, u11: Complex64) -> Self {
        Self {
            u00_re: u00.re,
            u00_im: u00.im,
            u01_re: u01.re,
            u01_im: u01.im,
            u10_re: u10.re,
            u10_im: u10.im,
            u11_re: u11.re,
            u11_im: u11.im,
        }
    }

    #[inline(always)]
    pub fn u00(&self) -> Complex64 {
        Complex64::new(self.u00_re, self.u00_im)
    }

    #[inline(always)]
    pub fn u01(&self) -> Complex64 {
        Complex64::new(self.u01_re, self.u01_im)
    }

    #[inline(always)]
    pub fn u10(&self) -> Complex64 {
        Complex64::new(self.u10_re, self.u10_im)
    }

    #[inline(always)]
    pub fn u11(&self) -> Complex64 {
        Complex64::new(self.u11_re, self.u11_im)
    }

    /// Single-qubit Y-rotation: Ry(θ) = [cos(θ/2), -sin(θ/2); sin(θ/2), cos(θ/2)]
    pub fn ry(theta: f64) -> Self {
        let half = theta * 0.5;
        let c = half.cos();
        let s = half.sin();
        Self {
            u00_re: c,
            u00_im: 0.0,
            u01_re: -s,
            u01_im: 0.0,
            u10_re: s,
            u10_im: 0.0,
            u11_re: c,
            u11_im: 0.0,
        }
    }

    /// Single-qubit Z-rotation: Rz(θ) = [exp(-i θ/2), 0; 0, exp(i θ/2)]
    pub fn rz(theta: f64) -> Self {
        let half = theta * 0.5;
        Self {
            u00_re: half.cos(),
            u00_im: -half.sin(),
            u01_re: 0.0,
            u01_im: 0.0,
            u10_re: 0.0,
            u10_im: 0.0,
            u11_re: half.cos(),
            u11_im: half.sin(),
        }
    }

    /// Hadamard gate: H = (1/√2) [1, 1; 1, -1]
    pub fn hadamard() -> Self {
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        Self {
            u00_re: inv_sqrt2,
            u00_im: 0.0,
            u01_re: inv_sqrt2,
            u01_im: 0.0,
            u10_re: inv_sqrt2,
            u10_im: 0.0,
            u11_re: -inv_sqrt2,
            u11_im: 0.0,
        }
    }

    /// Matrix multiplication: (Self * other)
    /// If gate A is applied first, then gate B, the combined operator is B.matmul(&A).
    pub fn matmul(&self, other: &Self) -> Self {
        let (a00, a01, a10, a11) = (self.u00(), self.u01(), self.u10(), self.u11());
        let (b00, b01, b10, b11) = (other.u00(), other.u01(), other.u10(), other.u11());
        Self::new(
            a00 * b00 + a01 * b10,
            a00 * b01 + a01 * b11,
            a10 * b00 + a11 * b10,
            a10 * b01 + a11 * b11,
        )
    }

    /// Returns true if off-diagonal elements are numerically zero (diagonal gate).
    pub fn is_diagonal(&self, eps: f64) -> bool {
        self.u01().norm_sqr() < eps && self.u10().norm_sqr() < eps
    }

    /// Returns true if matrix is approximately identity.
    pub fn is_identity(&self, eps: f64) -> bool {
        (self.u00() - Complex64::new(1.0, 0.0)).norm_sqr() < eps
            && self.u01().norm_sqr() < eps
            && self.u10().norm_sqr() < eps
            && (self.u11() - Complex64::new(1.0, 0.0)).norm_sqr() < eps
    }
}

/// Elementary Quantum Circuit Operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CircuitOp {
    /// Single-qubit parameterized Y-rotation
    Ry { qubit: usize, theta: f64 },
    /// Single-qubit parameterized Z-rotation
    Rz { qubit: usize, theta: f64 },
    /// Arbitrary single-qubit 2x2 complex unitary (e.g., fused gates)
    Unitary1Q { qubit: usize, matrix: Unitary2x2 },
    /// Two-qubit Controlled-Z (CZ) entangling gate
    CZ { q1: usize, q2: usize },
    /// Two-qubit CNOT (Controlled-NOT) gate
    CNOT { control: usize, target: usize },
    /// Multi-qubit Walsh-Rademacher ±π diagonal phase layer
    RademacherPhase { signs: Vec<f64> },
}

impl CircuitOp {
    /// Returns the list of qubits read/written by this operation.
    pub fn qubits(&self) -> Vec<usize> {
        match self {
            CircuitOp::Ry { qubit, .. }
            | CircuitOp::Rz { qubit, .. }
            | CircuitOp::Unitary1Q { qubit, .. } => vec![*qubit],
            CircuitOp::CZ { q1, q2 } => vec![*q1, *q2],
            CircuitOp::CNOT { control, target } => vec![*control, *target],
            CircuitOp::RademacherPhase { signs } => (0..signs.len()).collect(),
        }
    }

    /// Returns true if the operation touches qubit `q`.
    pub fn touches_qubit(&self, q: usize) -> bool {
        match self {
            CircuitOp::Ry { qubit, .. }
            | CircuitOp::Rz { qubit, .. }
            | CircuitOp::Unitary1Q { qubit, .. } => *qubit == q,
            CircuitOp::CZ { q1, q2 } => *q1 == q || *q2 == q,
            CircuitOp::CNOT { control, target } => *control == q || *target == q,
            CircuitOp::RademacherPhase { signs } => q < signs.len(),
        }
    }

    /// Returns true if this is a single-qubit operation.
    pub fn is_single_qubit(&self) -> bool {
        matches!(
            self,
            CircuitOp::Ry { .. } | CircuitOp::Rz { .. } | CircuitOp::Unitary1Q { .. }
        )
    }

    /// Returns true if the operation is diagonal in the computational Z-basis.
    /// All diagonal gates commute with each other.
    pub fn is_diagonal(&self) -> bool {
        match self {
            CircuitOp::Rz { .. } | CircuitOp::CZ { .. } | CircuitOp::RademacherPhase { .. } => true,
            CircuitOp::Unitary1Q { matrix, .. } => matrix.is_diagonal(1e-12),
            CircuitOp::Ry { .. } | CircuitOp::CNOT { .. } => false,
        }
    }

    /// Determines whether two operations commute algebraically.
    /// If operations commute, their relative ordering in the schedule does not alter state fidelity.
    pub fn commutes_with(&self, other: &Self) -> bool {
        let q1 = self.qubits();
        let q2 = other.qubits();

        // 1. Operations on completely disjoint qubits ALWAYS commute
        let disjoint = !q1.iter().any(|q| q2.contains(q));
        if disjoint {
            return true;
        }

        // 2. Diagonal gates on any set of qubits commute with each other: [Z_i, Z_j] = 0, [CZ, Z] = 0
        if self.is_diagonal() && other.is_diagonal() {
            return true;
        }

        // 3. Same single-qubit rotations commute with themselves
        match (self, other) {
            (CircuitOp::Ry { qubit: qa, .. }, CircuitOp::Ry { qubit: qb, .. }) if qa == qb => true,
            (CircuitOp::Rz { qubit: qa, .. }, CircuitOp::Rz { qubit: qb, .. }) if qa == qb => true,
            _ => false,
        }
    }

    /// Convert any single-qubit operation into an explicit Unitary2x2.
    pub fn to_unitary2x2(&self) -> Option<Unitary2x2> {
        match self {
            CircuitOp::Ry { theta, .. } => Some(Unitary2x2::ry(*theta)),
            CircuitOp::Rz { theta, .. } => Some(Unitary2x2::rz(*theta)),
            CircuitOp::Unitary1Q { matrix, .. } => Some(*matrix),
            _ => None,
        }
    }

    /// Attempts to fuse `self` followed by `next` on the same qubit into a single compound Unitary1Q.
    /// If successful, returns the combined operation which cuts statevector traversal overhead by 2x.
    pub fn try_fuse_single_qubit(&self, next: &Self) -> Option<CircuitOp> {
        let q_self = match self {
            CircuitOp::Ry { qubit, .. }
            | CircuitOp::Rz { qubit, .. }
            | CircuitOp::Unitary1Q { qubit, .. } => *qubit,
            _ => return None,
        };

        let q_next = match next {
            CircuitOp::Ry { qubit, .. }
            | CircuitOp::Rz { qubit, .. }
            | CircuitOp::Unitary1Q { qubit, .. } => *qubit,
            _ => return None,
        };

        if q_self != q_next {
            return None;
        }

        // Fast path for two consecutive Rz rotations: Rz(θ1 + θ2)
        if let (CircuitOp::Rz { theta: t1, .. }, CircuitOp::Rz { theta: t2, .. }) = (self, next) {
            return Some(CircuitOp::Rz {
                qubit: q_self,
                theta: t1 + t2,
            });
        }

        // Fast path for two consecutive Ry rotations: Ry(θ1 + θ2)
        if let (CircuitOp::Ry { theta: t1, .. }, CircuitOp::Ry { theta: t2, .. }) = (self, next) {
            return Some(CircuitOp::Ry {
                qubit: q_self,
                theta: t1 + t2,
            });
        }

        // General unitary fusion: U_fused = U_next * U_self
        let u_self = self.to_unitary2x2()?;
        let u_next = next.to_unitary2x2()?;
        let u_fused = u_next.matmul(&u_self);

        Some(CircuitOp::Unitary1Q {
            qubit: q_self,
            matrix: u_fused,
        })
    }
}
