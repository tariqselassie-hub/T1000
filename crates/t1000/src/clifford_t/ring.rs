// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Exact algebraic arithmetic in the cyclotomic ring Z[ω] with ω = exp(i*pi/4) = (1 + i) / sqrt(2).
//!
//! Every single-qubit Clifford+T operator can be represented as:
//! U = (1 / sqrt(2)^k) * [ u, -conj(v); v, conj(u) ]
//! where u, v in Z[ω] and |u|^2 + |v|^2 = 2^k.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Element of the cyclotomic integer ring Z[ω] represented in the basis {1, ω, ω^2, ω^3}:
/// z = a + b*ω + c*ω^2 + d*ω^3, with ω^4 = -1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ZOmega {
    pub a: i64,
    pub b: i64,
    pub c: i64,
    pub d: i64,
}

impl ZOmega {
    /// Zero element: 0
    pub const ZERO: Self = Self {
        a: 0,
        b: 0,
        c: 0,
        d: 0,
    };
    /// Identity element: 1
    pub const ONE: Self = Self {
        a: 1,
        b: 0,
        c: 0,
        d: 0,
    };
    /// Primitive 8th root of unity: ω = exp(i*pi/4)
    pub const OMEGA: Self = Self {
        a: 0,
        b: 1,
        c: 0,
        d: 0,
    };
    /// Phase gate generator: ω^2 = i (S gate)
    pub const OMEGA2: Self = Self {
        a: 0,
        b: 0,
        c: 1,
        d: 0,
    };
    /// Third power: ω^3 = (-1 + i) / sqrt(2)
    pub const OMEGA3: Self = Self {
        a: 0,
        b: 0,
        c: 0,
        d: 1,
    };
    /// Exact representation of sqrt(2) in Z[ω]: ω - ω^3
    pub const SQRT2: Self = Self {
        a: 0,
        b: 1,
        c: 0,
        d: -1,
    };

    /// Constructs a ring element from basis coordinates.
    pub const fn new(a: i64, b: i64, c: i64, d: i64) -> Self {
        Self { a, b, c, d }
    }

    /// Addition: self + rhs
    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
            c: self.c + rhs.c,
            d: self.d + rhs.d,
        }
    }

    /// Subtraction: self - rhs
    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
            c: self.c - rhs.c,
            d: self.d - rhs.d,
        }
    }

    /// Multiplication in Z[ω] modulo (ω^4 + 1 = 0).
    ///
    /// Identities:
    /// ω^4 = -1, ω^5 = -ω, ω^6 = -ω^2, ω^7 = -ω^3.
    #[inline(always)]
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a - self.b * rhs.d - self.c * rhs.c - self.d * rhs.b,
            b: self.a * rhs.b + self.b * rhs.a - self.c * rhs.d - self.d * rhs.c,
            c: self.a * rhs.c + self.b * rhs.b + self.c * rhs.a - self.d * rhs.d,
            d: self.a * rhs.d + self.b * rhs.c + self.c * rhs.b + self.d * rhs.a,
        }
    }

    /// Scalar multiplication: s * self
    #[inline(always)]
    pub fn mul_scalar(self, s: i64) -> Self {
        Self {
            a: self.a * s,
            b: self.b * s,
            c: self.c * s,
            d: self.d * s,
        }
    }

    /// Complex conjugate in Z[ω]: ω -> ω^7 = -ω^3, ω^2 -> -ω^2, ω^3 -> -ω.
    #[inline(always)]
    pub fn conj(self) -> Self {
        Self {
            a: self.a,
            b: -self.d,
            c: -self.c,
            d: -self.b,
        }
    }

    /// Exact Euclidean norm squared |z|^2 as f64.
    pub fn norm_sq(self) -> f64 {
        let sqrt2 = std::f64::consts::SQRT_2;
        let re = self.a as f64 + (self.b as f64 - self.d as f64) / sqrt2;
        let im = self.c as f64 + (self.b as f64 + self.d as f64) / sqrt2;
        re * re + im * im
    }

    /// Converts to double-precision complex number.
    pub fn to_complex(self) -> Complex64 {
        let sqrt2 = std::f64::consts::SQRT_2;
        let re = self.a as f64 + (self.b as f64 - self.d as f64) / sqrt2;
        let im = self.c as f64 + (self.b as f64 + self.d as f64) / sqrt2;
        Complex64::new(re, im)
    }

    /// Exact division by sqrt(2) = (ω - ω^3) in Z[ω].
    ///
    /// Since (ω - ω^3)^2 = 2, multiplying by (ω - ω^3) and checking if coordinates are even
    /// yields exact quotient without floating-point rounding error.
    pub fn div_sqrt2(self) -> Option<Self> {
        let prod = self.mul(Self::SQRT2);
        if (prod.a & 1 == 0) && (prod.b & 1 == 0) && (prod.c & 1 == 0) && (prod.d & 1 == 0) {
            Some(Self {
                a: prod.a / 2,
                b: prod.b / 2,
                c: prod.c / 2,
                d: prod.d / 2,
            })
        } else {
            None
        }
    }

    /// Exact division by lambda = (1 + ω) in Z[ω].
    ///
    /// Since (1 + ω) * (1 - ω + ω^2 - ω^3) = 1 - ω^4 = 2,
    /// multiplying by (1 - ω + ω^2 - ω^3) and checking if coordinates are even
    /// yields exact quotient without floating-point rounding error.
    pub fn div_lambda(self) -> Option<Self> {
        let mult = Self::new(1, -1, 1, -1);
        let prod = self.mul(mult);
        if (prod.a & 1 == 0) && (prod.b & 1 == 0) && (prod.c & 1 == 0) && (prod.d & 1 == 0) {
            Some(Self {
                a: prod.a / 2,
                b: prod.b / 2,
                c: prod.c / 2,
                d: prod.d / 2,
            })
        } else {
            None
        }
    }

    /// Parity of element sum (a + b + c + d) mod 2.
    #[inline(always)]
    pub fn parity(self) -> i64 {
        (self.a + self.b + self.c + self.d) & 1
    }
}

impl std::ops::Add for ZOmega {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl std::ops::Sub for ZOmega {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

impl std::ops::Mul for ZOmega {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}
