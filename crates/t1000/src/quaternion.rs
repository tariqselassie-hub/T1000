// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 256-bit AVX2 Vectorized Quaternionic Mathematics Engine.
//!
//! Provides the fundamental double-precision quaternion algebra $\mathbb{H} = \mathbb{R} \oplus \mathbb{R}\mathbf{i} \oplus \mathbb{R}\mathbf{j} \oplus \mathbb{R}\mathbf{k}$
//! optimized for quantum state representations, spinor evolutions, and the Second Hopf Fibration.
//! A quaternion $(w, x, y, z)$ exactly fits a single 256-bit AVX2 register (`__m256d`).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use serde::{Deserialize, Serialize};

/// Double-precision quaternion $q = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k} \in \mathbb{H}$.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(C, align(32))]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quaternion {
    /// Identity quaternion $1 + 0\mathbf{i} + 0\mathbf{j} + 0\mathbf{k}$.
    pub const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Zero quaternion $0 + 0\mathbf{i} + 0\mathbf{j} + 0\mathbf{k}$.
    pub const ZERO: Self = Self {
        w: 0.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Creates a quaternion from explicit real and imaginary components.
    #[inline(always)]
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Cayley-Dickson construction: $q = c_0 + c_1 \mathbf{j}$ from complex numbers
    /// $c_0 = a_0 + i b_0$ and $c_1 = a_1 + i b_1$.
    #[inline(always)]
    pub fn from_complex_pair(c0_re: f64, c0_im: f64, c1_re: f64, c1_im: f64) -> Self {
        // (a0 + b0*i) + (a1 + b1*i)*j = a0 + b0*i + a1*j + b1*k
        Self {
            w: c0_re,
            x: c0_im,
            y: c1_re,
            z: c1_im,
        }
    }

    /// Converts quaternion into a complex pair $(c_0, c_1)$ under Cayley-Dickson embedding.
    #[inline(always)]
    pub fn to_complex_pair(self) -> (num_complex::Complex64, num_complex::Complex64) {
        (
            num_complex::Complex64::new(self.w, self.x),
            num_complex::Complex64::new(self.y, self.z),
        )
    }

    /// Quaternionic conjugate: $q^* = w - x\mathbf{i} - y\mathbf{j} - z\mathbf{k}$.
    #[inline(always)]
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Quaternionic squared Euclidean norm: $|q|^2 = w^2 + x^2 + y^2 + z^2$.
    #[inline(always)]
    pub fn norm_sq(self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Quaternionic Euclidean norm: $|q| = \sqrt{|q|^2}$.
    #[inline(always)]
    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Normalized unit quaternion: $q / |q|$. Returns zero if $|q| \approx 0$.
    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n > 1e-15 {
            let inv = 1.0 / n;
            Self {
                w: self.w * inv,
                x: self.x * inv,
                y: self.y * inv,
                z: self.z * inv,
            }
        } else {
            Self::ZERO
        }
    }

    /// Quaternionic inverse: $q^{-1} = q^* / |q|^2$.
    #[inline]
    pub fn inverse(self) -> Option<Self> {
        let n_sq = self.norm_sq();
        if n_sq > 1e-15 {
            let inv = 1.0 / n_sq;
            Some(Self {
                w: self.w * inv,
                x: -self.x * inv,
                y: -self.y * inv,
                z: -self.z * inv,
            })
        } else {
            None
        }
    }

    /// Hamilton product: $p \times q \in \mathbb{H}$.
    ///
    /// Automatically invokes AVX2+FMA SIMD vectorization on supported x86_64 CPUs.
    #[inline(always)]
    pub fn mul(self, rhs: Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { self.mul_avx2(rhs) };
            }
        }
        self.mul_scalar(rhs)
    }

    /// Scalar Hamilton multiplication:
    /// w = p.w*q.w - p.x*q.x - p.y*q.y - p.z*q.z
    /// x = p.w*q.x + p.x*q.w + p.y*q.z - p.z*q.y
    /// y = p.w*q.y - p.x*q.z + p.y*q.w + p.z*q.x
    /// z = p.w*q.z + p.x*q.y - p.y*q.x + p.z*q.w
    #[inline(always)]
    pub fn mul_scalar(self, rhs: Self) -> Self {
        let w = self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z;
        let x = self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y;
        let y = self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x;
        let z = self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w;
        Self { w, x, y, z }
    }

    /// Hardware-accelerated 256-bit AVX2+FMA Hamilton product.
    ///
    /// # Safety
    /// Caller must ensure AVX2 and FMA CPU instruction sets are supported.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn mul_avx2(self, rhs: Self) -> Self {
        // Broadcast self components across 4-lane double registers
        let pw = _mm256_set1_pd(self.w);
        let px = _mm256_set1_pd(self.x);
        let py = _mm256_set1_pd(self.y);
        let pz = _mm256_set1_pd(self.z);

        // Load rhs into double-precision vector [w, x, y, z]
        let q_orig = _mm256_loadu_pd(&rhs.w as *const f64);

        // Term 1: pw * [w, x, y, z]
        let mut res = _mm256_mul_pd(pw, q_orig);

        // Term 2: px * [-x, w, -z, y]
        let q_px = _mm256_set_pd(rhs.y, -rhs.z, rhs.w, -rhs.x);
        res = _mm256_fmadd_pd(px, q_px, res);

        // Term 3: py * [-y, z, w, -x]
        let q_py = _mm256_set_pd(-rhs.x, rhs.w, rhs.z, -rhs.y);
        res = _mm256_fmadd_pd(py, q_py, res);

        // Term 4: pz * [-z, -y, x, w]
        let q_pz = _mm256_set_pd(rhs.w, rhs.x, -rhs.y, -rhs.z);
        res = _mm256_fmadd_pd(pz, q_pz, res);

        let mut out = Self::ZERO;
        _mm256_storeu_pd(&mut out.w as *mut f64, res);
        out
    }

    /// Real scalar addition.
    #[inline(always)]
    pub fn add(self, rhs: Self) -> Self {
        Self {
            w: self.w + rhs.w,
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }

    /// Real scalar subtraction.
    #[inline(always)]
    pub fn sub(self, rhs: Self) -> Self {
        Self {
            w: self.w - rhs.w,
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }

    /// Multiplication by a real scalar $s \in \mathbb{R}$.
    #[inline(always)]
    pub fn scale(self, s: f64) -> Self {
        Self {
            w: self.w * s,
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Quaternionic inner product $\langle p, q \rangle = \text{Re}(p^* q) = p_w q_w + p_x q_x + p_y q_y + p_z q_z$.
    #[inline(always)]
    pub fn dot(self, rhs: Self) -> f64 {
        self.w * rhs.w + self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl std::ops::Add for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        self.add(rhs)
    }
}

impl std::ops::Sub for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        self.sub(rhs)
    }
}

impl std::ops::Mul for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        self.mul(rhs)
    }
}

impl std::ops::Mul<f64> for Quaternion {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: f64) -> Self {
        self.scale(s)
    }
}

impl std::ops::Mul<Quaternion> for f64 {
    type Output = Quaternion;
    #[inline(always)]
    fn mul(self, q: Quaternion) -> Quaternion {
        q.scale(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_identity_and_units() {
        let one = Quaternion::IDENTITY;
        let qi = Quaternion::new(0.0, 1.0, 0.0, 0.0);
        let qj = Quaternion::new(0.0, 0.0, 1.0, 0.0);
        let qk = Quaternion::new(0.0, 0.0, 0.0, 1.0);

        // i * i = j * j = k * k = -1
        assert_eq!(qi * qi, Quaternion::new(-1.0, 0.0, 0.0, 0.0));
        assert_eq!(qj * qj, Quaternion::new(-1.0, 0.0, 0.0, 0.0));
        assert_eq!(qk * qk, Quaternion::new(-1.0, 0.0, 0.0, 0.0));

        // i * j = k, j * k = i, k * i = j
        assert_eq!(qi * qj, qk);
        assert_eq!(qj * qk, qi);
        assert_eq!(qk * qi, qj);

        // Anti-commutation: j * i = -k
        assert_eq!(qj * qi, Quaternion::new(0.0, 0.0, 0.0, -1.0));

        // Identity check
        assert_eq!(one * qi, qi);
        assert_eq!(qi * one, qi);
    }

    #[test]
    fn test_cayley_dickson_roundtrip() {
        // c0 = 1.5 + 2.5i, c1 = -0.5 + 3.0i
        let q = Quaternion::from_complex_pair(1.5, 2.5, -0.5, 3.0);
        let (c0, c1) = q.to_complex_pair();
        assert!((c0.re - 1.5).abs() < 1e-14);
        assert!((c0.im - 2.5).abs() < 1e-14);
        assert!((c1.re - (-0.5)).abs() < 1e-14);
        assert!((c1.im - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_norm_and_inverse() {
        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let norm_sq = q.norm_sq();
        assert_eq!(norm_sq, 30.0);
        assert!((q.norm() - 30.0f64.sqrt()).abs() < 1e-14);

        let q_inv = q.inverse().expect("Inverse exists");
        let prod = q * q_inv;
        assert!((prod.w - 1.0).abs() < 1e-14);
        assert!(prod.x.abs() < 1e-14);
        assert!(prod.y.abs() < 1e-14);
        assert!(prod.z.abs() < 1e-14);
    }

    #[test]
    fn test_avx2_vs_scalar_consistency() {
        let p = Quaternion::new(0.4, -0.7, 1.2, 0.5);
        let q = Quaternion::new(-0.8, 1.5, -0.3, 0.9);

        let res_scalar = p.mul_scalar(q);
        let res_default = p.mul(q);

        assert!((res_scalar.w - res_default.w).abs() < 1e-14);
        assert!((res_scalar.x - res_default.x).abs() < 1e-14);
        assert!((res_scalar.y - res_default.y).abs() < 1e-14);
        assert!((res_scalar.z - res_default.z).abs() < 1e-14);
    }
}
