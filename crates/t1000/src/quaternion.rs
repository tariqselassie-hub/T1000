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

    /// Construct a unit rotation quaternion from an axis and angle (radians).
    /// q = (cos(theta/2), sin(theta/2) * axis/|axis|)
    #[inline]
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let half = angle * 0.5;
        let (s, c) = half.sin_cos();
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if n < 1e-15 {
            return Self::IDENTITY;
        }
        let inv = s / n;
        Self {
            w: c,
            x: axis[0] * inv,
            y: axis[1] * inv,
            z: axis[2] * inv,
        }
    }

    /// Rotate a 3D vector v by this quaternion: v' = q * (0,v) * q*.
    #[inline]
    pub fn rotate_vector(self, v: [f64; 3]) -> [f64; 3] {
        let q = self.normalize();
        let vc = Self::new(0.0, v[0], v[1], v[2]);
        let rotated = q.mul(vc).mul(q.conjugate());
        [rotated.x, rotated.y, rotated.z]
    }

    /// Convert this (unit) quaternion to a 3x3 SO(3) rotation matrix.
    pub fn to_rotation_matrix(self) -> [[f64; 3]; 3] {
        let q = self.normalize();
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }

    /// Convert a 3x3 rotation matrix back to a unit quaternion (Shepperd).
    pub fn from_rotation_matrix(m: &[[f64; 3]; 3]) -> Self {
        let tr = m[0][0] + m[1][1] + m[2][2];
        let q = if tr > 0.0 {
            let s = 2.0 * (tr + 1.0).sqrt();
            Self {
                w: 0.25 * s,
                x: (m[2][1] - m[1][2]) / s,
                y: (m[0][2] - m[2][0]) / s,
                z: (m[1][0] - m[0][1]) / s,
            }
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).max(0.0).sqrt();
            let inv = if s > 1e-15 { 1.0 / s } else { 0.0 };
            Self {
                w: (m[2][1] - m[1][2]) * inv,
                x: 0.25 * s,
                y: (m[0][1] + m[1][0]) * inv,
                z: (m[0][2] + m[2][0]) * inv,
            }
        } else if m[1][1] > m[2][2] {
            let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).max(0.0).sqrt();
            let inv = if s > 1e-15 { 1.0 / s } else { 0.0 };
            Self {
                w: (m[0][2] - m[2][0]) * inv,
                x: (m[0][1] + m[1][0]) * inv,
                y: 0.25 * s,
                z: (m[1][2] + m[2][1]) * inv,
            }
        } else {
            let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).max(0.0).sqrt();
            let inv = if s > 1e-15 { 1.0 / s } else { 0.0 };
            Self {
                w: (m[1][0] - m[0][1]) * inv,
                x: (m[0][2] + m[2][0]) * inv,
                y: (m[1][2] + m[2][1]) * inv,
                z: 0.25 * s,
            }
        };
        q.normalize()
    }

    /// Spherical linear interpolation between two unit quaternions.
    pub fn slerp(self, other: Self, t: f64) -> Self {
        let mut cos_omega = self.dot(other).clamp(-1.0, 1.0);
        let mut other = other;
        if cos_omega < 0.0 {
            cos_omega = -cos_omega;
            other = other.scale(-1.0);
        }
        if cos_omega > 0.9999 {
            // Linear fallback
            return self
                .scale(1.0 - t)
                .add(other.scale(t))
                .normalize();
        }
        let omega = cos_omega.acos();
        let sin_omega = omega.sin();
        let a = ((1.0 - t) * omega).sin() / sin_omega;
        let b = (t * omega).sin() / sin_omega;
        self.scale(a).add(other.scale(b))
    }

    /// Convert this quaternion to an SU(2) single-qubit unitary (a, b) such that
    /// U = [[a, -conj(b)], [b, conj(a)]] with a = w + i x, b = y + i z.
    /// Returns (a_re, a_im, b_re, b_im).
    pub fn to_su2(self) -> (f64, f64, f64, f64) {
        let q = self.normalize();
        (q.w, q.x, q.y, q.z)
    }

    /// Apply this quaternion as an SU(2) rotation to a single-qubit state
    /// (re0, im0, re1, im1) -> (re0', im0', re1', im1').
    pub fn apply_single_qubit(&self, re0: f64, im0: f64, re1: f64, im1: f64) -> (f64, f64, f64, f64) {
        let q = self.normalize();
        let (a_re, a_im, b_re, b_im) = (q.w, q.x, q.y, q.z);
        // U = [[a, -conj(b)], [b, conj(a)]], a=w+ix, b=y+iz
        // new0 = a*psi0 - conj(b)*psi1
        // new1 = b*psi0 + conj(a)*psi1
        // a*psi0
        let a0_re = a_re * re0 - a_im * im0;
        let a0_im = a_re * im0 + a_im * re0;
        // conj(b) = b_re - i b_im ; -conj(b)*psi1 = (-b_re + i b_im)*psi1
        let nb_re = -b_re;
        let nb_im = b_im;
        let t_re = nb_re * re1 - nb_im * im1;
        let t_im = nb_re * im1 + nb_im * re1;
        let n0_re = a0_re + t_re;
        let n0_im = a0_im + t_im;
        // b*psi0
        let b0_re = b_re * re0 - b_im * im0;
        let b0_im = b_re * im0 + b_im * re0;
        // conj(a)*psi1 = (a_re - i a_im)*psi1
        let ca_re = a_re;
        let ca_im = -a_im;
        let c1_re = ca_re * re1 - ca_im * im1;
        let c1_im = ca_re * im1 + ca_im * re1;
        let n1_re = b0_re + c1_re;
        let n1_im = b0_im + c1_im;
        (n0_re, n0_im, n1_re, n1_im)
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

    #[test]
    fn test_axis_angle_and_rotate_vector() {
        // 90 deg around Z rotates X -> Y
        let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        let v = [1.0, 0.0, 0.0];
        let vr = q.rotate_vector(v);
        assert!((vr[0]).abs() < 1e-12);
        assert!((vr[1] - 1.0).abs() < 1e-12);
        assert!((vr[2]).abs() < 1e-12);
    }

    #[test]
    fn test_rotation_matrix_roundtrip() {
        let q = Quaternion::from_axis_angle([0.577, 0.577, 0.577], 1.25);
        let m = q.to_rotation_matrix();
        let q2 = Quaternion::from_rotation_matrix(&m);
        assert!((q.w - q2.w).abs() < 1e-10);
        assert!((q.x - q2.x).abs() < 1e-10);
        assert!((q.y - q2.y).abs() < 1e-10);
        assert!((q.z - q2.z).abs() < 1e-10);
    }

    #[test]
    fn test_slerp_endpoints() {
        let a = Quaternion::from_axis_angle([0.0, 0.0, 1.0], 0.0);
        let b = Quaternion::from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::PI);
        let mid = a.slerp(b, 0.5);
        // angle should be ~pi/2
        let q = Quaternion::from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        assert!((mid.w - q.w).abs() < 1e-10);
        assert!((mid.z - q.z).abs() < 1e-10);
    }

    #[test]
    fn test_apply_single_qubit_identity() {
        // identity quaternion -> identity gate
        let q = Quaternion::IDENTITY;
        let (r0, i0, r1, i1) = q.apply_single_qubit(0.3, 0.4, 0.5, 0.6);
        assert!((r0 - 0.3).abs() < 1e-12);
        assert!((i0 - 0.4).abs() < 1e-12);
        assert!((r1 - 0.5).abs() < 1e-12);
        assert!((i1 - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_apply_single_qubit_unitarity() {
        // norm preservation
        let q = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 0.7);
        let (r0, i0, r1, i1) = q.apply_single_qubit(0.3, 0.4, 0.5, 0.6);
        let n_before = 0.3 * 0.3 + 0.4 * 0.4 + 0.5 * 0.5 + 0.6 * 0.6;
        let n_after = r0 * r0 + i0 * i0 + r1 * r1 + i1 * i1;
        assert!((n_before - n_after).abs() < 1e-12);
    }
}
