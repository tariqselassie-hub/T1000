// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 8-Node Throwing-Jack & Pulsar Topology Specification
//!
//! Models the spatial node configuration of a 6-pronged throwing jack
//! where the two central nodes coalesce into a singular shared fulcrum at Node (4, 5).

use serde::{Deserialize, Serialize};

/// 3D Cartesian vector with double precision.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vector3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3D {
    /// Creates a new 3D vector with the given coordinates.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Returns the origin zero vector (0, 0, 0).
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Computes the Euclidean norm (length) of the vector.
    #[inline]
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Computes the vector cross product: self x other.
    #[inline]
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Computes the Euclidean dot product: self . other.
    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Scales the vector uniformly by scalar `s`.
    #[inline]
    pub fn scale(&self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Adds another vector component-wise: self + other.
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    /// Subtracts another vector component-wise: self - other.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

/// Represents a physical node in the 8-node throwing jack topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JackNode {
    pub id: usize,
    pub name: String,
    pub role: String,
    pub initial_pos: Vector3D,
    pub current_pos: Vector3D,
    pub mass: f64,
    pub is_shared_core: bool,
}

/// The complete 8-node throwing jack topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JackTopology {
    pub nodes: Vec<JackNode>,
    pub center_of_mass: Vector3D,
    pub arm_length: f64,
}

impl JackTopology {
    /// Generates the canonical 8-node throwing jack configuration.
    ///
    /// - Nodes 1, 2, 3: Upper Polar Jet A (z > 0)
    /// - Node 4, 5: Shared Central Fulcrum (z = 0, fused at origin)
    /// - Nodes 6, 7, 8: Lower Polar Jet B (z < 0, inverted parity)
    pub fn new(arm_length: f64) -> Self {
        let r = arm_length;
        let cos30 = (30.0_f64).to_radians().cos();
        let sin30 = (30.0_f64).to_radians().sin();
        let z_h = r * (2.0_f64 / 3.0).sqrt(); // sqrt(2/3) for symmetric tetrahedral spread

        let nodes = vec![
            JackNode {
                id: 1,
                name: "Node 1".to_string(),
                role: "Polar Jet A (Prong 1)".to_string(),
                initial_pos: Vector3D::new(0.0, r, z_h),
                current_pos: Vector3D::new(0.0, r, z_h),
                mass: 1.0,
                is_shared_core: false,
            },
            JackNode {
                id: 2,
                name: "Node 2".to_string(),
                role: "Polar Jet A (Prong 2)".to_string(),
                initial_pos: Vector3D::new(-r * cos30, -r * sin30, z_h),
                current_pos: Vector3D::new(-r * cos30, -r * sin30, z_h),
                mass: 1.0,
                is_shared_core: false,
            },
            JackNode {
                id: 3,
                name: "Node 3".to_string(),
                role: "Polar Jet A (Prong 3)".to_string(),
                initial_pos: Vector3D::new(r * cos30, -r * sin30, z_h),
                current_pos: Vector3D::new(r * cos30, -r * sin30, z_h),
                mass: 1.0,
                is_shared_core: false,
            },
            JackNode {
                id: 4,
                name: "Node 4".to_string(),
                role: "Shared Fulcrum Core (Primary)".to_string(),
                initial_pos: Vector3D::zero(),
                current_pos: Vector3D::zero(),
                mass: 2.0,
                is_shared_core: true,
            },
            JackNode {
                id: 5,
                name: "Node 5".to_string(),
                role: "Shared Fulcrum Core (Secondary)".to_string(),
                initial_pos: Vector3D::zero(),
                current_pos: Vector3D::zero(),
                mass: 2.0,
                is_shared_core: true,
            },
            JackNode {
                id: 6,
                name: "Node 6".to_string(),
                role: "Polar Jet B (Prong 1, Antipodal)".to_string(),
                initial_pos: Vector3D::new(0.0, -r, -z_h),
                current_pos: Vector3D::new(0.0, -r, -z_h),
                mass: 1.0,
                is_shared_core: false,
            },
            JackNode {
                id: 7,
                name: "Node 7".to_string(),
                role: "Polar Jet B (Prong 2, Antipodal)".to_string(),
                initial_pos: Vector3D::new(r * cos30, r * sin30, -z_h),
                current_pos: Vector3D::new(r * cos30, r * sin30, -z_h),
                mass: 1.0,
                is_shared_core: false,
            },
            JackNode {
                id: 8,
                name: "Node 8".to_string(),
                role: "Polar Jet B (Prong 3, Antipodal)".to_string(),
                initial_pos: Vector3D::new(-r * cos30, r * sin30, -z_h),
                current_pos: Vector3D::new(-r * cos30, r * sin30, -z_h),
                mass: 1.0,
                is_shared_core: false,
            },
        ];

        let com = Self::compute_center_of_mass(&nodes);

        Self {
            nodes,
            center_of_mass: com,
            arm_length,
        }
    }

    /// Evaluates the center of mass vector R_CM = (sum m_i r_i) / (sum m_i).
    pub fn compute_center_of_mass(nodes: &[JackNode]) -> Vector3D {
        let mut total_mass = 0.0;
        let mut sum_mr = Vector3D::zero();

        for n in nodes {
            total_mass += n.mass;
            sum_mr = sum_mr.add(&n.current_pos.scale(n.mass));
        }

        if total_mass > 0.0 {
            sum_mr.scale(1.0 / total_mass)
        } else {
            Vector3D::zero()
        }
    }
}
