// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! # 2D Plaquette-Zipper & Wilson Flux Tableau Module
//!
//! Lifts the 1D Bloch-Zipper into 2D via discrete Stokes' theorem and
//! algebraic involution cancellation along shared spokes ($Z_e^2 = I$).

pub mod geometry;
pub mod nighthawk_dual;
pub mod tableau;
pub mod zipper2d;

pub use geometry::{GaugeLink, Plaquette, PlaquetteLattice};
pub use nighthawk_dual::{
    CouplerMask, NighthawkDualGraph, PlaquetteNode, PlaquetteZipperWindow, NUM_COUPLERS,
    NUM_PLAQUETTES, NUM_VERTICES,
};
pub use tableau::{DiscreteStokesEngine, StokesPressureReport, WilsonFluxTableau};
pub use zipper2d::{PlaquetteStepTelemetry, PlaquetteZipper2D, PlaquetteZipperReport};
