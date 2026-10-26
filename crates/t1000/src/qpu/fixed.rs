// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Fixed-point arithmetic of the 1000_QPU datapath.
//!
//! * A state component is a signed 32-bit integer holding a value scaled by `2^state_frac`
//!   (the hardware never needs `state_frac`; it only saturates at 32 bits).
//! * A gate coefficient is a signed 25-bit integer scaled by `2^COEF_FRAC`, the width of the
//!   25-bit port of a DSP48E1 multiplier.
//! * A rotated component is `round_half_up(Σ coefficient · component / 2^COEF_FRAC)`, saturated to
//!   32 bits. Products and sums are exact (no intermediate rounding).

/// Bits of a stored state component.
pub const STATE_BITS: u32 = 32;
/// Bits of a gate coefficient.
pub const COEF_BITS: u32 = 25;
/// Fractional bits of a gate coefficient (the right shift after the multiply-accumulate).
pub const COEF_FRAC: u32 = 23;

pub const STATE_MAX: i64 = (1 << (STATE_BITS - 1)) - 1;
pub const STATE_MIN: i64 = -(1 << (STATE_BITS - 1));
pub const COEF_MAX: i64 = (1 << (COEF_BITS - 1)) - 1;
pub const COEF_MIN: i64 = -(1 << (COEF_BITS - 1));

/// Quantises `x` to a coefficient scaled by `2^COEF_FRAC`, rounding to nearest and saturating.
pub fn quantize_coef(x: f64) -> i64 {
    ((x * (1u64 << COEF_FRAC) as f64).round() as i64).clamp(COEF_MIN, COEF_MAX)
}

/// Quantises `x` to a state component scaled by `2^frac`, rounding to nearest and saturating.
pub fn quantize_state(x: f64, frac: u32) -> i64 {
    ((x * (1u64 << frac) as f64).round() as i64).clamp(STATE_MIN, STATE_MAX)
}

/// The real value of a state component.
pub fn dequantize_state(v: i64, frac: u32) -> f64 {
    v as f64 / (1u64 << frac) as f64
}

/// The real value of a coefficient.
pub fn dequantize_coef(v: i64) -> f64 {
    v as f64 / (1u64 << COEF_FRAC) as f64
}

/// `round_half_up(acc / 2^COEF_FRAC)` saturated to the state range.
pub fn round_and_saturate(acc: i128) -> i64 {
    let rounded = (acc + (1i128 << (COEF_FRAC - 1))) >> COEF_FRAC;
    rounded.clamp(i128::from(STATE_MIN), i128::from(STATE_MAX)) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficient_one_is_exactly_representable_only_below_the_signed_limit() {
        assert_eq!(quantize_coef(0.5), 1 << 22);
        assert_eq!(quantize_coef(-0.5), -(1 << 22));
        // +1.0 is 2^23, well inside the 25-bit range of Q2.23.
        assert_eq!(quantize_coef(1.0), 1 << 23);
        assert_eq!(quantize_coef(100.0), COEF_MAX);
    }

    #[test]
    fn state_quantisation_rounds_to_nearest_and_saturates() {
        assert_eq!(quantize_state(1.0, 28), 1 << 28);
        assert_eq!(quantize_state(0.49999999, 1), 1);
        assert_eq!(quantize_state(1e9, 28), STATE_MAX);
        assert_eq!(quantize_state(-1e9, 28), STATE_MIN);
        assert!((dequantize_state(quantize_state(0.123456789, 28), 28) - 0.123456789).abs() < 2e-9);
    }

    #[test]
    fn rounding_is_half_up_and_saturating() {
        let half = 1i128 << (COEF_FRAC - 1);
        assert_eq!(round_and_saturate(half), 1);
        assert_eq!(round_and_saturate(half - 1), 0);
        assert_eq!(round_and_saturate(-half), 0, "ties round toward +infinity");
        assert_eq!(round_and_saturate(-half - 1), -1);
        assert_eq!(
            round_and_saturate(i128::from(STATE_MAX) << (COEF_FRAC + 4)),
            STATE_MAX
        );
        assert_eq!(
            round_and_saturate(i128::from(STATE_MIN) << (COEF_FRAC + 4)),
            STATE_MIN
        );
    }
}
