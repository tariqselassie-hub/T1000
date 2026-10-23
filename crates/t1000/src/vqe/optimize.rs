// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! BFGS minimisation with an Armijo backtracking line search.

/// Outcome of a minimisation.
#[derive(Debug, Clone)]
pub struct OptResult {
    pub x: Vec<f64>,
    pub value: f64,
    pub iterations: usize,
    /// Number of objective/gradient evaluations.
    pub evaluations: usize,
    /// Largest absolute gradient component at `x`.
    pub grad_max: f64,
    pub converged: bool,
    /// Objective value after each iteration (for convergence plots).
    pub history: Vec<f64>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `H g` for the dense inverse-Hessian estimate `h` (row-major).
fn mat_vec(h: &[f64], g: &[f64]) -> Vec<f64> {
    let n = g.len();
    (0..n).map(|i| dot(&h[i * n..(i + 1) * n], g)).collect()
}

/// BFGS update of the inverse-Hessian estimate with step `s` and gradient change `y`.
fn bfgs_update(h: &mut [f64], s: &[f64], y: &[f64]) {
    let n = s.len();
    let sy = dot(s, y);
    if sy < 1e-12 * dot(s, s).sqrt() * dot(y, y).sqrt() {
        return; // curvature condition violated: keep the previous estimate
    }
    let hy = mat_vec(h, y);
    let yhy = dot(y, &hy);
    let factor = (1.0 + yhy / sy) / sy;
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] += factor * s[i] * s[j] - (hy[i] * s[j] + s[i] * hy[j]) / sy;
        }
    }
}

/// Minimises `f`, which returns the value and gradient at a point.
pub fn bfgs(
    mut f: impl FnMut(&[f64]) -> (f64, Vec<f64>),
    x0: Vec<f64>,
    max_iter: usize,
    grad_tol: f64,
) -> OptResult {
    let n = x0.len();
    let mut x = x0;
    let (mut value, mut grad) = f(&x);
    let mut evaluations = 1;
    let mut h = vec![0.0; n * n];
    (0..n).for_each(|i| h[i * n + i] = 1.0);
    let mut history = vec![value];
    let grad_max = |g: &[f64]| g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut iterations = 0;
    while iterations < max_iter && grad_max(&grad) > grad_tol {
        iterations += 1;
        let mut dir: Vec<f64> = mat_vec(&h, &grad).iter().map(|v| -v).collect();
        let mut slope = dot(&grad, &dir);
        if slope >= 0.0 {
            // Not a descent direction: restart from steepest descent.
            (0..n).for_each(|i| (0..n).for_each(|j| h[i * n + j] = f64::from(u8::from(i == j))));
            dir = grad.iter().map(|v| -v).collect();
            slope = -dot(&grad, &grad);
        }
        let mut step = if iterations == 1 {
            (1.0 / grad_max(&grad)).min(1.0)
        } else {
            1.0
        };
        let (new_x, new_value, new_grad) = loop {
            let trial: Vec<f64> = x.iter().zip(&dir).map(|(a, d)| a + step * d).collect();
            let (v, g) = f(&trial);
            evaluations += 1;
            if v <= value + 1e-4 * step * slope || step < 1e-14 {
                break (trial, v, g);
            }
            step *= 0.5;
        };
        let s: Vec<f64> = new_x.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = new_grad.iter().zip(&grad).map(|(a, b)| a - b).collect();
        bfgs_update(&mut h, &s, &y);
        let stalled = (value - new_value).abs() < 1e-15 && step < 1e-10;
        x = new_x;
        value = new_value;
        grad = new_grad;
        history.push(value);
        if stalled {
            break;
        }
    }
    let gm = grad_max(&grad);
    OptResult {
        x,
        value,
        iterations,
        evaluations,
        grad_max: gm,
        converged: gm <= grad_tol,
        history,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimises_a_quadratic_bowl() {
        let f = |x: &[f64]| {
            let v = (x[0] - 1.0).powi(2) + 10.0 * (x[1] + 2.0).powi(2) + 0.5 * x[0] * x[1];
            (
                v,
                vec![
                    2.0 * (x[0] - 1.0) + 0.5 * x[1],
                    20.0 * (x[1] + 2.0) + 0.5 * x[0],
                ],
            )
        };
        let r = bfgs(f, vec![5.0, 5.0], 100, 1e-10);
        assert!(r.converged, "grad {}", r.grad_max);
        // Stationary point: x = 1 - y/4 and 19.875 y = -40.5.
        let y = -40.5 / 19.875;
        assert!((r.x[1] - y).abs() < 1e-8 && (r.x[0] - (1.0 - y / 4.0)).abs() < 1e-8);
    }

    #[test]
    fn minimises_the_rosenbrock_valley() {
        let f = |x: &[f64]| {
            let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
            (
                a * a + 100.0 * b * b,
                vec![-2.0 * a - 400.0 * x[0] * b, 200.0 * b],
            )
        };
        let r = bfgs(f, vec![-1.2, 1.0], 500, 1e-8);
        assert!(r.converged);
        assert!((r.x[0] - 1.0).abs() < 1e-6 && (r.x[1] - 1.0).abs() < 1e-6);
        assert!(
            r.history.windows(2).all(|w| w[1] <= w[0] + 1e-12),
            "monotone decrease"
        );
    }

    #[test]
    fn an_already_stationary_start_returns_immediately() {
        let r = bfgs(
            |x: &[f64]| (x[0] * x[0], vec![2.0 * x[0]]),
            vec![0.0],
            10,
            1e-9,
        );
        assert!(r.converged && r.iterations == 0 && r.evaluations == 1);
    }
}
