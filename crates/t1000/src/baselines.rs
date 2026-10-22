//! Classical regression baselines for judging QNN property-prediction results.
//!
//! Every ZINC experiment should be read against these: if a QNN readout does not beat ridge
//! regression on the same raw features and the same train/test split, the circuit is not
//! adding predictive value.
//!
//! - **Mean predictor**: predicts the training mean (R^2 ~ 0); the floor any model must clear.
//! - **Ridge (linear)**: ridge regression on the standardized raw feature vector.
//! - **Ridge (quadratic)**: ridge on raw features plus all pairwise products, a classical
//!   counterpart to readouts that use two-body (ZZ) correlators.
//!
//! The ridge penalty is chosen on a deterministic validation split of the training data
//! (every fifth sample) and the model is then refit on the full training set. The test set
//! is never used for model selection.

use serde::{Deserialize, Serialize};

/// Target names for the standard ZINC (logP, QED, SAS) regression task.
pub const ZINC_TARGETS: [&str; 3] = ["logP", "QED", "SAS"];

/// Penalty grid searched by [`fit_ridge_validated`] (applied to standardized features).
pub const RIDGE_LAMBDAS: [f64; 9] = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0];

/// Regression quality on one target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetMetrics {
    pub target: String,
    pub mae: f64,
    pub mse: f64,
    pub r2: f64,
    pub pearson_r: f64,
}

/// One baseline model evaluated on the test set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineResult {
    pub model: String,
    pub n_features: usize,
    /// Ridge penalty selected on the validation split (`None` for the mean predictor).
    pub lambda: Option<f64>,
    pub metrics: Vec<TargetMetrics>,
}

/// MAE, MSE, R^2 and Pearson r for each target column.
pub fn regression_metrics(
    preds: &[Vec<f64>],
    targets: &[Vec<f64>],
    names: &[&str],
) -> Vec<TargetMetrics> {
    assert_eq!(
        preds.len(),
        targets.len(),
        "prediction/target row count mismatch"
    );
    assert!(!targets.is_empty(), "cannot score an empty set");
    let k = names.len();
    let n = targets.len() as f64;
    (0..k)
        .map(|t| {
            let p: Vec<f64> = preds.iter().map(|row| row[t]).collect();
            let y: Vec<f64> = targets.iter().map(|row| row[t]).collect();
            let mae = p.iter().zip(&y).map(|(a, b)| (a - b).abs()).sum::<f64>() / n;
            let mse = p
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                / n;
            let y_mean = y.iter().sum::<f64>() / n;
            let p_mean = p.iter().sum::<f64>() / n;
            let ss_tot = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>();
            let r2 = if ss_tot > 0.0 {
                1.0 - mse * n / ss_tot
            } else {
                0.0
            };
            let cov = p
                .iter()
                .zip(&y)
                .map(|(a, b)| (a - p_mean) * (b - y_mean))
                .sum::<f64>();
            let var_p = p.iter().map(|v| (v - p_mean).powi(2)).sum::<f64>();
            let pearson_r = if var_p > 0.0 && ss_tot > 0.0 {
                cov / (var_p * ss_tot).sqrt()
            } else {
                0.0
            };
            TargetMetrics {
                target: names[t].to_string(),
                mae,
                mse,
                r2,
                pearson_r,
            }
        })
        .collect()
}

/// Raw features followed by all products `x_i * x_j` for `i <= j`.
pub fn quadratic_features(x: &[f64]) -> Vec<f64> {
    let d = x.len();
    let mut out = Vec::with_capacity(d + d * (d + 1) / 2);
    out.extend_from_slice(x);
    for i in 0..d {
        for j in i..d {
            out.push(x[i] * x[j]);
        }
    }
    out
}

/// Multi-target ridge regression on standardized features with an unpenalized intercept.
///
/// Minimizes `(1/n) * ||y - ybar - Z w||^2 + lambda * ||w||^2` per target, where `Z` holds the
/// z-scored features (constant features are centered only and end up with zero weight).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RidgeRegression {
    pub lambda: f64,
    pub feature_mean: Vec<f64>,
    pub feature_scale: Vec<f64>,
    pub target_mean: Vec<f64>,
    /// `coef[t][j]`: weight of standardized feature `j` for target `t`.
    pub coef: Vec<Vec<f64>>,
}

/// Sufficient statistics for ridge fits that share a design matrix but differ in `lambda`.
pub(crate) struct RidgeProblem {
    feature_mean: Vec<f64>,
    feature_scale: Vec<f64>,
    target_mean: Vec<f64>,
    /// `Z^T Z / n`, row-major `d x d`.
    gram: Vec<f64>,
    /// `Z^T (y - ybar) / n`, one column per target.
    cross: Vec<Vec<f64>>,
}

impl RidgeProblem {
    pub(crate) fn new(x: &[Vec<f64>], y: &[Vec<f64>]) -> Self {
        assert_eq!(x.len(), y.len(), "feature/target row count mismatch");
        assert!(!x.is_empty(), "cannot fit on an empty training set");
        let n = x.len();
        let d = x[0].len();
        let k = y[0].len();
        assert!(x.iter().all(|r| r.len() == d), "ragged feature rows");
        assert!(y.iter().all(|r| r.len() == k), "ragged target rows");
        let nf = n as f64;

        let mut feature_mean = vec![0.0; d];
        for row in x {
            for (m, v) in feature_mean.iter_mut().zip(row) {
                *m += v / nf;
            }
        }
        let mut feature_scale = vec![0.0; d];
        for row in x {
            for j in 0..d {
                feature_scale[j] += (row[j] - feature_mean[j]).powi(2) / nf;
            }
        }
        let constant: Vec<bool> = feature_scale.iter().map(|&v| v <= 1e-24).collect();
        for s in feature_scale.iter_mut() {
            *s = if *s > 1e-24 { s.sqrt() } else { 1.0 };
        }
        let mut target_mean = vec![0.0; k];
        for row in y {
            for (m, v) in target_mean.iter_mut().zip(row) {
                *m += v / nf;
            }
        }

        // Raw second moments over the non-zero entries of each row (fingerprints are sparse),
        // standardised afterwards: Z^T Z / n = (X^T X / n - mean mean^T) / (scale scale^T) and,
        // because the centred targets sum to zero, Z^T y_c / n = X^T y_c / (n * scale).
        let mut gram = vec![0.0; d * d];
        let mut cross = vec![vec![0.0; d]; k];
        let mut nz: Vec<usize> = Vec::with_capacity(d);
        for (row, yrow) in x.iter().zip(y) {
            nz.clear();
            nz.extend((0..d).filter(|&j| row[j] != 0.0));
            for (pos, &i) in nz.iter().enumerate() {
                let xi = row[i];
                let g = &mut gram[i * d..i * d + d];
                for &j in &nz[pos..] {
                    g[j] += xi * row[j];
                }
            }
            for t in 0..k {
                let yc = yrow[t] - target_mean[t];
                for &j in &nz {
                    cross[t][j] += row[j] * yc;
                }
            }
        }
        for i in 0..d {
            for j in i..d {
                let v = (gram[i * d + j] / nf - feature_mean[i] * feature_mean[j])
                    / (feature_scale[i] * feature_scale[j]);
                gram[i * d + j] = v;
                gram[j * d + i] = v;
            }
        }
        for col in cross.iter_mut() {
            for (j, v) in col.iter_mut().enumerate() {
                *v /= nf * feature_scale[j];
            }
        }
        // Constant features carry no information; zero them exactly so rounding residue in the
        // moment subtraction cannot be amplified by a small penalty.
        for j in (0..d).filter(|&j| constant[j]) {
            for i in 0..d {
                gram[i * d + j] = 0.0;
                gram[j * d + i] = 0.0;
            }
            for col in cross.iter_mut() {
                col[j] = 0.0;
            }
        }

        Self {
            feature_mean,
            feature_scale,
            target_mean,
            gram,
            cross,
        }
    }

    pub(crate) fn solve(&self, lambda: f64) -> RidgeRegression {
        assert!(
            lambda > 0.0,
            "ridge penalty must be positive (got {lambda})"
        );
        let d = self.feature_mean.len();
        let mut a = self.gram.clone();
        for i in 0..d {
            a[i * d + i] += lambda;
        }
        let l = cholesky(&a, d).expect("Z^T Z / n + lambda I is positive definite for lambda > 0");
        let coef = self
            .cross
            .iter()
            .map(|b| cholesky_solve(&l, d, b))
            .collect();
        RidgeRegression {
            lambda,
            feature_mean: self.feature_mean.clone(),
            feature_scale: self.feature_scale.clone(),
            target_mean: self.target_mean.clone(),
            coef,
        }
    }
}

impl RidgeRegression {
    /// Fits with a fixed penalty `lambda > 0`.
    pub fn fit(x: &[Vec<f64>], y: &[Vec<f64>], lambda: f64) -> Self {
        RidgeProblem::new(x, y).solve(lambda)
    }

    pub fn predict_one(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.feature_mean.len(), "feature count mismatch");
        self.coef
            .iter()
            .zip(&self.target_mean)
            .map(|(w, ybar)| {
                ybar + w
                    .iter()
                    .zip(x)
                    .zip(self.feature_mean.iter().zip(&self.feature_scale))
                    .map(|((wj, xj), (m, s))| wj * (xj - m) / s)
                    .sum::<f64>()
            })
            .collect()
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        x.iter().map(|row| self.predict_one(row)).collect()
    }
}

/// Fits ridge regression, selecting `lambda` from `lambdas` on a validation split made of
/// every fifth training row, then refits on all rows with the chosen penalty.
/// Selection minimizes the mean over targets of validation MSE / validation variance.
pub fn fit_ridge_validated(x: &[Vec<f64>], y: &[Vec<f64>], lambdas: &[f64]) -> RidgeRegression {
    assert!(!lambdas.is_empty(), "need at least one candidate lambda");
    assert!(
        x.len() >= 10,
        "need at least 10 training rows for a validation split"
    );
    let (mut xt, mut yt, mut xv, mut yv) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, (xr, yr)) in x.iter().zip(y).enumerate() {
        if i % 5 == 4 {
            xv.push(xr.clone());
            yv.push(yr.clone());
        } else {
            xt.push(xr.clone());
            yt.push(yr.clone());
        }
    }
    let k = y[0].len();
    let names: Vec<String> = (0..k).map(|t| t.to_string()).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let variances: Vec<f64> = (0..k)
        .map(|t| {
            let m = yv.iter().map(|r| r[t]).sum::<f64>() / yv.len() as f64;
            (yv.iter().map(|r| (r[t] - m).powi(2)).sum::<f64>() / yv.len() as f64).max(1e-300)
        })
        .collect();

    let problem = RidgeProblem::new(&xt, &yt);
    let best = lambdas
        .iter()
        .map(|&lambda| {
            let model = problem.solve(lambda);
            let m = regression_metrics(&model.predict(&xv), &yv, &name_refs);
            let score = m
                .iter()
                .zip(&variances)
                .map(|(tm, v)| tm.mse / v)
                .sum::<f64>()
                / k as f64;
            (lambda, score)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(lambda, _)| lambda)
        .expect("lambda grid is non-empty");
    RidgeRegression::fit(x, y, best)
}

/// Mean predictor, linear ridge and quadratic ridge, trained on `train` and scored on `test`.
/// Each item is `(features, targets)`; all rows must have the same lengths.
pub fn evaluate_baselines(
    train: &[(Vec<f64>, Vec<f64>)],
    test: &[(Vec<f64>, Vec<f64>)],
    target_names: &[&str],
) -> Vec<BaselineResult> {
    let x_train: Vec<Vec<f64>> = train.iter().map(|(x, _)| x.clone()).collect();
    let y_train: Vec<Vec<f64>> = train.iter().map(|(_, y)| y.clone()).collect();
    let x_test: Vec<Vec<f64>> = test.iter().map(|(x, _)| x.clone()).collect();
    let y_test: Vec<Vec<f64>> = test.iter().map(|(_, y)| y.clone()).collect();
    let d = x_train[0].len();

    let k = target_names.len();
    let mean: Vec<f64> = (0..k)
        .map(|t| y_train.iter().map(|r| r[t]).sum::<f64>() / y_train.len() as f64)
        .collect();
    let mean_preds = vec![mean; y_test.len()];

    let linear = fit_ridge_validated(&x_train, &y_train, &RIDGE_LAMBDAS);

    let q_train: Vec<Vec<f64>> = x_train.iter().map(|r| quadratic_features(r)).collect();
    let q_test: Vec<Vec<f64>> = x_test.iter().map(|r| quadratic_features(r)).collect();
    let quadratic = fit_ridge_validated(&q_train, &y_train, &RIDGE_LAMBDAS);

    vec![
        BaselineResult {
            model: "Mean predictor".to_string(),
            n_features: 0,
            lambda: None,
            metrics: regression_metrics(&mean_preds, &y_test, target_names),
        },
        BaselineResult {
            model: "Ridge (linear)".to_string(),
            n_features: d,
            lambda: Some(linear.lambda),
            metrics: regression_metrics(&linear.predict(&x_test), &y_test, target_names),
        },
        BaselineResult {
            model: "Ridge (quadratic)".to_string(),
            n_features: q_train[0].len(),
            lambda: Some(quadratic.lambda),
            metrics: regression_metrics(&quadratic.predict(&q_test), &y_test, target_names),
        },
    ]
}

/// [`evaluate_baselines`] for the ZINC (logP, QED, SAS) task.
pub fn zinc_baselines<'a>(
    train: impl IntoIterator<Item = (&'a [f64], [f64; 3])>,
    test: impl IntoIterator<Item = (&'a [f64], [f64; 3])>,
) -> Vec<BaselineResult> {
    let train: Vec<(Vec<f64>, Vec<f64>)> = train
        .into_iter()
        .map(|(x, y)| (x.to_vec(), y.to_vec()))
        .collect();
    let test: Vec<(Vec<f64>, Vec<f64>)> = test
        .into_iter()
        .map(|(x, y)| (x.to_vec(), y.to_vec()))
        .collect();
    evaluate_baselines(&train, &test, &ZINC_TARGETS)
}

/// QNN result row for [`print_comparison`]; leave a metric `None` when the experiment does not
/// report it.
#[derive(Debug, Clone, Default)]
pub struct QnnScore {
    pub model: String,
    pub mae: Option<[f64; 3]>,
    pub mse: Option<[f64; 3]>,
    pub r2: Option<[f64; 3]>,
    pub pearson_r: Option<[f64; 3]>,
}

impl QnnScore {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Self::default()
        }
    }
    pub fn mae(mut self, v: [f64; 3]) -> Self {
        self.mae = Some(v);
        self
    }
    pub fn mse(mut self, v: [f64; 3]) -> Self {
        self.mse = Some(v);
        self
    }
    pub fn r2(mut self, v: [f64; 3]) -> Self {
        self.r2 = Some(v);
        self
    }
    pub fn pearson_r(mut self, v: [f64; 3]) -> Self {
        self.pearson_r = Some(v);
        self
    }
}

/// Prints QNN rows next to the classical baselines, one block per ZINC target.
pub fn print_comparison(qnn: &[QnnScore], baselines: &[BaselineResult]) {
    print_comparison_named(qnn, baselines, &ZINC_TARGETS);
}

/// As [`print_comparison`] with explicit target names (any three targets).
pub fn print_comparison_named(qnn: &[QnnScore], baselines: &[BaselineResult], names: &[&str]) {
    let fmt = |v: Option<f64>| v.map_or_else(|| "      -".to_string(), |x| format!("{x:>7.4}"));
    println!("\n=========================================================================");
    println!(" CLASSICAL BASELINES vs QNN (same raw features, same train/test split)");
    println!("=========================================================================");
    for (t, name) in names.iter().enumerate() {
        println!("  {name}");
        println!(
            "    {:<40} {:>7} {:>7} {:>7} {:>7}",
            "model", "MAE", "MSE", "R2", "r"
        );
        for q in qnn {
            println!(
                "    {:<40} {} {} {} {}",
                format!("QNN: {}", q.model),
                fmt(q.mae.map(|v| v[t])),
                fmt(q.mse.map(|v| v[t])),
                fmt(q.r2.map(|v| v[t])),
                fmt(q.pearson_r.map(|v| v[t])),
            );
        }
        for b in baselines {
            let m = &b.metrics[t];
            println!(
                "    {:<40} {} {} {} {}",
                format!("Classical: {}", b.model),
                fmt(Some(m.mae)),
                fmt(Some(m.mse)),
                fmt(Some(m.r2)),
                fmt(Some(m.pearson_r)),
            );
        }
    }
}

// =========================================================================
// Capacity-matched nonlinear baselines
// =========================================================================

fn gaussian(rng: &mut rand_chacha::ChaCha8Rng) -> f64 {
    use rand::Rng;
    let u1: f64 = rng.gen_range(1e-12..1.0);
    let u2: f64 = rng.gen_range(0.0..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Fixed random Fourier feature map `z_k(x) = sqrt(2/K) cos(w_k . x + b_k)` with
/// `w_k ~ N(0, I / bandwidth^2)` and `b_k ~ U(0, 2 pi)`: a classical random nonlinear feature
/// map, the counterpart of reading out a random fixed circuit.
#[derive(Debug, Clone, PartialEq)]
pub struct RandomFourierMap {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
}

impl RandomFourierMap {
    pub fn new(n_inputs: usize, n_features: usize, bandwidth: f64, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        assert!(bandwidth > 0.0, "bandwidth must be positive");
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let w = (0..n_features)
            .map(|_| {
                (0..n_inputs)
                    .map(|_| gaussian(&mut rng) / bandwidth)
                    .collect()
            })
            .collect();
        let b = (0..n_features)
            .map(|_| rng.gen_range(0.0..2.0 * std::f64::consts::PI))
            .collect();
        Self { w, b }
    }

    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        let norm = (2.0 / self.w.len() as f64).sqrt();
        self.w
            .iter()
            .zip(&self.b)
            .map(|(wk, bk)| norm * (wk.iter().zip(x).map(|(a, v)| a * v).sum::<f64>() + bk).cos())
            .collect()
    }
}

/// One-hidden-layer tanh MLP. Parameter layout: `[W1 (h x in) | b1 (h) | W2 (out x h) | b2 (out)]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    pub n_in: usize,
    pub n_hidden: usize,
    pub n_out: usize,
    pub params: Vec<f64>,
}

impl Mlp {
    /// Xavier-uniform weights, zero biases.
    pub fn new(n_in: usize, n_hidden: usize, n_out: usize, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let n = n_hidden * n_in + n_hidden + n_out * n_hidden + n_out;
        let mut params = vec![0.0; n];
        let a1 = (6.0 / (n_in + n_hidden) as f64).sqrt();
        let a2 = (6.0 / (n_hidden + n_out) as f64).sqrt();
        for p in params[..n_hidden * n_in].iter_mut() {
            *p = rng.gen_range(-a1..a1);
        }
        let w2 = n_hidden * n_in + n_hidden;
        for p in params[w2..w2 + n_out * n_hidden].iter_mut() {
            *p = rng.gen_range(-a2..a2);
        }
        Self {
            n_in,
            n_hidden,
            n_out,
            params,
        }
    }

    pub fn n_params(&self) -> usize {
        self.params.len()
    }

    fn hidden(&self, x: &[f64]) -> Vec<f64> {
        let (h, i) = (self.n_hidden, self.n_in);
        (0..h)
            .map(|j| {
                let row = &self.params[j * i..(j + 1) * i];
                (row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>() + self.params[h * i + j]).tanh()
            })
            .collect()
    }

    fn output(&self, hidden: &[f64]) -> Vec<f64> {
        let (h, i, o) = (self.n_hidden, self.n_in, self.n_out);
        let w2 = h * i + h;
        (0..o)
            .map(|t| {
                let row = &self.params[w2 + t * h..w2 + (t + 1) * h];
                row.iter().zip(hidden).map(|(w, v)| w * v).sum::<f64>()
                    + self.params[w2 + o * h + t]
            })
            .collect()
    }

    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        self.output(&self.hidden(x))
    }

    /// Mean squared error over the outputs and its gradient.
    pub fn loss_and_gradient(&self, x: &[f64], y: &[f64]) -> (f64, Vec<f64>) {
        let (h, i, o) = (self.n_hidden, self.n_in, self.n_out);
        let hid = self.hidden(x);
        let out = self.output(&hid);
        let d: Vec<f64> = (0..o).map(|t| 2.0 * (out[t] - y[t]) / o as f64).collect();
        let loss = (0..o).map(|t| (out[t] - y[t]).powi(2)).sum::<f64>() / o as f64;
        let mut g = vec![0.0; self.params.len()];
        let w2 = h * i + h;
        for t in 0..o {
            for j in 0..h {
                g[w2 + t * h + j] = d[t] * hid[j];
            }
            g[w2 + o * h + t] = d[t];
        }
        for j in 0..h {
            let dh: f64 = (0..o).map(|t| d[t] * self.params[w2 + t * h + j]).sum();
            let da = dh * (1.0 - hid[j] * hid[j]);
            for k in 0..i {
                g[j * i + k] = da * x[k];
            }
            g[h * i + j] = da;
        }
        (loss, g)
    }
}

/// Training settings for [`train_mlp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MlpTraining {
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub patience: usize,
}

impl Default for MlpTraining {
    fn default() -> Self {
        Self {
            epochs: 300,
            batch_size: 64,
            lr: 0.01,
            patience: 20,
        }
    }
}

/// An MLP trained on standardized targets, predicting in original units.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainedMlp {
    pub net: Mlp,
    pub y_mean: Vec<f64>,
    pub y_scale: Vec<f64>,
    pub best_epoch: usize,
}

impl TrainedMlp {
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        self.net
            .predict(x)
            .iter()
            .zip(&self.y_mean)
            .zip(&self.y_scale)
            .map(|((z, m), s)| z * s + m)
            .collect()
    }
}

/// Trains an MLP with Adam on mini-batches; the epoch with the best validation score (mean over
/// targets of MSE / variance) is kept. Inputs should already be standardized.
pub fn train_mlp(
    x_train: &[Vec<f64>],
    y_train: &[Vec<f64>],
    x_valid: &[Vec<f64>],
    y_valid: &[Vec<f64>],
    n_hidden: usize,
    seed: u64,
    settings: MlpTraining,
) -> TrainedMlp {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let k = y_train[0].len();
    let n = y_train.len() as f64;
    let y_mean: Vec<f64> = (0..k)
        .map(|t| y_train.iter().map(|r| r[t]).sum::<f64>() / n)
        .collect();
    let y_scale: Vec<f64> = (0..k)
        .map(|t| {
            let v = y_train
                .iter()
                .map(|r| (r[t] - y_mean[t]).powi(2))
                .sum::<f64>()
                / n;
            if v > 1e-24 {
                v.sqrt()
            } else {
                1.0
            }
        })
        .collect();
    let std_y =
        |y: &[f64]| -> Vec<f64> { (0..k).map(|t| (y[t] - y_mean[t]) / y_scale[t]).collect() };
    let ys: Vec<Vec<f64>> = y_train.iter().map(|y| std_y(y)).collect();

    let mut net = Mlp::new(x_train[0].len(), n_hidden, k, seed);
    let mut opt = crate::optimizer::AdamOptimizer::new_unwrapped(net.n_params(), settings.lr);
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed.wrapping_add(31));
    let mut order: Vec<usize> = (0..x_train.len()).collect();
    let names: Vec<String> = (0..k).map(|t| t.to_string()).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let valid_score = |net: &Mlp| -> f64 {
        let preds: Vec<Vec<f64>> = x_valid
            .iter()
            .map(|x| {
                net.predict(x)
                    .iter()
                    .zip(&y_mean)
                    .zip(&y_scale)
                    .map(|((z, m), s)| z * s + m)
                    .collect()
            })
            .collect();
        regression_metrics(&preds, y_valid, &refs)
            .iter()
            .map(|m| 1.0 - m.r2)
            .sum::<f64>()
            / k as f64
    };

    let mut best = (valid_score(&net), 0usize, net.params.clone());
    for epoch in 1..=settings.epochs {
        order.shuffle(&mut rng);
        for batch in order.chunks(settings.batch_size.max(1)) {
            let mut grad = vec![0.0; net.n_params()];
            for &i in batch {
                let (_, g) = net.loss_and_gradient(&x_train[i], &ys[i]);
                for (a, b) in grad.iter_mut().zip(&g) {
                    *a += b / batch.len() as f64;
                }
            }
            opt.step(&mut net.params, &grad);
        }
        let score = valid_score(&net);
        if score < best.0 {
            best = (score, epoch, net.params.clone());
        } else if epoch - best.1 >= settings.patience.max(1) {
            break;
        }
    }
    net.params = best.2;
    TrainedMlp {
        net,
        y_mean,
        y_scale,
        best_epoch: best.1,
    }
}

// ----- dense SPD solver -----

/// Lower-triangular Cholesky factor of a row-major `d x d` SPD matrix.
fn cholesky(a: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = a[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * d + i] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Some(l)
}

fn cholesky_solve(l: &[f64], d: usize, b: &[f64]) -> Vec<f64> {
    let mut z = vec![0.0; d];
    for i in 0..d {
        let s: f64 = (0..i).map(|k| l[i * d + k] * z[k]).sum();
        z[i] = (b[i] - s) / l[i * d + i];
    }
    let mut x = vec![0.0; d];
    for i in (0..d).rev() {
        let s: f64 = (i + 1..d).map(|k| l[k * d + i] * x[k]).sum();
        x[i] = (z[i] - s) / l[i * d + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn rows(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut r = ChaCha8Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| r.gen_range(-3.0..3.0)).collect())
            .collect()
    }

    #[test]
    fn ridge_recovers_exact_linear_model() {
        let x = rows(200, 3, 1);
        let y: Vec<Vec<f64>> = x
            .iter()
            .map(|r| vec![3.0 * r[0] - 2.0 * r[1] + 5.0, -r[2] + 0.5])
            .collect();
        let model = RidgeRegression::fit(&x, &y, 1e-10);
        for (xr, yr) in x.iter().zip(&y).take(20) {
            let p = model.predict_one(xr);
            assert!(
                (p[0] - yr[0]).abs() < 1e-6 && (p[1] - yr[1]).abs() < 1e-6,
                "{p:?} vs {yr:?}"
            );
        }
    }

    #[test]
    fn ridge_matches_closed_form_with_penalty() {
        // One standardized feature: w = cov(z, y) / (1 + lambda) since var(z) = 1.
        let x: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let y: Vec<Vec<f64>> = (0..10).map(|i| vec![2.0 * i as f64 + 1.0]).collect();
        let lambda = 0.5;
        let model = RidgeRegression::fit(&x, &y, lambda);
        let sd = (99.0f64 / 12.0).sqrt(); // population sd of 0..9
        let cov_zy = 2.0 * sd; // y = 2x + 1 => cov(z, y) = 2 * sd
        assert!((model.coef[0][0] - cov_zy / (1.0 + lambda)).abs() < 1e-12);
        assert!((model.target_mean[0] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn constant_features_are_harmless() {
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64, 7.0]).collect();
        let y: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 * 0.5]).collect();
        let model = RidgeRegression::fit(&x, &y, 1e-9);
        assert!(model.coef[0][1].abs() < 1e-12);
        assert!((model.predict_one(&[10.0, 7.0])[0] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn quadratic_ridge_fits_interaction_that_linear_cannot() {
        let x = rows(400, 2, 2);
        let train: Vec<(Vec<f64>, Vec<f64>)> = x[..300]
            .iter()
            .map(|r| (r.clone(), vec![r[0] * r[1]; 3]))
            .collect();
        let test: Vec<(Vec<f64>, Vec<f64>)> = x[300..]
            .iter()
            .map(|r| (r.clone(), vec![r[0] * r[1]; 3]))
            .collect();
        let res = evaluate_baselines(&train, &test, &ZINC_TARGETS);
        let (mean, lin, quad) = (&res[0], &res[1], &res[2]);
        assert_eq!(quad.n_features, 2 + 3);
        assert!(mean.metrics[0].r2.abs() < 0.1);
        assert!(lin.metrics[0].r2 < 0.2, "linear r2 {}", lin.metrics[0].r2);
        assert!(
            quad.metrics[0].r2 > 0.999,
            "quadratic r2 {}",
            quad.metrics[0].r2
        );
        assert!(
            quad.lambda.unwrap() <= 1e-3,
            "noiseless data should pick a small penalty"
        );
    }

    #[test]
    fn metrics_match_hand_computed_values() {
        let preds = vec![vec![1.0], vec![2.0], vec![4.0]];
        let targets = vec![vec![1.0], vec![3.0], vec![5.0]];
        let m = &regression_metrics(&preds, &targets, &["t"])[0];
        assert!((m.mae - 2.0 / 3.0).abs() < 1e-12);
        assert!((m.mse - 2.0 / 3.0).abs() < 1e-12);
        // ss_tot = (1-3)^2 + 0 + (5-3)^2 = 8; ss_res = 2 => r2 = 0.75
        assert!((m.r2 - 0.75).abs() < 1e-12);
        // preds centered: -4/3, -1/3, 5/3; targets centered: -2, 0, 2
        let cov = (-4.0 / 3.0) * -2.0 + (5.0 / 3.0) * 2.0;
        let var_p = 16.0 / 9.0 + 1.0 / 9.0 + 25.0 / 9.0;
        assert!((m.pearson_r - cov / (var_p * 8.0f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn quadratic_features_layout() {
        assert_eq!(
            quadratic_features(&[2.0, 3.0]),
            vec![2.0, 3.0, 4.0, 6.0, 9.0]
        );
        assert_eq!(quadratic_features(&[1.0; 19]).len(), 19 + 190);
    }

    #[test]
    fn mlp_gradient_matches_finite_differences() {
        let net = Mlp::new(4, 5, 3, 7);
        assert_eq!(net.n_params(), 4 * 5 + 5 + 3 * 5 + 3);
        let x = [0.3, -1.1, 0.8, 2.0];
        let y = [0.5, -0.4, 1.2];
        let (loss, g) = net.loss_and_gradient(&x, &y);
        let loss_at = |m: &Mlp| {
            m.predict(&x)
                .iter()
                .zip(&y)
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / 3.0
        };
        assert!((loss - loss_at(&net)).abs() < 1e-12);
        for j in 0..net.n_params() {
            let h = 1e-6;
            let mut p = net.clone();
            p.params[j] += h;
            let mut m = net.clone();
            m.params[j] -= h;
            let fd = (loss_at(&p) - loss_at(&m)) / (2.0 * h);
            assert!((g[j] - fd).abs() < 1e-7, "param {j}: {} vs FD {fd}", g[j]);
        }
    }

    #[test]
    fn mlp_learns_an_interaction_linear_ridge_cannot() {
        let x = rows(600, 2, 3);
        let y: Vec<Vec<f64>> = x.iter().map(|r| vec![r[0] * r[1] / 3.0; 3]).collect();
        let (xt, xv) = (&x[..500], &x[500..]);
        let (yt, yv) = (&y[..500], &y[500..]);
        let mlp = train_mlp(
            xt,
            yt,
            xv,
            yv,
            12,
            1,
            MlpTraining {
                epochs: 400,
                lr: 0.02,
                ..MlpTraining::default()
            },
        );
        let preds: Vec<Vec<f64>> = xv.iter().map(|r| mlp.predict(r)).collect();
        let r2 = regression_metrics(&preds, yv, &ZINC_TARGETS)[0].r2;
        let ridge = RidgeRegression::fit(xt, yt, 1e-6);
        let r2_lin = regression_metrics(&ridge.predict(xv), yv, &ZINC_TARGETS)[0].r2;
        assert!(r2 > 0.9, "MLP R2 {r2}");
        assert!(r2_lin < 0.2, "linear R2 {r2_lin}");
    }

    #[test]
    fn random_fourier_map_is_deterministic_and_bounded() {
        let a = RandomFourierMap::new(5, 8, 1.5, 4);
        assert_eq!(a, RandomFourierMap::new(5, 8, 1.5, 4));
        assert_ne!(a, RandomFourierMap::new(5, 8, 1.5, 5));
        let z = a.transform(&[0.1, -0.2, 0.3, 1.0, -2.0]);
        assert_eq!(z.len(), 8);
        let bound = (2.0f64 / 8.0).sqrt();
        assert!(z.iter().all(|v| v.abs() <= bound + 1e-12));
    }
}
