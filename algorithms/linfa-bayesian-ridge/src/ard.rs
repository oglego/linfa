use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, Norm};
use crate::posterior::BayesianPosterior;
use crate::error::BayesianError;

#[derive(Debug, Clone)]
pub struct ARDParams {
    pub max_iter: usize,
    pub tol: f64,
    pub alpha_init: f64,
    pub lambda_init: f64,
}

pub struct ARDResult {
    pub posterior: BayesianPosterior,
    pub alpha: f64,
    pub lambdas: Array1<f64>, // per-feature precision
}

pub fn fit_ard(
    x: &Array2<f64>,
    y: &Array1<f64>,
    params: &ARDParams,
) -> Result<ARDResult, BayesianError> {
    let n_samples = x.nrows();
    let n_features = x.ncols();

    let mut alpha = params.alpha_init;
    let mut lambdas = Array1::from_elem(n_features, params.lambda_init);

    let mut mean = Array1::<f64>::zeros(n_features);
    let mut covariance = Array2::<f64>::eye(n_features); 

    for _ in 0..params.max_iter {
        let lambda_mat = Array2::from_diag(&lambdas);
        let xtx = x.t().dot(x);

        let precision = &lambda_mat + &(xtx * alpha);

        covariance = precision
            .inv()
            .map_err(|_| BayesianError::ConvergenceError)?;

        let new_mean = covariance.dot(&x.t().dot(y)) * alpha;

        // ---- convergence ----
        let delta = (&new_mean - &mean).norm_l2();
        mean = new_mean;
        if delta < params.tol {
            break;
        }

        // ---- update lambdas ----
        let mut gamma_sum = 0.0;
        for j in 0..n_features {
            let gamma_j = 1.0 - lambdas[j] * covariance[(j, j)];
            lambdas[j] = gamma_j / mean[j].powi(2).max(1e-12);
            gamma_sum += gamma_j;
        }

        // ---- update alpha ----
        let residual = y - &x.dot(&mean);
        alpha = (n_samples as f64 - gamma_sum)
            / residual.norm_l2().powi(2).max(1e-12);
    }

    Ok(ARDResult {
        posterior: BayesianPosterior::new(mean, covariance),
        alpha,
        lambdas,
    })
}
