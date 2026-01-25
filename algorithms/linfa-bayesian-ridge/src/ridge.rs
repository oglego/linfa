use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa::Float;

use ndarray::{Array1, Array2, Axis};

use crate::error::BayesianError;
use crate::posterior::BayesianPosterior;

/// Bayesian Ridge Regression
///
/// This implementation follows scikit-learn's BayesianRidge:
/// - Features and targets are centered if `fit_intercept = true`
/// - The intercept is recovered after fitting
pub struct BayesianRidge {
    pub max_iter: usize,
    pub tol: f64,
    pub alpha_init: Option<f64>,
    pub lambda_init: Option<f64>,
    pub fit_intercept: bool,
}

impl Default for BayesianRidge {
    fn default() -> Self {
        Self {
            max_iter: 300,
            tol: 1e-4,
            alpha_init: None,
            lambda_init: None,
            fit_intercept: true,
        }
    }
}

impl BayesianRidge {
    pub fn max_iter(mut self, value: usize) -> Self {
        self.max_iter = value;
        self
    }

    pub fn tol(mut self, value: f64) -> Self {
        self.tol = value;
        self
    }

    pub fn alpha_init(mut self, value: f64) -> Self {
        self.alpha_init = Some(value);
        self
    }

    pub fn lambda_init(mut self, value: f64) -> Self {
        self.lambda_init = Some(value);
        self
    }

    pub fn fit_intercept(mut self, value: bool) -> Self {
        self.fit_intercept = value;
        self
    }
}

/// Fitted Bayesian Ridge model
pub struct BayesianRidgeModel {
    pub posterior: BayesianPosterior,
    pub intercept: f64,
    pub alpha: f64,
    pub lambda: f64,
}

impl BayesianRidgeModel {
    /// Predict posterior mean
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.posterior.mean) + self.intercept
    }

    /// Predict posterior mean and variance
    pub fn predict_dist(&self, x: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        let mean = self.predict(x);

        // var(y*) = 1/alpha + x*^T Σ x*
        // (stubbed for now)
        let var = Array1::zeros(x.nrows());

        (mean, var)
    }

    pub fn posterior(&self) -> &BayesianPosterior {
        &self.posterior
    }
}

impl Fit<Array2<f64>, Array1<f64>, BayesianError> for BayesianRidge {
    type Object = BayesianRidgeModel;

    // Change &Dataset<f64, f64> to &DatasetBase<Array2<f64>, Array1<f64>>
    fn fit(&self, dataset: &DatasetBase<Array2<f64>, Array1<f64>>) -> Result<Self::Object, BayesianError> {
        let x = dataset.records();
        let y = dataset.targets();

        if x.nrows() != y.len() {
            return Err(BayesianError::DimensionMismatch);
        }

        // ---- Clone since we mutate ----
        let mut x = x.to_owned();
        let mut y = y.to_owned();

        // ---- Center X and y if intercept is enabled ----
        let (x_mean, y_mean) = if self.fit_intercept {
            let xm = x.mean_axis(Axis(0)).unwrap();
            let ym = y.mean().unwrap();

            for mut row in x.rows_mut() {
                row -= &xm;
            }
            y -= ym;

            (Some(xm), Some(ym))
        } else {
            (None, None)
        };

        let n_features = x.ncols();

        let alpha = self.alpha_init.unwrap_or(1.0);
        let lambda = self.lambda_init.unwrap_or(1.0);

        // ---- Stub posterior ----
        let coef = Array1::<f64>::zeros(n_features);
        let covariance = Array2::<f64>::eye(n_features);

        let intercept = if let (Some(xm), Some(ym)) = (x_mean, y_mean) {
            ym - xm.dot(&coef)
        } else {
            0.0
        };

        Ok(BayesianRidgeModel {
            posterior: BayesianPosterior::new(coef, covariance),
            intercept,
            alpha,
            lambda,
        })
    }
}

