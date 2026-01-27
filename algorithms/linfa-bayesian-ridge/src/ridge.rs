use linfa::traits::Fit;
use linfa::DatasetBase;
use ndarray::{Array1, Array2, Axis};

use crate::ard::{fit_ard, ARDParams, ARDResult};
use crate::posterior::BayesianPosterior;
use crate::error::BayesianError;

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

pub struct BayesianRidgeModel {
    pub posterior: BayesianPosterior,
    pub intercept: f64,
    pub alpha: f64,
    pub lambda: f64,
}

impl BayesianRidgeModel {
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        x.dot(&self.posterior.mean) + self.intercept
    }
}

impl Fit<Array2<f64>, Array1<f64>, BayesianError> for BayesianRidge {
    type Object = BayesianRidgeModel;

    fn fit(
        &self,
        dataset: &DatasetBase<Array2<f64>, Array1<f64>>,
    ) -> Result<Self::Object, BayesianError> {
        let mut x = dataset.records().to_owned();
        let mut y = dataset.targets().to_owned();

        // ---- center ----
        let (x_mean, y_mean) = if self.fit_intercept {
            let xm = x.mean_axis(Axis(0)).unwrap();
            let ym = y.mean().unwrap();
            x -= &xm;
            y -= ym;
            (Some(xm), Some(ym))
        } else {
            (None, None)
        };

        let params = ARDParams {
            max_iter: self.max_iter,
            tol: self.tol,
            alpha_init: self.alpha_init.unwrap_or(1.0),
            lambda_init: self.lambda_init.unwrap_or(1.0),
        };

        let ard = fit_ard(&x, &y, &params)?;

        // ---- recover intercept ----
        let intercept = if let (Some(xm), Some(ym)) = (x_mean, y_mean) {
            ym - xm.dot(&ard.posterior.mean)
        } else {
            0.0
        };

        Ok(BayesianRidgeModel {
            posterior: ard.posterior,
            intercept,
            alpha: ard.alpha,
            lambda: ard.lambdas.mean().unwrap(),
        })
    }
}
