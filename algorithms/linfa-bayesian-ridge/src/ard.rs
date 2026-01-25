/// Automatic Relevance Determination Regression

use ndarray::{Array1, Array2};
pub struct ARDRegression;

impl ARDRegression {
    pub fn default() -> Self {
        Self
    }

    pub fn fit(&self, _x: &Array2<f64>, _y: &Array1<f64>) -> Result<(), crate::error::BayesianError> {
        unimplemented!()
    }

    pub fn predict(&self, _x: &Array2<f64>) -> Array1<f64> {
        unimplemented!()
    }
}
