use ndarray::{Array1, Array2};

pub struct BayesianPosterior {
    pub mean: Array1<f64>,
    pub covariance: Array2<f64>,
}

impl BayesianPosterior {
    pub fn new(mean: Array1<f64>, covariance: Array2<f64>) -> Self {
        Self { mean, covariance }
    }
}
