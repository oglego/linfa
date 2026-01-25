use linfa::prelude::*;
use linfa::Dataset;
use linfa_bayesian_ridge::BayesianRidge;
use ndarray::{array, Array2, Array1};

fn main() {
    let x: Array2<f64> = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ];
    let y: Array1<f64> = array![1.0, 2.0, 3.0];

    let dataset = Dataset::new(x, y);

    let model = BayesianRidge::default()
        .fit(&dataset)
        .expect("fit failed");

    println!("Intercept: {}", model.intercept);
    println!("Coefficients: {:?}", model.posterior.mean);
}
