//! linfa-bayesian-ridge
//!
//! Bayesian Ridge Regression for Linfa
//! Provides uncertainty-aware linear regression with posterior predictive distributions.

pub mod ridge;
pub mod ard;
pub mod posterior;
pub mod error;

pub use ridge::BayesianRidge;
pub use posterior::BayesianPosterior;
pub use error::BayesianError;

