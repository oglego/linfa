use linfa::Error as LinfaError;

pub fn validate_k(k: usize, n_samples: usize) -> Result<(), LinfaError> {
    if k == 0 {
        return Err(LinfaError::Parameters("k must be > 0".into()));
    }
    if k > n_samples {
        return Err(LinfaError::Parameters(
            "k must be <= number of samples".into(),
        ));
    }
    Ok(())
}
