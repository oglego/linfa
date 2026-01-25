use linfa::Error as LinfaError;

#[derive(Debug)]
pub enum BayesianError {
    ConvergenceError,
    DimensionMismatch,
    InvalidInput(String),
    Linfa(LinfaError),
}

impl From<LinfaError> for BayesianError {
    fn from(err: LinfaError) -> Self {
        BayesianError::Linfa(err)
    }
}

impl std::fmt::Display for BayesianError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BayesianError::ConvergenceError => write!(f, "Algorithm failed to converge"),
            BayesianError::DimensionMismatch => write!(f, "Input dimensions do not match"),
            BayesianError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            BayesianError::Linfa(err) => write!(f, "Linfa error: {}", err),
        }
    }
}

impl std::error::Error for BayesianError {}
