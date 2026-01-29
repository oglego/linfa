/// Weighting strategy for neighbor contributions.
#[derive(Clone, Copy, Debug)]
pub enum Weights {
    /// All neighbors contribute equally.
    Uniform,
    /// Weight neighbors by inverse distance.
    Distance,
}

impl Default for Weights {
    fn default() -> Self {
        Weights::Uniform
    }
}
