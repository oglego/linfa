use linfa_nn::distance::Distance;
use linfa_nn::KdTree;
use ndarray::Array1;
use num_traits::Float;

use crate::Weights;

/// K-Nearest Neighbors classifier.
#[derive(Debug, Clone)]
pub struct KNeighborsClassifier<F: Float> {
    // hyperparameters
    pub(crate) k: usize,
    pub(crate) distance: Distance,
    pub(crate) weights: Weights,

    // learned state
    pub(crate) index: Option<KdTree<F>>,
    pub(crate) targets: Option<Array1<usize>>,
}

impl<F: Float> KNeighborsClassifier<F> {
    /// Create a new KNN classifier with `k` neighbors.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            distance: Distance::Euclidean,
            weights: Weights::default(),
            index: None,
            targets: None,
        }
    }

    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    pub fn with_weights(mut self, weights: Weights) -> Self {
        self.weights = weights;
        self
    }
}
