use linfa::Error as LinfaError;
use linfa::prelude::*;
use linfa_nn::KdTree;
use ndarray::Array2;
use num_traits::Float;

use super::model::KNeighborsClassifier;
use crate::params::validate_k;

impl<F: Float> Fit<Array2<F>, ndarray::Array1<usize>, LinfaError>
    for KNeighborsClassifier<F>
{
    type Object = KNeighborsClassifier<F>;

    fn fit(
        &self,
        dataset: &DatasetBase<Array2<F>, ndarray::Array1<usize>>,
    ) -> Result<Self::Object, LinfaError> {
        validate_k(self.k, dataset.nsamples())?;

        let index = KdTree::new(dataset.records().view(), self.distance);

        Ok(KNeighborsClassifier {
            k: self.k,
            distance: self.distance,
            weights: self.weights,
            index: Some(index),
            targets: Some(dataset.targets().to_owned()),
        })
    }
}
