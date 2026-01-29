use linfa::prelude::*;
use ndarray::{Array1, Array2};
use num_traits::Float;

use super::model::KNeighborsClassifier;
use crate::utils::vote::{majority_vote};

impl<F: Float> Predict<Array2<F>, Array1<usize>>
    for KNeighborsClassifier<F>
{
    fn predict(&self, records: &Array2<F>) -> Array1<usize> {
        let index = self.index.as_ref().expect("model not fitted");
        let targets = self.targets.as_ref().expect("model not fitted");

        records
            .outer_iter()
            .map(|row| {
                let neighbors = index.nearest(row.view(), self.k);

                match self.weights {
                    crate::Weights::Uniform => {
                        majority_vote(&neighbors, targets)
                    }
                    crate::Weights::Distance => {
                        majority_vote(&neighbors, targets)
                    }
                }
            })
            .collect()
    }
}
