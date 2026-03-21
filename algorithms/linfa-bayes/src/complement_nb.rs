use linfa::dataset::{AsSingleTargets, DatasetBase, Labels};
use linfa::traits::{Fit, FitWith, PredictInplace};
use linfa::{Float, Label};
use ndarray::{Array1, ArrayBase, ArrayView2, Axis, Data, Ix2};
use std::collections::HashMap;
use std::hash::Hash;

use crate::base_nb::{NaiveBayes, NaiveBayesValidParams};
use crate::error::{NaiveBayesError, Result};
use crate::hyperparams::{ComplementNbParams, ComplementNbValidParams};
use crate::{filter, ClassHistogram};

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

impl<'a, F, L, D, T> NaiveBayesValidParams<'a, F, L, D, T> for ComplementNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
}

impl<F, L, D, T> Fit<ArrayBase<D, Ix2>, T, NaiveBayesError> for ComplementNbValidParams<F, L>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = ComplementNb<F, L>;
    // Thin wrapper around the corresponding method of NaiveBayesValidParams
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        NaiveBayesValidParams::fit(self, dataset, None)
    }
}

impl<'a, F, L, D, T> FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError>
    for ComplementNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    type ObjectIn = Option<ComplementNb<F, L>>;
    type ObjectOut = ComplementNb<F, L>;

    fn fit_with(
        &self,
        model_in: Self::ObjectIn,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::ObjectOut> {
        let x = dataset.records();
        let y = dataset.as_single_targets();

        let mut model = match model_in {
            Some(temp) => temp,
            None => ComplementNb {
                class_info: HashMap::new(),
            },
        };

        let yunique = dataset.labels();
        let alpha_val = self.alpha();

        for class in yunique {
            // filter dataset for current class
            let xclass = filter(x.view(), y.view(), &class);
            let info = model
                .class_info
                .entry(class.clone())
                .or_insert_with(ClassHistogram::default);

            let class_feature_sum = xclass.sum_axis(Axis(0));

            if info.class_count > 0 {
                info.feature_count = &info.feature_count + &class_feature_sum;
            } else {
                info.feature_count = class_feature_sum;
            }

            info.class_count += xclass.nrows();
        }

        let class_count_sum = model
            .class_info
            .values()
            .map(|info| info.class_count)
            .sum::<usize>();

        if class_count_sum == 0 {
            return Ok(model);
        }

        let total_feature_counts = model
            .class_info
            .values()
            .fold(Array1::zeros(x.ncols()), |acc, info| {
                acc + &info.feature_count
            });

        for info in model.class_info.values_mut() {
            info.prior = F::cast(info.class_count) / F::cast(class_count_sum);

            // Recompute complement weights from the accumulated feature counts.
            let mut complement_counts = &total_feature_counts - &info.feature_count;
            complement_counts.mapv_inplace(|c| c + alpha_val);

            let smoothed_counts: F = complement_counts.sum();
            let mut weights: Array1<F> = complement_counts.mapv(|c: F| (c / smoothed_counts).ln());

            let l1_norm = weights.fold(F::zero(), |acc: F, w: &F| acc + w.abs());
            if l1_norm > F::cast(1e-10) {
                weights.mapv_inplace(|w| w / l1_norm);
            }

            info.feature_log_prob = weights;
        }

        Ok(model)
    }
}

impl<F: Float, L: Label, D> PredictInplace<ArrayBase<D, Ix2>, Array1<L>> for ComplementNb<F, L>
where
    D: Data<Elem = F>,
{
    // Thin wrapper around the corresponding method of NaiveBayes
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {
        NaiveBayes::predict_inplace(self, x, y);
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}

/// Fitted Complement Naive Bayes classifier.
///
/// See [ComplementNbParams] for more information on the hyper-parameters.
///
/// # Model assumptions
///
/// The family of Naive Bayes classifiers assume independence between variables. They do not model
/// moments between variables and lack therefore in modelling capability. The advantage is a linear
/// fitting time with maximum-likelihood training in a closed form.
///
/// # Model usage example
///
/// The example below creates a set of hyperparameters, and then uses it to fit a Complement Naive
/// Bayes classifier on provided data.
///
/// ```rust
/// use linfa_bayes::{ComplementNbParams, ComplementNbValidParams, Result};
/// use linfa::prelude::*;
/// use ndarray::array;
///
/// let x = array![
///     [-2., -1.],
///     [-1., -1.],
///     [-1., -2.],
///     [1., 1.],
///     [1., 2.],
///     [2., 1.]
/// ];
/// let y = array![1, 1, 1, 2, 2, 2];
/// let ds = DatasetView::new(x.view(), y.view());
///
/// // create a new parameter set with smoothing parameter equals `1`
/// let unchecked_params = ComplementNbParams::new()
///     .alpha(1.0);
///
/// // fit model with unchecked parameter set
/// let model = unchecked_params.fit(&ds)?;
///
/// // transform into a verified parameter set
/// let checked_params = unchecked_params.check()?;
///
/// // update model with the verified parameters, this only returns
/// // errors originating from the fitting process
/// let model = checked_params.fit_with(Some(model), &ds)?;
/// # Result::Ok(())
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct ComplementNb<F: PartialEq, L: Eq + Hash> {
    class_info: HashMap<L, ClassHistogram<F>>,
}

impl<F: Float, L: Label> ComplementNb<F, L> {
    /// Construct a new set of hyperparameters
    pub fn params() -> ComplementNbParams<F, L> {
        ComplementNbParams::new()
    }
}

impl<F, L> NaiveBayes<'_, F, L> for ComplementNb<F, L>
where
    F: Float,
    L: Label + Ord,
{
    // Compute unnormalized posterior log probability
    fn joint_log_likelihood(&self, x: ArrayView2<F>) -> HashMap<&L, Array1<F>> {
        let mut joint_log_likelihood = HashMap::new();
        for (class, info) in self.class_info.iter() {
            // Negate the complement score so the shared argmax-based prediction logic
            // still prefers the most likely class.
            let nij = x.dot(&info.feature_log_prob);
            joint_log_likelihood.insert(class, -nij);
        }
        joint_log_likelihood
    }
}

#[cfg(test)]
mod tests {
    use super::{ComplementNb, NaiveBayes, Result};
    use linfa::{
        traits::{Fit, FitWith, Predict},
        Dataset, DatasetView, Error,
    };

    use crate::{ComplementNbParams, ComplementNbValidParams};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Axis};
    use std::collections::HashMap;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<ComplementNb<f64, usize>>();
        has_autotraits::<ComplementNbValidParams<f64, usize>>();
        has_autotraits::<ComplementNbParams<f64, usize>>();
    }

    #[test]
    fn test_complement_nb() -> Result<()> {
        let ds = Dataset::new(
            array![[1., 0.], [2., 0.], [3., 0.], [0., 1.], [0., 2.], [0., 3.]],
            array![1, 1, 1, 2, 2, 2],
        );

        let fitted_clf = ComplementNb::params().fit(&ds)?;
        let pred = fitted_clf.predict(ds.records());

        assert_abs_diff_eq!(pred, ds.targets());

        let jll = fitted_clf.joint_log_likelihood(ds.records().view());
        let mut expected = HashMap::new();

        // Values calculated from the crate's Complement NB scoring convention.
        expected.insert(
            &1usize,
            array![0.93965973, 1.87931945, 2.81897918, 0.06034027, 0.12068055, 0.18102082],
        );

        expected.insert(
            &2usize,
            array![0.06034027, 0.12068055, 0.18102082, 0.93965973, 1.87931945, 2.81897918],
        );

        for (key, value) in jll.iter() {
            assert_abs_diff_eq!(value, expected.get(key).unwrap(), epsilon = 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_cnb_fit_with_matches_fit() -> Result<()> {
        let x = array![[1., 0.], [2., 0.], [3., 0.], [0., 1.], [0., 2.], [0., 3.]];
        let y = array![1, 1, 1, 2, 2, 2];
        let ds = DatasetView::new(x.view(), y.view());

        let clf = ComplementNb::params();
        let single_pass_model = clf.fit(&ds)?;

        let chunked_model = x
            .axis_chunks_iter(Axis(0), 2)
            .zip(y.axis_chunks_iter(Axis(0), 2))
            .map(|(a, b)| DatasetView::new(a, b))
            .try_fold(None, |current, d| clf.fit_with(current, &d).map(Some))?
            .ok_or(Error::NotEnoughSamples)?;

        let pred = chunked_model.predict(&x);

        assert_abs_diff_eq!(pred, y);

        let single_pass_jll = single_pass_model.joint_log_likelihood(x.view());
        let chunked_jll = chunked_model.joint_log_likelihood(x.view());

        for (key, value) in single_pass_jll.iter() {
            assert_abs_diff_eq!(value, chunked_jll.get(key).unwrap(), epsilon = 1e-6);
        }

        Ok(())
    }
}
