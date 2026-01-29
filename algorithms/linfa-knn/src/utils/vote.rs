use ndarray::Array1;
use std::collections::HashMap;

/// Majority vote over nearest-neighbour indices
pub fn majority_vote(
    neighbors: &[(ndarray::ArrayView1<'_, f64>, usize)],
    targets: &Array1<usize>,
) -> usize {
    let mut counts = HashMap::<usize, usize>::new();

    for (_, idx) in neighbors {
        let label = targets[*idx];
        *counts.entry(label).or_insert(0) += 1;
    }

    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .expect("at least one neighbor")
        .0
}
