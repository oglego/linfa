use linfa::Dataset;
use linfa_knn::classifier::KNeighborsClassifier;
use ndarray::array;

#[test]
fn knn_classifier_predicts_training_data() {
    let dataset = Dataset::new(
        array![[0., 0.], [1., 1.], [0., 1.], [1., 0.]],
        array![0, 1, 0, 1],
    );

    let model = KNeighborsClassifier::new(1)
        .fit(&dataset)
        .unwrap();

    let preds = model.predict(dataset.records());

    println!("Predictions: {:?}", preds);
    assert_eq!(preds, dataset.targets());
}
