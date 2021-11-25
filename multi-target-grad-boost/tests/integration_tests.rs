use common::data_reader::read_csv_data_multi_target;
use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    split_finder::{SplitFinder, SplitMetric},
};
use multi_target_grad_boost::grad_boost_ensemble;

#[test]
fn test_gradient_boosting() {
    let true_data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );

    let split_finder = SplitFinder::new(SplitMetric::Variance);
    println!("{:?}", true_data.labels[10]);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 5,
        max_levels: 8,
    };

    let grad_boost_ensemble =
        grad_boost_ensemble::GradientBoostedEnsemble::train(true_data, tree_config, 100, 0.1);

    let test_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_test_mt.csv",
        "./../common/data-files/multi-target/labels_test_mt.csv",
    );

    let mean_squared_error = grad_boost_ensemble.calculate_error(&test_set);
    let root_mean_squared_error = f64::sqrt(mean_squared_error);
    println!("{:?}", mean_squared_error);
    println!("{:?}", root_mean_squared_error);

    let prediction = grad_boost_ensemble.predict(&test_set.features[99]);
    println!("{:?}", test_set.labels[99]);
    println!("{:?}", prediction);
}
