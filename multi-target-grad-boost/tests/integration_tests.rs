use common::data_reader::{read_csv_data_multi_target, read_csv_data_one_hot_multi_target};
use multi_target_decision_tree::{
    decision_trees::{
        grad_boost_leaf_output::{LeafOutputCalculator, LeafOutputType},
        TreeConfig,
    },
    split_finder::{SplitFinder, SplitMetric},
};
use multi_target_grad_boost::grad_boost_ensemble::{
    ensemble_multi_class::GradientBoostedEnsembleMultiClass,
    ensemble_regression::GradientBoostedEnsembleRegression, GradientBoostedEnsemble,
};

#[test]
fn test_gradient_boosting_single_threaded_tree_building() {
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

    let leaf_output_calculator = LeafOutputCalculator::new(LeafOutputType::Regression);
    let grad_boost_ensemble = GradientBoostedEnsembleRegression::train(
        true_data,
        tree_config,
        leaf_output_calculator,
        100,
        0.1,
    );

    let test_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_test_mt.csv",
        "./../common/data-files/multi-target/labels_test_mt.csv",
    );

    let mean_squared_error = grad_boost_ensemble.calculate_score(&test_set);
    let root_mean_squared_error = f64::sqrt(mean_squared_error);
    println!("{:?}", mean_squared_error);
    println!("{:?}", root_mean_squared_error);

    let prediction = grad_boost_ensemble.predict(&test_set.feature_rows[0]);
    println!("{:?}", test_set.labels[0]);
    println!("{:?}", prediction);
}

#[test]
fn test_gradient_boosting_multi_threaded_tree_building() {
    let true_data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );

    let split_finder = SplitFinder::new(SplitMetric::Variance);
    println!("{:?}", true_data.labels[10]);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: true,
        number_of_classes: 10,
        max_levels: 3,
    };

    let leaf_output_calculator = LeafOutputCalculator::new(LeafOutputType::Regression);
    let grad_boost_ensemble = GradientBoostedEnsembleRegression::train(
        true_data,
        tree_config,
        leaf_output_calculator,
        300,
        0.1,
    );

    let test_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_test_mt.csv",
        "./../common/data-files/multi-target/labels_test_mt.csv",
    );

    let mean_squared_error = grad_boost_ensemble.calculate_score(&test_set);
    let root_mean_squared_error = f64::sqrt(mean_squared_error);
    println!("{:?}", mean_squared_error);
    println!("{:?}", root_mean_squared_error);

    let prediction = grad_boost_ensemble.predict(&test_set.feature_rows[0]);
    println!("{:?}", test_set.labels[0]);
    println!("{:?}", prediction);
}

#[test]
fn test_gradient_boosting_multi_threaded_tree_building_multi_class() {
    let true_data =
        read_csv_data_one_hot_multi_target("./../common/data-files/digits_train.csv", 10);

    let split_finder = SplitFinder::new(SplitMetric::Variance);
    println!("{:?}", true_data.labels[10]);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: true,
        number_of_classes: 10,
        max_levels: 12,
    };

    let leaf_output_calculator =
        LeafOutputCalculator::new(LeafOutputType::MultiClassClassification);
    let grad_boost_ensemble = GradientBoostedEnsembleMultiClass::train(
        true_data,
        tree_config,
        leaf_output_calculator,
        50,
        0.1,
    );

    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/digits_test.csv", 10);
    let prediction = grad_boost_ensemble.predict(&test_set.feature_rows[10]);
    println!("{:?}", test_set.labels[10]);
    println!("{:?}", prediction);

    let accuracy = grad_boost_ensemble.calculate_score(&test_set);
    println!("{:?}", accuracy)
}

#[test]
fn test_gradient_boosting_multi_threaded_tree_building_multi_class_mnist() {
    let true_data =
        read_csv_data_one_hot_multi_target("./../common/data-files/mnist_train.csv", 10);

    let split_finder = SplitFinder::new(SplitMetric::Variance);
    println!("{:?}", true_data.labels[10]);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: true,
        number_of_classes: 10,
        max_levels: 12,
    };

    let leaf_output_calculator =
        LeafOutputCalculator::new(LeafOutputType::MultiClassClassification);
    let grad_boost_ensemble = GradientBoostedEnsembleMultiClass::train(
        true_data,
        tree_config,
        leaf_output_calculator,
        50,
        0.1,
    );

    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/mnist_test.csv", 10);
    let prediction = grad_boost_ensemble.predict(&test_set.feature_rows[10]);
    println!("{:?}", test_set.labels[10]);
    println!("{:?}", prediction);

    let accuracy = grad_boost_ensemble.calculate_score(&test_set);
    println!("{:?}", accuracy)
}
