use common::{
    data_processor::{self},
    data_reader::{
        get_feature_names, read_csv_data_multi_target, read_csv_data_one_hot_multi_target,
    },
};
use multi_target_decision_tree::{
    decision_trees::{RegressionMultiTargetDecisionTree, TreeConfig},
    printer::print_tree_regression,
    scorer::{
        classification::calculate_accuracy, regression::calculate_overall_mean_squared_error,
    },
    split_finder::{SplitFinder, SplitMetric},
};
use std::time::Instant;

#[test]
fn test_decision_tree_for_iris() {
    let original_data = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
    let data = data_processor::create_dataset_with_sorted_features(&original_data);
    let number_of_classes = original_data.labels[0].len() as u32;
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes,
        max_levels: 8,
    };

    let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris_test.csv", 3);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree);
    println!("{}", accuracy);
    assert_eq!(accuracy, 1.0);
}

#[test]
fn test_decision_tree_for_synthetic() {
    let original_data =
        read_csv_data_one_hot_multi_target("./../common/data-files/synthetic_1.csv", 2);
    let data = data_processor::create_dataset_with_sorted_features(&original_data);
    let number_of_classes = original_data.labels[0].len() as u32;
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes,
        max_levels: 8,
    };

    let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/synthetic_1.csv", 2);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree);
    println!("{}", accuracy);
    assert_eq!(accuracy, 1.0);
}

#[test]
fn test_decision_tree_for_digits() {
    let original_data =
        read_csv_data_one_hot_multi_target("./../common/data-files/digits_train.csv", 10);
    let data = data_processor::create_dataset_with_sorted_features(&original_data);
    let number_of_classes = original_data.labels[0].len() as u32;
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes,
        max_levels: 12,
    };

    let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/digits_test.csv", 10);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree);
    println!("{}", accuracy);
    assert!(accuracy > 0.80)
}

#[test]
fn test_decision_tree_for_wine() {
    let original_data =
        read_csv_data_one_hot_multi_target("./../common/data-files/wine_train.csv", 3);
    let data = data_processor::create_dataset_with_sorted_features(&original_data);
    let number_of_classes = original_data.labels[0].len() as u32;
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes,
        max_levels: 8,
    };

    let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/wine_test.csv", 3);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree);
    println!("{}", accuracy);
    assert!(accuracy > 0.80)
}

#[test]
fn test_decision_tree_for_covtype() {
    let original_data =
        read_csv_data_one_hot_multi_target("./../common/data-files/covtype_train.csv", 7);
    let data = data_processor::create_dataset_with_sorted_features(&original_data);
    let number_of_classes = original_data.labels[0].len() as u32;
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes,
        max_levels: 0,
    };

    let before = Instant::now();
    let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);
    println!("Elapsed time: {:.2?}", before.elapsed());
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/covtype_test.csv", 7);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree);
    println!("{}", accuracy);
    assert!(accuracy > 0.90)
}

#[test]
fn test_decision_tree_for_covtype_multi_threaded() {
    let original_data =
        read_csv_data_one_hot_multi_target("./../common/data-files/covtype_train.csv", 7);
    let data = data_processor::create_dataset_with_sorted_features(&original_data);
    let number_of_classes = original_data.labels[0].len() as u32;
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: true,
        number_of_classes,
        max_levels: 24,
    };

    let before = Instant::now();
    let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);
    println!("Elapsed time: {:.2?}", before.elapsed());
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/covtype_test.csv", 7);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree);
    println!("{}", accuracy);
    assert!(accuracy > 0.90)
}

#[test]
fn test_decision_tree_for_regression_dataset() {
    let original_data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );
    // let data = data_processor::create_dataset_with_binned_sorted_features(&original_data, 1000);
    let data = data_processor::create_dataset_with_sorted_features(&original_data);
    let number_of_classes = original_data.labels[0].len() as u32;
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes,
        max_levels: 10,
    };

    let before = Instant::now();
    let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);

    println!("Elapsed time: {:.2?}", before.elapsed());
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_test_mt.csv",
        "./../common/data-files/multi-target/labels_test_mt.csv",
    );
    let score = calculate_overall_mean_squared_error(&test_set, &boxed_tree);
    let rmse = f64::sqrt(score);
    println!("{}", score);
    println!("{}", rmse);
}

#[test]
fn print_tree_for_wine() {
    let original_data =
        read_csv_data_one_hot_multi_target("./../common/data-files/wine_train.csv", 3);
    let data = data_processor::create_dataset_with_sorted_features(&original_data);
    let number_of_classes = original_data.labels[0].len() as u32;
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes,
        max_levels: 0,
    };

    let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);
    let feature_names = get_feature_names("./../common/data-files/wine_train.csv");
    print_tree_regression(&Box::new(tree.root), "".to_string(), &feature_names);
}

#[test]
fn print_tree_for_synthetic() {
    let original_data =
        read_csv_data_one_hot_multi_target("./../common/data-files/synthetic_1.csv", 2);
    let data = data_processor::create_dataset_with_sorted_features(&original_data);
    let number_of_classes = original_data.labels[0].len() as u32;
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes,
        max_levels: 0,
    };

    let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);
    let feature_names = get_feature_names("./../common/data-files/synthetic_1.csv");
    print_tree_regression(&Box::new(tree.root), "".to_string(), &feature_names);
}
