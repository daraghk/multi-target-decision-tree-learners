use common::data_reader::{
    get_feature_names, read_csv_data_multi_target, read_csv_data_one_hot_multi_target,
};
use multi_target_decision_tree::{
    decision_trees::{
        OneHotMultiTargetDecisionTree, RegressionMultiTargetDecisionTree, TreeConfig,
    },
    printer::print_tree,
    scorer::{calculate_accuracy, calculate_overall_mean_squared_error},
    split_finder::{SplitFinder, SplitMetric},
};
use std::time::Instant;

#[test]
fn test_decision_tree_for_iris() {
    let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 3,
        max_levels: 0,
    };

    let tree = OneHotMultiTargetDecisionTree::new(data_set, tree_config);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris_test.csv", 3);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 3);
    println!("{}", accuracy);
    assert_eq!(accuracy, 1.0);
}

#[test]
fn test_decision_tree_for_synthetic() {
    let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/synthetic_1.csv", 2);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 2,
        max_levels: 0,
    };

    let tree = OneHotMultiTargetDecisionTree::new(data_set, tree_config);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/synthetic_1.csv", 2);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 2);
    println!("{}", accuracy);
    assert_eq!(accuracy, 1.0);
}

#[test]
fn test_decision_tree_for_digits() {
    let data_set =
        read_csv_data_one_hot_multi_target("./../common/data-files/digits_train.csv", 10);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 10,
        max_levels: 0,
    };

    let tree = OneHotMultiTargetDecisionTree::new(data_set, tree_config);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/digits_test.csv", 10);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 10);
    println!("{}", accuracy);
    assert!(accuracy > 0.80)
}

#[test]
fn test_decision_tree_for_wine() {
    let train_set = read_csv_data_one_hot_multi_target("./../common/data-files/wine_train.csv", 3);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 3,
        max_levels: 0,
    };

    let tree = OneHotMultiTargetDecisionTree::new(train_set, tree_config);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/wine_test.csv", 3);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 3);
    println!("{}", accuracy);
    assert!(accuracy > 0.80)
}

#[test]
fn test_decision_tree_for_covtype() {
    let data_set =
        read_csv_data_one_hot_multi_target("./../common/data-files/covtype_train.csv", 7);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 7,
        max_levels: 0,
    };

    let before = Instant::now();
    let tree = OneHotMultiTargetDecisionTree::new(data_set, tree_config);
    println!("Elapsed time: {:.2?}", before.elapsed());
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/covtype_test.csv", 7);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 7);
    println!("{}", accuracy);
    assert!(accuracy > 0.90)
}

#[test]
fn test_decision_tree_for_covtype_multi_threaded() {
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global().unwrap();
    let data_set =
        read_csv_data_one_hot_multi_target("./../common/data-files/covtype_train.csv", 7);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: true,
        number_of_classes: 7,
        max_levels: 0,
    };

    let before = Instant::now();
    let tree = OneHotMultiTargetDecisionTree::new(data_set, tree_config);
    println!("Elapsed time: {:.2?}", before.elapsed());
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/covtype_test.csv", 7);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 7);
    println!("{}", accuracy);
    assert!(accuracy > 0.90)
}

#[test]
fn test_decision_tree_for_regression() {
    let data_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 5,
        max_levels: 0,
    };

    let before = Instant::now();
    let tree = RegressionMultiTargetDecisionTree::new(data_set, tree_config);

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
    let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/wine_train.csv", 3);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 3,
        max_levels: 0,
    };

    let tree = OneHotMultiTargetDecisionTree::new(data_set, tree_config);
    let feature_names = get_feature_names("./../common/data-files/wine_train.csv");
    print_tree(Box::new(tree.root), "".to_string(), &feature_names);
}

#[test]
fn print_tree_for_synthetic() {
    let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/synthetic_1.csv", 2);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 2,
        max_levels: 0,
    };

    let tree = OneHotMultiTargetDecisionTree::new(data_set, tree_config);
    let feature_names = get_feature_names("./../common/data-files/synthetic_1.csv");
    print_tree(Box::new(tree.root), "".to_string(), &feature_names);
}
