use common::data_reader::{get_feature_names, read_csv_data_multi_target};
use multi_target_decision_tree::{
    classifier::calculate_accuracy,
    decision_tree::{print_tree, MultiTargetDecisionTree},
    split_finder::{SplitFinder, SplitMetric},
};

#[test]
fn test_decision_tree_for_iris() {
    let data_set = read_csv_data_multi_target("./../common/data_files/iris.csv", 3);
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    let tree = MultiTargetDecisionTree::new(data_set, split_finder, 3, false);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_multi_target("./../common/data_files/iris_test.csv", 3);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 3);
    assert_eq!(accuracy, 1.0);
}

#[test]
fn test_decision_tree_for_synthetic() {
    let data_set = read_csv_data_multi_target("./../common/data_files/synthetic_1.csv", 2);
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    let tree = MultiTargetDecisionTree::new(data_set, split_finder, 3, false);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_multi_target("./../common/data_files/synthetic_1.csv", 2);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 2);
    assert_eq!(accuracy, 1.0);
}

#[test]
fn test_decision_tree_for_digits() {
    let data_set = read_csv_data_multi_target("./../common/data_files/digits_train.csv", 10);
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    let tree = MultiTargetDecisionTree::new(data_set, split_finder, 10, false);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_multi_target("./../common/data_files/digits_test.csv", 10);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 10);
    assert!(accuracy > 0.80)
}

#[test]
fn test_decision_tree_for_wine() {
    let train_set = read_csv_data_multi_target("./../common/data_files/wine_train.csv", 3);
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    let tree = MultiTargetDecisionTree::new(train_set, split_finder, 3, false);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data_multi_target("./../common/data_files/wine_test.csv", 3);
    let accuracy = calculate_accuracy(&test_set, &boxed_tree, 3);
    println!("{}", accuracy);
    assert!(accuracy > 0.80)
}

#[test]
fn print_tree_for_wine() {
    let data_set = read_csv_data_multi_target("./../common/data_files/wine_train.csv", 3);
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    let tree = MultiTargetDecisionTree::new(data_set, split_finder, 3, false);
    let feature_names = get_feature_names("./../common/data_files/wine_train.csv");
    print_tree(Box::new(tree.root), "".to_string(), &feature_names);
}

#[test]
fn print_tree_for_synthetic() {
    let data_set = read_csv_data_multi_target("./../common/data_files/synthetic_1.csv", 2);
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    let tree = MultiTargetDecisionTree::new(data_set, split_finder, 2, false);
    let feature_names = get_feature_names("./../common/data_files/synthetic_1.csv");
    print_tree(Box::new(tree.root), "".to_string(), &feature_names);
}
