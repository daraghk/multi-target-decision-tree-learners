use common::data_reader::read_csv_data;
use decision_tree::{
    classifier::calculate_accuracy,
    decision_tree::DecisionTree,
    split_finder::{SplitFinder, SplitMetric},
};

#[test]
fn test_decision_tree_for_iris() {
    let data_set = read_csv_data("./../common/data_files/iris.csv");
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    let tree = DecisionTree::new(data_set, split_finder, 3, false);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data("./../common/data_files/iris_test.csv");
    let accuracy = calculate_accuracy(&test_set, &boxed_tree);
    assert_eq!(accuracy, 1.0);
}

#[test]
fn test_decision_tree_for_wine() {
    let data_set = read_csv_data("./../common/data_files/wine_train.csv");
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    let tree = DecisionTree::new(data_set, split_finder, 3, false);
    let boxed_tree = Box::new(tree.root);
    let test_set = read_csv_data("./../common/data_files/wine_test.csv");
    let accuracy = calculate_accuracy(&test_set, &boxed_tree);
    assert!(accuracy > 0.90)
}
