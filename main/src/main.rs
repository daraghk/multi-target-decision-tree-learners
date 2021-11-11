#![allow(unused)]

use std::time::Instant;

use common::data_reader::{read_csv_data_multi_target, read_csv_data_one_hot_multi_target};
use multi_target_decision_tree::{
    scorer::{calculate_accuracy, calculate_overall_mean_squared_error},
    split_finder::{SplitFinder, SplitMetric},
};

fn main() {
    // //multi-target tree with one hot 'classification' data
    // let data_set = read_csv_data_one_hot_multi_target("./common/data-files/iris.csv", 3);
    // let split_finder = SplitFinder::new(SplitMetric::Variance);
    // let tree = MultiTargetDecisionTree::new(data_set, split_finder, 3, false, false);
    // let boxed_tree = Box::new(tree.root);
    // let test_set = read_csv_data_one_hot_multi_target("./common/data-files/iris_test.csv", 3);
    // let accuracy = calculate_accuracy(&test_set, &boxed_tree, 3);
    // println!("{}", accuracy);
    // assert_eq!(accuracy, 1.0);

    // //multi-target tree with regression data
    // let data_set_regression = read_csv_data_multi_target(
    //     "./common/data-files/multi-target/features_train_mt.csv",
    //     "./common/data-files/multi-target/labels_train_mt.csv",
    // );
    // let before = Instant::now();
    // let regression_tree =
    //     MultiTargetDecisionTree::new(data_set_regression, split_finder, 5, true, true);
    // println!("Elapsed time: {:.2?}", before.elapsed());
    // let boxed_regression_tree = Box::new(regression_tree.root);
    // let test_set_regression = read_csv_data_multi_target(
    //     "./common/data-files/multi-target/features_test_mt.csv",
    //     "./common/data-files/multi-target/labels_test_mt.csv",
    // );
    // let score = calculate_overall_mean_squared_error(&test_set_regression, &boxed_regression_tree);
    // let rmse = f32::sqrt(score);
    // println!("{}", score);
    // println!("{}", rmse);
}
