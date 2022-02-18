use std::time::Instant;

use common::data_reader::read_csv_data_one_hot_multi_target;
use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    split_finder::{SplitFinder, SplitMetric},
};
use multi_target_grad_boost::boosting_ensemble::{
    boosting_types::AMGBoostModel, GradientBoostedEnsemble,
};

fn main() {
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

    let before = Instant::now();
    let grad_boost_ensemble = AMGBoostModel::train(true_data, tree_config, 50, 0.1);
    println!("Elapsed time: {:.2?}", before.elapsed());

    let test_set = read_csv_data_one_hot_multi_target("./../common/data-files/digits_test.csv", 10);
    let prediction = grad_boost_ensemble.predict(&test_set.feature_rows[10]);
    println!("{:?}", test_set.labels[10]);
    println!("{:?}", prediction);

    let accuracy = grad_boost_ensemble.calculate_score(&test_set);
    println!("{:?}", accuracy)
}
