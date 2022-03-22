use std::time::Instant;

use common::data_reader::read_csv_data_multi_target;
use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    split_finder::{SplitFinder, SplitMetric},
};
use multi_target_grad_boost::boosting_ensemble::{
    boosting_types::RegressionBoostModel, GradientBoostedEnsemble,
};

fn main() {
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

    let before = Instant::now();
    let grad_boost_ensemble = RegressionBoostModel::train(true_data, tree_config, 300, 0.1);
    println!("Elapsed time: {:.2?}", before.elapsed());

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
