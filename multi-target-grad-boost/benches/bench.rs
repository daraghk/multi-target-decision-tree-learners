use common::data_reader::read_csv_data_multi_target;
use criterion::{criterion_group, criterion_main, Criterion};
use multi_target_decision_tree::{
    decision_trees::{
        grad_boost_leaf_output::{LeafOutputCalculator, LeafOutputType},
        TreeConfig,
    },
    split_finder::{SplitFinder, SplitMetric},
};
use multi_target_grad_boost::boosting_ensemble::{
    boosting_types::RegressionBoostModel, GradientBoostedEnsemble,
};

fn perform_gradient_boosting_single_threaded(c: &mut Criterion) {
    let data_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 10,
        max_levels: 3,
    };

    let leaf_output_calculator = LeafOutputCalculator::new(LeafOutputType::Regression);
    c.bench_function("multi target grad boost tree build - single thread", |b| {
        b.iter(|| {
            return RegressionBoostModel::train(
                data_set.clone(),
                tree_config,
                leaf_output_calculator,
                300,
                0.1,
            );
        })
    });
}

fn perform_gradient_boosting_multi_threaded(c: &mut Criterion) {
    let data_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: true,
        number_of_classes: 10,
        max_levels: 3,
    };

    let leaf_output_calculator = LeafOutputCalculator::new(LeafOutputType::Regression);
    c.bench_function("multi target grad boost tree build - multi threaded", |b| {
        b.iter(|| {
            return RegressionBoostModel::train(
                data_set.clone(),
                tree_config,
                leaf_output_calculator,
                300,
                0.1,
            );
        })
    });
}

criterion_group!(
    benches,
    perform_gradient_boosting_single_threaded,
    perform_gradient_boosting_multi_threaded
);
criterion_main!(benches);
