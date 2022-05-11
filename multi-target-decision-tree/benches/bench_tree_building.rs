use std::time::Duration;

use common::{
    data_processor::create_dataset_with_sorted_features, data_reader::read_csv_data_multi_target,
};
use criterion::{criterion_group, criterion_main, Criterion};
use multi_target_decision_tree::{
    decision_trees::{RegressionMultiTargetDecisionTree, TreeConfig},
    split_finder::{SplitFinder, SplitMetric},
};

fn benchmark_build_tree(c: &mut Criterion) {
    let original_data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );
    let data = create_dataset_with_sorted_features(&original_data);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: true,
        number_of_classes: 10,
        max_levels: 8,
    };

    c.bench_function("multi target tree build", |b| {
        b.iter(|| RegressionMultiTargetDecisionTree::new(data.clone(), tree_config))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(30));
    targets = benchmark_build_tree
);
criterion_main!(benches);
