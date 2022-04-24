use std::time::Duration;

use common::{
    data_processor::create_dataset_with_sorted_features,
    data_reader::{read_csv_data_multi_target},
};
use criterion::{criterion_group, criterion_main, Criterion};
use multi_target_decision_tree::{
    decision_trees::{
        RegressionMultiTargetDecisionTree, RegressionMultiTargetDecisionTreeNewPartition,
        TreeConfig,
    },
    split_finder::{SplitFinder, SplitMetric},
};

fn benchmark_build_tree_old_partition(c: &mut Criterion) {
    let data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 10,
        max_levels: 8,
    };

    c.bench_function("multi target tree build - old partition", |b| {
        b.iter(|| {
            return RegressionMultiTargetDecisionTree::new(data.clone(), tree_config);
        })
    });
}

fn benchmark_build_tree_new_partition(c: &mut Criterion) {
    let data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );

    let processed_dataset = create_dataset_with_sorted_features(&data);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 10,
        max_levels: 8,
    };

    c.bench_function("multi target tree build - new partition", |b| {
        b.iter(|| {
            return RegressionMultiTargetDecisionTreeNewPartition::new(
                processed_dataset.clone(),
                tree_config,
            );
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(30));
    targets = benchmark_build_tree_old_partition, benchmark_build_tree_new_partition
);
criterion_main!(benches);
