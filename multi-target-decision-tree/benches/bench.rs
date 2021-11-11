use common::data_reader::read_csv_data_one_hot_multi_target;
use criterion::{criterion_group, criterion_main, Criterion};
use multi_target_decision_tree::{
    decision_tree::{OneHotMultiTargetDecisionTree, TreeConfig},
    leaf::OneHotMultiClassLeaf,
    split_finder::{SplitFinder, SplitMetric},
};

fn benchmark_build_tree_single_threaded(c: &mut Criterion) {
    let data_set =
        read_csv_data_one_hot_multi_target("./../common/data-files/covtype_train.csv", 7);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        is_regression_tree: false,
        number_of_classes: 7,
        max_levels: 0,
        is_grad_boost_tree: false,
    };

    c.bench_function("multi target tree build - single thread", |b| {
        b.iter(|| {
            return OneHotMultiTargetDecisionTree::new(data_set.clone(), tree_config);
        })
    });
}

fn benchmark_build_tree_multi_threaded(c: &mut Criterion) {
    let data_set =
        read_csv_data_one_hot_multi_target("./../common/data-files/covtype_train.csv", 7);
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: true,
        is_regression_tree: false,
        number_of_classes: 7,
        max_levels: 0,
        is_grad_boost_tree: false,
    };

    c.bench_function("multi target tree build - multi thread", |b| {
        b.iter(|| {
            return OneHotMultiTargetDecisionTree::new(data_set.clone(), tree_config);
        })
    });
}

criterion_group!(
    benches,
    benchmark_build_tree_single_threaded,
    benchmark_build_tree_multi_threaded
);
criterion_main!(benches);
