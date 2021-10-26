use common::data_reader::read_csv_data_multi_target;
use criterion::{criterion_group, criterion_main, Criterion};
use multi_target_decision_tree::{
    decision_tree::MultiTargetDecisionTree,
    split_finder::{SplitFinder, SplitMetric},
};

fn benchmark_build_tree_single_threaded(c: &mut Criterion) {
    let data_set = read_csv_data_multi_target("./../common/data_files/digits_train.csv", 10);
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    c.bench_function("multi target tree build - single thread", |b| {
        b.iter(|| {
            return MultiTargetDecisionTree::new(data_set.clone(), split_finder, 10, false);
        })
    });
}

fn benchmark_build_tree_multi_threaded(c: &mut Criterion) {
    let data_set = read_csv_data_multi_target("./../common/data_files/digits_train.csv", 10);
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    c.bench_function("multi target tree build - multi thread", |b| {
        b.iter(|| {
            return MultiTargetDecisionTree::new(data_set.clone(), split_finder, 10, true);
        })
    });
}

criterion_group!(
    benches,
    benchmark_build_tree_single_threaded,
    benchmark_build_tree_multi_threaded
);
criterion_main!(benches);
