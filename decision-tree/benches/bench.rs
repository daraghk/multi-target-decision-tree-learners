use common::data_reader::read_csv_data;
use criterion::{criterion_group, criterion_main, Criterion};
use decision_tree::{
    decision_tree::DecisionTree,
    split_finder::{SplitFinder, SplitMetric},
};

fn benchmark_build_tree_single_threaded(c: &mut Criterion) {
    let data_set = read_csv_data("./../common/data_files/iris.csv");
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    c.bench_function("tree build - single thread", |b| {
        b.iter(|| {
            return DecisionTree::new(data_set.clone(), split_finder, 3, false);
        })
    });
}

fn benchmark_build_tree_multi_threaded(c: &mut Criterion) {
    let data_set = read_csv_data("./../common/data_files/iris.csv");
    let split_finder = SplitFinder::new(SplitMetric::Variance);
    c.bench_function("tree build - multi thread", |b| {
        b.iter(|| {
            return DecisionTree::new(data_set.clone(), split_finder, 3, true);
        })
    });
}

criterion_group!(
    benches,
    benchmark_build_tree_single_threaded,
    benchmark_build_tree_multi_threaded
);
criterion_main!(benches);
