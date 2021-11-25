use common::data_reader::read_csv_data_multi_target;
use criterion::{criterion_group, criterion_main, Criterion};
use multi_target_decision_tree::{decision_trees::TreeConfig, split_finder::{SplitFinder, SplitMetric}};
use multi_target_grad_boost::grad_boost_ensemble::GradientBoostedEnsemble;

fn benchmark_build_tree_single_threaded(c: &mut Criterion) {
    let data_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );
    let split_finder = SplitFinder::new(SplitMetric::Variance);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: true,
        number_of_classes: 5,
        max_levels: 8,
    };

    c.bench_function("multi target grad boost tree build - single thread", |b| {
        b.iter(|| {
            return GradientBoostedEnsemble::train(data_set.clone(), tree_config, 100, 0.1);
        })
    });
}

criterion_group!(
    benches,
    benchmark_build_tree_single_threaded,
    //benchmark_build_tree_multi_threaded
);
criterion_main!(benches);
