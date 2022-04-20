use common::{
    data_processor::{self, process_dataset},
    data_reader::read_csv_data_multi_target,
    question::Question,
};
use criterion::{criterion_group, criterion_main, Criterion};
use multi_target_decision_tree::data_partitioner::partition;

fn bench_partition_current(c: &mut Criterion) {
    let data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );

    let question = Question::new(0, 1.);
    c.bench_function("current partitioning", |b| {
        b.iter(|| return partition(&data, &question))
    });
}

fn bench_partition_new(c: &mut Criterion) {
    let data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );
    let processed_dataset = process_dataset(data.clone());

    c.bench_function("new partitioning", |b| {
        b.iter(|| data_processor::new_partition(&processed_dataset, 0, 1., &data.labels))
    });
}

criterion_group!(benches, bench_partition_current, bench_partition_new);
criterion_main!(benches);
