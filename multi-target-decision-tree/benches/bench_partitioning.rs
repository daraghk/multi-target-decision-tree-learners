use common::{
    data_processor::{self, create_dataset_with_sorted_features},
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
    let processed_dataset = create_dataset_with_sorted_features(&data);
    let mut label_refs = vec![];
    for label in processed_dataset.labels.iter() {
        label_refs.push(*label);
    }
    c.bench_function("new partitioning", |b| {
        b.iter(|| data_processor::new_partition(&processed_dataset, 0, 1., &label_refs))
    });
}

criterion_group!(benches, bench_partition_current, bench_partition_new);
criterion_main!(benches);
