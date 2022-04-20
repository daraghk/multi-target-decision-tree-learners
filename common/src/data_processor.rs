use std::collections::HashSet;

use crate::{
    datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures},
    feature_sorter::get_sorted_feature_tuple_vector,
};

pub fn process_dataset(original_dataset_copy: MultiTargetDataSet) -> MultiTargetDataSetSortedFeatures {
    // sort feature columns
    let mut sorted_feature_columns = Vec::new();
    for feature_column in original_dataset_copy.feature_columns.iter() {
        let sorted_feature_col = get_sorted_feature_tuple_vector(feature_column);
        sorted_feature_columns.push(sorted_feature_col);
    }
    MultiTargetDataSetSortedFeatures {
        labels: original_dataset_copy.labels.clone(),
        sorted_feature_columns,
    }
}

pub fn new_partition(
    dataset: &MultiTargetDataSetSortedFeatures,
    split_column: usize,
    split_value: f64,
    all_labels: &Vec<Vec<f64>>,
) -> (
    MultiTargetDataSetSortedFeatures,
    MultiTargetDataSetSortedFeatures,
) {
    let true_indices = collect_true_indices_for_partitioning(&dataset, split_column, split_value);

    let labels_split_up = collect_partitioned_labels(&true_indices, all_labels);
    let true_labels = labels_split_up.0;
    let false_labels = labels_split_up.1;

    let partitioned_feature_colummns = collect_partitioned_feature_columns(&dataset, split_value);
    let true_feature_columns = partitioned_feature_colummns.0;
    let false_feature_columns = partitioned_feature_colummns.1;

    let true_dataset = MultiTargetDataSetSortedFeatures {
        labels: true_labels,
        sorted_feature_columns: true_feature_columns,
    };
    let false_dataset = MultiTargetDataSetSortedFeatures {
        labels: false_labels,
        sorted_feature_columns: false_feature_columns,
    };

    (true_dataset, false_dataset)
}

fn collect_true_indices_for_partitioning(
    dataset: &MultiTargetDataSetSortedFeatures,
    split_column: usize,
    split_value: f64,
) -> HashSet<usize> {
    let mut true_indices = HashSet::new();
    let feature_column_split_on = &dataset.sorted_feature_columns[split_column];
    for feature_value_index_pair in feature_column_split_on.iter() {
        let value = feature_value_index_pair.0;
        let index = feature_value_index_pair.1;
        if value >= split_value {
            true_indices.insert(index);
        }
    }
    true_indices
}

fn collect_partitioned_labels(
    true_indices: &HashSet<usize>,
    all_labels: &Vec<Vec<f64>>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut true_labels = Vec::new();
    let mut false_labels = Vec::new();
    for i in 0..all_labels.len() {
        if true_indices.contains(&i) {
            true_labels.push(all_labels[i].clone());
        } else {
            false_labels.push(all_labels[i].clone());
        }
    }
    (true_labels, false_labels)
}

fn collect_partitioned_feature_columns(
    dataset: &MultiTargetDataSetSortedFeatures,
    split_value: f64,
) -> (Vec<Vec<(f64, usize)>>, Vec<Vec<(f64, usize)>>) {
    let mut true_feature_columns = Vec::new();
    let mut false_feature_columns = Vec::new();
    for feature_column in dataset.sorted_feature_columns.iter() {
        let mut true_feature_column = Vec::new();
        let mut false_feature_column = Vec::new();
        for feature_value_index_pair in feature_column {
            let value = feature_value_index_pair.0;
            if value >= split_value {
                true_feature_column.push(*feature_value_index_pair);
            } else {
                false_feature_column.push(*feature_value_index_pair);
            }
        }
        true_feature_columns.push(true_feature_column);
        false_feature_columns.push(false_feature_column);
    }
    (true_feature_columns, false_feature_columns)
}

#[cfg(test)]
mod tests {
    use crate::data_reader::read_csv_data_multi_target;

    use super::process_dataset;

    #[test]
    fn test_init() {
        let data_set = read_csv_data_multi_target(
            "./../common/data-files/multi-target/features_train_mt.csv",
            "./../common/data-files/multi-target/labels_train_mt.csv",
        );
        let sorted_features_dataset = process_dataset(data_set);
        println!("{:?}", sorted_features_dataset.sorted_feature_columns[1]);
    }
}
