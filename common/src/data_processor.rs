use crate::{
    datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures},
    feature_sorter::get_sorted_feature_tuple_vector,
};
use rayon::prelude::*;

pub fn create_dataset_with_sorted_features(
    original_dataset: &MultiTargetDataSet,
) -> MultiTargetDataSetSortedFeatures {
    let mut sorted_feature_columns = Vec::new();
    for feature_column in original_dataset.feature_columns.iter() {
        let sorted_feature_col = get_sorted_feature_tuple_vector(feature_column);
        sorted_feature_columns.push(sorted_feature_col);
    }

    let mut label_refs = vec![];
    for label in original_dataset.labels.iter() {
        label_refs.push(label);
    }
    MultiTargetDataSetSortedFeatures {
        labels: original_dataset.labels.clone(),
        sorted_feature_columns,
    }
}

pub fn partition(
    dataset: &MultiTargetDataSetSortedFeatures,
    split_column: usize,
    split_value: f64,
    all_labels: &Vec<Vec<f64>>,
) -> (
    MultiTargetDataSetSortedFeatures,
    MultiTargetDataSetSortedFeatures,
) {
    let all_labels_size = all_labels.len();
    let partitioned_indices =
        collect_indices_for_partitioning(&dataset, split_column, split_value, all_labels_size);
    let true_indices = partitioned_indices.0;
    let false_indices = partitioned_indices.1;

    let partitioned_labels = collect_partitioned_labels(&true_indices, &false_indices, all_labels);
    let true_labels = partitioned_labels.0;
    let false_labels = partitioned_labels.1;

    let partitioned_feature_colummns = collect_partitioned_feature_columns(&dataset, &true_indices);
    let feature_columns_with_true_values = partitioned_feature_colummns.0;
    let feature_columns_with_false_values = partitioned_feature_colummns.1;

    let true_dataset = MultiTargetDataSetSortedFeatures {
        labels: true_labels,
        sorted_feature_columns: feature_columns_with_true_values,
    };
    let false_dataset = MultiTargetDataSetSortedFeatures {
        labels: false_labels,
        sorted_feature_columns: feature_columns_with_false_values,
    };

    (true_dataset, false_dataset)
}

fn collect_indices_for_partitioning(
    dataset: &MultiTargetDataSetSortedFeatures,
    split_column: usize,
    split_value: f64,
    all_labels_size: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut true_indices = vec![0; all_labels_size];
    let mut false_indices = vec![0; all_labels_size];
    let sorted_feature_column_split_on = &dataset.sorted_feature_columns[split_column];

    let mut first_encounter_index = 0;
    for i in 0..sorted_feature_column_split_on.len() {
        let feature_value_index_pair = sorted_feature_column_split_on[i];
        let value = feature_value_index_pair.0;
        let index = feature_value_index_pair.1;
        if value >= split_value {
            first_encounter_index = i;
            break;
        }
        false_indices[index] = 1;
    }

    for i in first_encounter_index..sorted_feature_column_split_on.len() {
        let index = sorted_feature_column_split_on[i].1;
        true_indices[index] = 1;
    }
    (true_indices, false_indices)
}

fn collect_partitioned_labels(
    true_indices: &Vec<u8>,
    false_indices: &Vec<u8>,
    all_labels: &Vec<Vec<f64>>,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut true_labels = Vec::new();
    let mut false_labels = Vec::new();
    for i in 0..all_labels.len() {
        if true_indices[i] == 1 {
            true_labels.push(all_labels[i].clone());
        } else if false_indices[i] == 1 {
            false_labels.push(all_labels[i].clone());
        }
    }
    (true_labels, false_labels)
}

fn collect_partitioned_feature_columns(
    dataset: &MultiTargetDataSetSortedFeatures,
    true_indices: &Vec<u8>,
) -> (Vec<Vec<(f64, usize)>>, Vec<Vec<(f64, usize)>>) {
    let (true_feature_columns, false_feature_columns): (
        Vec<Vec<(f64, usize)>>,
        Vec<Vec<(f64, usize)>>,
    ) = dataset
        .sorted_feature_columns
        .par_iter()
        .map(|sorted_feature_column| {
            let total_column_length = sorted_feature_column.len();
            let mut true_feature_values = Vec::with_capacity(total_column_length);
            let mut false_feature_values = Vec::with_capacity(total_column_length);
            for feature_value_index_pair in sorted_feature_column {
                let index = feature_value_index_pair.1;
                if true_indices[index] == 1 {
                    true_feature_values.push(*feature_value_index_pair);
                } else {
                    false_feature_values.push(*feature_value_index_pair);
                }
            }
            (true_feature_values, false_feature_values)
        })
        .collect();

    (true_feature_columns, false_feature_columns)
}

#[cfg(test)]
mod tests {
    use crate::data_reader::read_csv_data_multi_target;

    use super::{
        collect_indices_for_partitioning, collect_partitioned_labels,
        create_dataset_with_sorted_features,
    };

    #[test]
    fn test_init() {
        let data_set = read_csv_data_multi_target(
            "./../common/data-files/multi-target/features_train_mt.csv",
            "./../common/data-files/multi-target/labels_train_mt.csv",
        );
        let sorted_features_dataset = create_dataset_with_sorted_features(&data_set);
        println!("{:?}", sorted_features_dataset.sorted_feature_columns[1]);
    }

    #[test]
    fn test_collect_indices_for_partitioning() {
        let data_set = read_csv_data_multi_target(
            "./../common/data-files/multi-target/features_train_mt.csv",
            "./../common/data-files/multi-target/labels_train_mt.csv",
        );

        let original_dataset_size = data_set.feature_rows.len();
        let all_labels_size = data_set.labels.len();
        let sorted_features_dataset = create_dataset_with_sorted_features(&data_set);

        let split_column = 1;
        let chosen_value_index = 100;
        let split_value =
            sorted_features_dataset.sorted_feature_columns[split_column][chosen_value_index].0;
        let partitioned_indices = collect_indices_for_partitioning(
            &sorted_features_dataset,
            split_column,
            split_value,
            all_labels_size,
        );
        let true_indices = partitioned_indices.0;
        let false_indices = partitioned_indices.1;

        let true_indices_count = true_indices.iter().filter(|&n| *n == 1).count();
        let false_indices_count = false_indices.iter().filter(|&n| *n == 1).count();
        assert_eq!(
            true_indices_count,
            original_dataset_size - chosen_value_index
        );
        assert_eq!(false_indices_count, chosen_value_index);
    }

    #[test]
    fn test_collect_partitioned_labels() {
        let data_set = read_csv_data_multi_target(
            "./../common/data-files/multi-target/features_train_mt.csv",
            "./../common/data-files/multi-target/labels_train_mt.csv",
        );

        let original_dataset_size = data_set.feature_rows.len();
        let all_labels = data_set.labels.clone();
        let all_labels_size = data_set.labels.len();
        let sorted_features_dataset = create_dataset_with_sorted_features(&data_set);

        let split_column = 1;
        let chosen_value_index = 100;
        let split_value =
            sorted_features_dataset.sorted_feature_columns[split_column][chosen_value_index].0;
        let partitioned_indices = collect_indices_for_partitioning(
            &sorted_features_dataset,
            split_column,
            split_value,
            all_labels_size,
        );
        let true_indices = partitioned_indices.0;
        let false_indices = partitioned_indices.1;

        let partitioned_labels =
            collect_partitioned_labels(&true_indices, &false_indices, &all_labels);
        println!(
            "{:?}",
            (partitioned_labels.0.len(), partitioned_labels.1.len())
        );
        assert_eq!(
            partitioned_labels.0.len(),
            original_dataset_size - chosen_value_index
        );
        assert_eq!(partitioned_labels.1.len(), chosen_value_index);
    }
}
