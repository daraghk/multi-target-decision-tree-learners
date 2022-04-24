#[path = "threshold_finders/threshold_finder_variance.rs"]
mod threshold_finder_variance;
use rayon::prelude::*;
use std::sync::Arc;

use common::{
    datasets::MultiTargetDataSetSortedFeatures, question::Question, results::BestThresholdResult,
};

use super::*;
use crate::calculations::*;

pub(crate) fn find_best_split(
    data: &MultiTargetDataSetSortedFeatures,
    all_labels: &Vec<&Vec<f64>>,
    number_of_targets: u32,
    number_of_cols: usize,
) -> BestSplitResult {
    let mut best_gain = 0.0;
    let mut best_question = Question::new(0, 0.);
    let number_of_labels_in_subset = data.labels.len();
    let number_of_targets = number_of_targets as usize;

    let total_multi_target_label_metrics =
        get_multi_target_label_metrics(&data.labels, number_of_targets);
    let total_variance_sum = get_total_variance_sum(
        &total_multi_target_label_metrics,
        number_of_labels_in_subset as f64,
        number_of_targets,
    );

    let total_multi_target_label_metrics = Arc::new(total_multi_target_label_metrics);
    let result_vector: Vec<BestThresholdResult> = data
        .sorted_feature_columns
        .par_iter()
        .map(|feature_column| {
            threshold_finder_variance::determine_best_threshold(
                number_of_labels_in_subset,
                all_labels,
                feature_column,
                &total_multi_target_label_metrics,
                number_of_targets,
            )
        })
        .collect();

    assert_eq!(result_vector.len(), number_of_cols);

    for i in 0..number_of_cols {
        let feature_column_result = result_vector[i];
        let information_gain = total_variance_sum - feature_column_result.loss;
        if information_gain > best_gain {
            best_gain = information_gain;
            best_question.column = i as u32;
            best_question.value = feature_column_result.threshold_value;
        }
    }

    BestSplitResult {
        gain: best_gain,
        question: best_question,
    }
}

fn get_total_variance_sum(
    total_multi_target_label_metrics: &MultiTargetLabelMetrics,
    number_of_labels: f64,
    number_of_targets: usize,
) -> f64 {
    let total_variance_vector = calculate_variance_vector(
        total_multi_target_label_metrics,
        number_of_labels,
        number_of_targets,
    );
    total_variance_vector.iter().sum()
}

#[cfg(test)]
mod tests {
    use common::{
        data_processor::{self, create_dataset_with_sorted_features},
        data_reader::{create_feature_columns, read_csv_data_one_hot_multi_target},
        datasets::MultiTargetDataSet,
    };

    #[test]
    fn test_find_best_split_dummy() {
        let features = vec![vec![10., 2., 0.], vec![6., 2., 0.], vec![1., 2., 1.]];
        let labels = vec![vec![1., 0.], vec![1., 0.], vec![1., 1.]];

        let columns = create_feature_columns(&features);
        let number_of_cols = columns.len();
        let data = MultiTargetDataSet {
            feature_rows: features,
            feature_columns: columns,
            labels,
        };
        let processed_data = create_dataset_with_sorted_features(&data);
        let all_labels = processed_data.labels.clone();

        let result = super::find_best_split(&processed_data, &all_labels, 2, number_of_cols);
        println!("{:?}", result);
        assert_eq!(result.question.value, 6.);
    }

    #[test]
    fn test_find_first_best_split_iris() {
        let iris = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let number_of_cols = iris.feature_columns.len();
        let processed_data = create_dataset_with_sorted_features(&iris);
        let all_labels = processed_data.labels.clone();
        let result = super::find_best_split(&processed_data, &all_labels, 3, number_of_cols);
        assert_eq!(result.question.column, 2);
        assert_eq!(result.question.value, 30.);
    }
}
