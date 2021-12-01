#[path = "threshold_finders/threshold_finder_variance.rs"]
mod threshold_finder_variance;
use std::{sync::Arc, thread};

use common::{
    question::Question,
    results::{self, BestThresholdResult},
};

use super::*;
use crate::calculations::variance::*;

pub fn find_best_split(data: &MultiTargetDataSet, number_of_targets: u32) -> BestSplitResult {
    let mut best_gain = 0.0;
    let mut best_question = Question::new(0, 0.);
    let number_of_labels = data.labels.len();
    let number_of_targets = number_of_targets as usize;

    let total_multi_target_label_metrics =
        get_multi_target_label_metrics(&data.labels, number_of_targets);
    let total_variance_sum = get_total_variance_sum(
        &total_multi_target_label_metrics,
        number_of_labels as f64,
        number_of_targets,
    );

    let number_of_cols = data.feature_rows[0].len();
    let mut result_vector: Vec<BestThresholdResult> = Vec::with_capacity(number_of_cols);

    let mut thread_handles = vec![];
    let arc_data = Arc::new(data.to_owned());
    let arc_total_metrics = Arc::new(total_multi_target_label_metrics);

    for i in 0..number_of_cols {
        let arc_data_clone = arc_data.clone();
        let arc_total_metrics_clone = arc_total_metrics.clone();
        thread_handles.push(thread::spawn(move || {
            let best_threshold_for_feature = threshold_finder_variance::determine_best_threshold(
                number_of_labels,
                &arc_data_clone.labels,
                &arc_data_clone.feature_columns[i],
                &arc_total_metrics_clone,
                number_of_targets,
            );
            return best_threshold_for_feature;
        }));
    }

    for thread in thread_handles {
        let result = thread.join().unwrap();
        result_vector.push(result);
    }

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
        data_reader::{create_feature_columns, read_csv_data_one_hot_multi_target},
        datasets::MultiTargetDataSet,
    };

    #[test]
    fn test_find_best_split_dummy() {
        let features = vec![vec![10., 2., 0.], vec![6., 2., 0.], vec![1., 2., 1.]];
        let labels = vec![vec![1., 0.], vec![1., 0.], vec![1., 1.]];
        let indices = (0..labels.len()).collect::<Vec<usize>>();
        let columns = create_feature_columns(&features);
        let data = MultiTargetDataSet {
            feature_rows: features,
            feature_columns: columns,
            labels,
            indices,
        };
        let result = super::find_best_split(&data, 2);
        assert_eq!(result.question.value, 6.);
    }

    #[test]
    fn test_find_first_best_split_iris() {
        let iris = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let result = super::find_best_split(&iris, 3);
        assert_eq!(result.question.column, 2);
        assert_eq!(result.question.value, 30.);
    }
}
