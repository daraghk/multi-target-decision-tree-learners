use common::{datasets::MultiTargetDataSet, results::BestThresholdResult};

use crate::calculations::variance::MultiTargetLabelMetrics;
use crate::{
    calculations::variance::{calculate_loss_vector, calculate_variance_vector},
    feature_sorter::get_sorted_feature_tuple_vector,
};

struct VarianceValueTrackerMultiTarget {
    number_of_labels: f64,
    multi_target_label_metrics: MultiTargetLabelMetrics,
}

pub(super) fn determine_best_threshold(
    number_of_labels: usize,
    labels: &Vec<Vec<f64>>,
    feature_column: &[f64],
    total_multi_target_label_metrics: &MultiTargetLabelMetrics,
    number_of_targets: usize,
) -> BestThresholdResult {
    let mut best_result_container = BestThresholdResult {
        loss: f64::INFINITY,
        threshold_value: 0.0,
    };

    let mut left_value_tracker = VarianceValueTrackerMultiTarget {
        number_of_labels: 0.0,
        multi_target_label_metrics: MultiTargetLabelMetrics {
            sum_of_squared_labels_vector: vec![0.0; number_of_targets],
            sum_of_labels_vector: vec![0.0; number_of_targets],
            mean_of_labels_vector: vec![0.0; number_of_targets],
        },
    };

    let mut right_value_tracker = VarianceValueTrackerMultiTarget {
        number_of_labels: number_of_labels as f64,
        multi_target_label_metrics: MultiTargetLabelMetrics {
            sum_of_squared_labels_vector: total_multi_target_label_metrics
                .sum_of_squared_labels_vector
                .clone(),
            sum_of_labels_vector: total_multi_target_label_metrics
                .sum_of_labels_vector
                .clone(),
            mean_of_labels_vector: total_multi_target_label_metrics
                .mean_of_labels_vector
                .clone(),
        },
    };

    let sorted_feature_data = get_sorted_feature_tuple_vector(feature_column);
    let previous_feature_val = sorted_feature_data[0].0;
    sorted_feature_data.iter().for_each(|tuple| {
        let feature_value = tuple.0 as f64;

        //only calculate 'loss' on first encounter of a feature value
        if feature_value != previous_feature_val {
            let left_variance_vector = calculate_variance_vector(
                &left_value_tracker.multi_target_label_metrics,
                left_value_tracker.number_of_labels,
                number_of_targets,
            );

            let right_variance_vector = calculate_variance_vector(
                &right_value_tracker.multi_target_label_metrics,
                right_value_tracker.number_of_labels,
                number_of_targets,
            );

            let split_variance = calculate_loss_vector(
                left_variance_vector,
                right_variance_vector,
                left_value_tracker.number_of_labels,
                right_value_tracker.number_of_labels,
                number_of_targets,
            );

            let split_variance_sum = split_variance.iter().sum();
            if split_variance_sum < best_result_container.loss {
                best_result_container.loss = split_variance_sum;
                best_result_container.threshold_value = feature_value;
            }
        }

        let real_row_index = tuple.1;
        let label_vector = &labels[real_row_index];
        update_left_value_tracker(&mut left_value_tracker, label_vector, number_of_targets);
        update_right_value_tracker(&mut right_value_tracker, label_vector, number_of_targets);
    });
    best_result_container
}

fn update_left_value_tracker(
    left_value_tracker: &mut VarianceValueTrackerMultiTarget,
    label_vector: &[f64],
    number_of_targets: usize,
) {
    left_value_tracker.number_of_labels += 1.0;
    let left_multi_label_metrics = &mut left_value_tracker.multi_target_label_metrics;
    for i in 0..number_of_targets {
        let label_value = label_vector[i];
        left_multi_label_metrics.sum_of_squared_labels_vector[i] += label_value * label_value;
        left_multi_label_metrics.sum_of_labels_vector[i] += label_value;
        left_multi_label_metrics.mean_of_labels_vector[i] =
            left_multi_label_metrics.sum_of_labels_vector[i] / left_value_tracker.number_of_labels
    }
}

fn update_right_value_tracker(
    right_value_tracker: &mut VarianceValueTrackerMultiTarget,
    label_vector: &[f64],
    number_of_targets: usize,
) {
    right_value_tracker.number_of_labels -= 1.0;
    let right_multi_label_metrics = &mut right_value_tracker.multi_target_label_metrics;

    for i in 0..number_of_targets {
        let label_value = label_vector[i];
        right_multi_label_metrics.sum_of_squared_labels_vector[i] -= label_value * label_value;
        right_multi_label_metrics.sum_of_labels_vector[i] -= label_value;
        right_multi_label_metrics.mean_of_labels_vector[i] =
            right_multi_label_metrics.sum_of_labels_vector[i] / right_value_tracker.number_of_labels
    }
}

mod tests {
    use common::{
        data_reader::{create_feature_columns, read_csv_data_one_hot_multi_target},
        datasets::MultiTargetDataSet,
    };

    use crate::calculations::variance::get_multi_target_label_metrics;

    #[test]
    fn test_best_threshold_for_particular_feature() {
        let features = vec![vec![10., 2., 0.], vec![6., 2., 0.], vec![1., 2., 1.]];
        let labels = vec![vec![1., 0.], vec![1., 0.], vec![0., 1.]];
        let total_mt_label_metrics = get_multi_target_label_metrics(&labels, 2);
        // let indices = (0..labels.len()).collect::<Vec<usize>>();
        let columns = create_feature_columns(&features);
        let data = MultiTargetDataSet {
            feature_rows: features,
            feature_columns: columns,
            labels,
            // indices,
        };
        let column = 0;
        let number_labels = data.labels.len();
        let best = super::determine_best_threshold(
            number_labels,
            &data.labels,
            &data.feature_columns[column],
            &total_mt_label_metrics,
            2,
        );
        assert_eq!(best.loss, 0.0);
        assert_eq!(best.threshold_value, 6.0);
        println!("{:?}", best);
    }

    #[test]
    fn test_best_threshold_for_particular_feature_in_iris() {
        let iris = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let column = 2;
        let total_mt_label_metrics = get_multi_target_label_metrics(&iris.labels, 3);
        let number_labels = iris.labels.len();
        let best = super::determine_best_threshold(
            number_labels,
            &iris.labels,
            &iris.feature_columns[column],
            &total_mt_label_metrics,
            2,
        );
        println!("{:?}", best);
        assert_eq!(best.threshold_value, 30.0);
    }
}
