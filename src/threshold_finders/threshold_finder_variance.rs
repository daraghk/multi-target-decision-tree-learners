use core::num;

use crate::calculations::{
    self,
    variance_reduction::{calculate_loss, calculate_variance},
};

use super::{get_sorted_feature_tuple_vector, BestThresholdResult};

struct VarianceValueTracker {
    number_of_labels: f32,
    sum_of_squared_labels: f32,
    sum_of_labels: f32,
    mean_of_labels: f32,
}

pub(super) fn determine_best_threshold(
    data: &Vec<Vec<i32>>,
    column: u32,
    total_sum_of_squared_labels: f32,
    total_sum_of_labels: f32,
) -> BestThresholdResult {
    let mut best_result_container = BestThresholdResult {
        loss: f32::INFINITY,
        threshold_value: 0.0,
    };

    let mut left_value_tracker = VarianceValueTracker {
        number_of_labels: 0.0,
        sum_of_squared_labels: 0.0,
        sum_of_labels: 0.0,
        mean_of_labels: 0.0,
    };

    let right_mean_of_labels = total_sum_of_labels / data.len() as f32;
    let mut right_value_tracker = VarianceValueTracker {
        number_of_labels: data.len() as f32,
        sum_of_squared_labels: total_sum_of_squared_labels,
        sum_of_labels: total_sum_of_labels,
        mean_of_labels: right_mean_of_labels,
    };

    let sorted_feature_data = get_sorted_feature_tuple_vector(data, column);
    sorted_feature_data.iter().for_each(|tuple| {
        let feature_value = tuple.0 as f32;
        let left_variance = calculate_variance(
            left_value_tracker.sum_of_squared_labels,
            left_value_tracker.mean_of_labels,
            left_value_tracker.number_of_labels,
        );
        let right_variance = calculate_variance(
            right_value_tracker.sum_of_squared_labels,
            right_value_tracker.mean_of_labels,
            right_value_tracker.number_of_labels,
        );
        let split_variance = calculate_loss(
            left_variance,
            right_variance,
            left_value_tracker.number_of_labels,
            right_value_tracker.number_of_labels,
        );
        if (split_variance < best_result_container.loss) {
            best_result_container.loss = split_variance;
            best_result_container.threshold_value = feature_value as f32;
        }

        let real_row_index = tuple.1;
        let row = &data[real_row_index as usize];
        let label_value = row[row.len() - 1] as f32;
        update_left_value_tracker(&mut left_value_tracker, label_value);
        update_right_value_tracker(&mut right_value_tracker, label_value);
    });
    best_result_container
}

fn update_left_value_tracker(left_value_tracker: &mut VarianceValueTracker, label_value: f32) {
    left_value_tracker.sum_of_squared_labels += label_value * label_value;
    left_value_tracker.number_of_labels += 1.0;
    left_value_tracker.sum_of_labels += label_value;
    left_value_tracker.mean_of_labels =
        left_value_tracker.sum_of_labels / left_value_tracker.number_of_labels;
}

fn update_right_value_tracker(right_value_tracker: &mut VarianceValueTracker, label_value: f32) {
    right_value_tracker.sum_of_squared_labels -= label_value * label_value;
    right_value_tracker.number_of_labels -= 1.0;
    right_value_tracker.sum_of_labels -= label_value;
    right_value_tracker.mean_of_labels =
        right_value_tracker.sum_of_labels / right_value_tracker.number_of_labels;
}

#[test]
fn test_best_threshold_for_particular_feature() {
    let data = vec![vec![10, 2, 0], vec![6, 2, 0], vec![1, 2, 1]];
    let column = 0;
    let best = determine_best_threshold(&data, column, 1.0, 1.0);
    assert_eq!(best.threshold_value, 6.0);
}
