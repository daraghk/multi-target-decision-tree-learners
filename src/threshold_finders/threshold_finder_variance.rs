use core::num;

use crate::calculations;

use super::{get_sorted_feature_tuple_vector, BestThresholdResult};

pub(super) fn determine_best_threshold(
    data: &Vec<Vec<i32>>,
    column: u32,
    total_sum: f32,
) -> BestThresholdResult {
    let mut best_result_container = BestThresholdResult {
        loss: f32::INFINITY,
        threshold_value: 0.0,
    };

    let mut left_sum = 0.0;
    let mut right_sum = total_sum;

    let mut left_size: f32 = 0.0;
    let mut right_size: f32 = data.len() as f32;

    let sorted_feature_data = get_sorted_feature_tuple_vector(data, column);
    sorted_feature_data.iter().for_each(|tuple| {
        let feature_val = tuple.0 as f32;
        let left_variance = calculate_variance(left_sum, left_size);
        let right_variance = calculate_variance(right_sum, right_size);
        let split_variance =
            calculate_split_variance(left_variance, right_variance, left_size, right_size);
        if (split_variance < best_result_container.loss) {
            best_result_container.loss = split_variance;
            best_result_container.threshold_value = feature_val as f32;
        }
        let real_row_index = tuple.1;
        let row = &data[real_row_index as usize];
        let label_value = row[row.len() - 1] as f32;
        right_sum -= label_value;
        right_size -= 1.0;
        left_sum += label_value;
        left_size += 1.0;
    });
    best_result_container
}

fn calculate_split_variance(
    left_variance: f32,
    right_variance: f32,
    left_size: f32,
    right_size: f32,
) -> f32 {
    let total_size = left_size + right_size;
    ((left_size / total_size) * left_variance) + ((right_size / total_size) * right_variance)
}

fn calculate_variance(sum_of_labels: f32, number_of_labels: f32) -> f32 {
    let mean = sum_of_labels / number_of_labels;
    (1.0 / number_of_labels)
        * ((sum_of_labels * sum_of_labels) - (2.0 * mean * sum_of_labels) - (mean * mean))
}

#[test]
fn test_best_threshold_for_particular_feature() {
    let data = vec![vec![10, 2, 0], vec![6, 2, 0], vec![1, 2, 1]];
    let column = 0;
    //let total_variance = calculations::variance_reduction::variance(&data);
    let best = determine_best_threshold(&data, column, 1.0);
    //assert_eq!(best.loss, 0.0);
    assert_eq!(best.threshold_value, 6.0);
}
