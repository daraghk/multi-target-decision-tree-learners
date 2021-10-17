pub(super) fn calculate_loss(
    left_variance: f32,
    right_variance: f32,
    left_size: f32,
    right_size: f32,
) -> f32 {
    let total_size = left_size + right_size;
    ((left_size / total_size) * left_variance) + ((right_size / total_size) * right_variance)
}

pub(super) fn calculate_variance(
    sum_of_squared_labels: f32,
    mean_of_labels: f32,
    number_of_labels: f32,
) -> f32 {
    if number_of_labels == 0.0 {
        return 0.0;
    }
    let left = sum_of_squared_labels;
    let right = number_of_labels * (mean_of_labels * mean_of_labels);
    let variance = (left - right) / number_of_labels;
    variance
}

pub(super) fn get_label_sums(labels: &Vec<f32>) -> (f32, f32) {
    let mut sum_of_labels = 0.0;
    let mut sum_of_squared_labels = 0.0;
    labels.iter().for_each(|label| {
        let label_value = *label;
        sum_of_labels += label_value;
        sum_of_squared_labels += label_value * label_value;
    });
    (sum_of_labels, sum_of_squared_labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_variance_calculations() {
        let data = vec![vec![1, 2, 1], vec![2, 2, 2], vec![4, 2, 3], vec![5, 2, 4]];
        let true_variance = test_calculation_functions::variance(&data);
        let calulated_variance = calculate_variance(30.0, 2.5, 4.0);
        assert_eq!(true_variance, calulated_variance);
    }

    mod test_calculation_functions {
        pub fn split_variance(left_data: &Vec<Vec<i32>>, right_data: &Vec<Vec<i32>>) -> f32 {
            let total_data_size = left_data.len() + right_data.len();
            let left_variance = variance(left_data);
            let right_variance = variance(right_data);
            ((left_data.len() / total_data_size) as f32 * left_variance)
                + ((right_data.len() / total_data_size) as f32 * right_variance)
        }

        pub fn variance(data: &Vec<Vec<i32>>) -> f32 {
            let mean = output_mean(data);
            let mut sum_differences_squared = 0.0;
            data.iter().for_each(|row| {
                let output_value = row[row.len() - 1];
                let difference = output_value as f32 - mean;
                sum_differences_squared += difference * difference;
            });
            sum_differences_squared / data.len() as f32
        }

        fn output_mean(data: &Vec<Vec<i32>>) -> f32 {
            let mut sum = 0.0;
            data.iter().for_each(|row| {
                let output_value = row[row.len() - 1];
                sum += output_value as f32;
            });
            sum / data.len() as f32
        }
    }
}
