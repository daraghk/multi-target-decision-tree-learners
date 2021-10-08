use core::num;
use std::{collections::HashSet, hash::*};

use crate::{class_counter::ClassCounter, data::DataSet, question::Question};

pub mod gini {
    use super::*;
    use crate::class_counter::{self, ClassCounter};
    pub fn calculate_loss(
        number_of_rows: f32,
        true_rows_count: f32,
        class_counts_left: &ClassCounter,
        class_counts_right: &ClassCounter,
    ) -> f32 {
        let false_rows_count = number_of_rows - true_rows_count;
        let gini_left = calculate_gini(class_counts_left, false_rows_count);
        let gini_right = calculate_gini(class_counts_right, true_rows_count);
        let sum = (false_rows_count * gini_left) + (true_rows_count * gini_right);
        let result = sum / number_of_rows;
        result
    }

    pub fn calculate_gini(class_counts: &ClassCounter, number_of_rows: f32) -> f32 {
        let impurity: f32 = 1.0;
        let mut reduction: f32 = 0.0;
        class_counts.counts.iter().for_each(|class_count| {
            let probability_i = *class_count as f32 / number_of_rows;
            reduction += probability_i * probability_i;
        });
        impurity - reduction
    }
}

pub mod variance_reduction {
    pub fn calculate_loss(
        left_variance: f32,
        right_variance: f32,
        left_size: f32,
        right_size: f32,
    ) -> f32 {
        let total_size = left_size + right_size;
        ((left_size / total_size) * left_variance) + ((right_size / total_size) * right_variance)
    }

    pub fn calculate_variance(
        sum_of_squared_labels: f32,
        mean_of_labels: f32,
        number_of_labels: f32,
    ) -> f32 {
        if number_of_labels == 0.0{
            return 0.0
        }
        let left = sum_of_squared_labels;
        let right = number_of_labels * (mean_of_labels * mean_of_labels);
        let variance = (left - right) / number_of_labels;
        variance
    }

    pub fn get_label_sums(labels: &Vec<i32>) -> (f32, f32) {
        let mut sum_of_labels = 0.0;
        let mut sum_of_squared_labels = 0.0;
        labels.iter().for_each(|label| {
            let label_value = *label as f32;
            sum_of_labels += label_value;
            sum_of_squared_labels += label_value * label_value;
        });
        (sum_of_labels, sum_of_squared_labels)
    }
}

pub fn get_class_counts(classes: &Vec<i32>, number_of_unique_classes: u32) -> ClassCounter {
    let mut class_counter = ClassCounter::new(number_of_unique_classes);
    classes.iter().for_each(|class| {
        class_counter.counts[*class as usize] += 1;
    });
    class_counter
}

pub fn partition(data: &DataSet, question: &Question) -> (DataSet, DataSet) {
    let mut true_rows = vec![];
    let mut false_rows = vec![];
    let mut true_labels = vec![];
    let mut false_labels = vec![];

    let mut index: usize = 0;
    data.features.iter().for_each(|row| {
        let current_label = *data.labels.get(index).unwrap();
        index += 1;
        if (question.solve(row)) {
            true_rows.push(row.clone());
            true_labels.push(current_label);
        } else {
            false_rows.push(row.clone());
            false_labels.push(current_label);
        }
    });

    let false_data = DataSet{
        features: false_rows,
        labels: false_labels
    };

    let true_data = DataSet{
        features: true_rows,
        labels: true_labels
    };

    (false_data, true_data)
}

#[cfg(test)]
mod tests {
    use crate::{calculations::gini::*, question};

    use super::*;

    #[test]
    fn test_get_class_counts_one_row_and_class() {
        //set up data
        let number_classes = 1;
        let data = vec![0];
        let class_counts = get_class_counts(&data, number_classes);
        assert_eq!(class_counts.counts.get(0).unwrap(), &1);
    }

    #[test]
    fn test_get_class_counts_multiple_rows_and_classes() {
        //set up data
        let number_classes = 2;
        let data = vec![0, 0, 1];
        let class_counts = get_class_counts(&data, number_classes);
        assert_eq!(class_counts.counts.get(0).unwrap(), &2);
        assert_eq!(class_counts.counts.get(1).unwrap(), &1);
    }

    #[test]
    fn test_gini_calculation_no_impurity() {
        let number_classes = 1;
        let data = vec![vec![1, 2], vec![1, 2], vec![1, 2]];
        let class_counts = get_class_counts(&vec![0, 0, 0], number_classes);
        let gini_result = calculate_gini(&class_counts, data.len() as f32);
        assert_eq!(gini_result, 0.0);
    }

    #[test]
    fn test_gini_calculation_has_impurity() {
        let number_classes = 3;
        let data = vec![vec![1, 2, 0], vec![1, 2, 1], vec![1, 2, 2]];
        let class_counts = get_class_counts(&vec![0, 1, 2], number_classes);
        let gini_result = calculate_gini(&class_counts, data.len() as f32);
        assert_eq!(gini_result, 1.0 - (1.0 / 3.0));
    }

    #[test]
    fn test_partition_all_false() {
        let features = vec![vec![2, 2, 0], vec![2, 2, 1], vec![2, 2, 2]];
        let labels = vec![0, 1, 2];
        let data = DataSet{
            features,
            labels
        };
        let question = Question::new(0, false, 2);
        let partitioned_data = partition(&data, &question);
        println!("{:?}", partitioned_data);
        assert_eq!(partitioned_data.0.features.len(), 0);
        assert_eq!(partitioned_data.1.features.len(), 3);
    }

    #[test]
    fn test_partition_all_true() {
        let features = vec![vec![2, 2, 0], vec![2, 2, 1], vec![2, 2, 2]];
        let labels = vec![0, 1, 2];
        let data = DataSet{
            features,
            labels
        };
        let question = Question::new(0, false, 0);
        let partitioned_data = partition(&data, &question);
        println!("{:?}", partitioned_data);
        assert_eq!(partitioned_data.0.features.len(), 0);
        assert_eq!(partitioned_data.1.features.len(), 3);
    }

    #[test]
    fn test_partition_even() {
        let features = vec![vec![1, 2, 0], vec![2, 2, 1], vec![4, 2, 2], vec![5, 2, 2]];
        let labels = vec![0, 1, 2, 3];
        let data = DataSet{
            features,
            labels
        };
        let question = Question::new(0, false, 3);
        let partitioned_data = partition(&data, &question);
        println!("{:?}", partitioned_data);
        assert_eq!(partitioned_data.0.features.len(), 2);
        assert_eq!(partitioned_data.1.features.len(), 2);
    }

    #[test]
    fn test_no_output_variance() {
        // let data = vec![vec![1, 2, 1], vec![2, 2, 1], vec![4, 2, 1], vec![5, 2, 1]];
        // let variance_result = variance_reduction::variance(&data);
        // assert_eq!(variance_result, 0.0);
    }

    #[test]
    fn test_compare_variance_calculations() {
        let data = vec![vec![1, 2, 1], vec![2, 2, 2], vec![4, 2, 3], vec![5, 2, 4]];
        let true_variance = test_calculation_functions::variance(&data);
        let calulated_variance = variance_reduction::calculate_variance(30.0, 2.5, 4.0);
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
                let difference = (output_value as f32 - mean);
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
