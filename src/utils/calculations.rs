use core::num;
use std::{collections::HashSet, hash::*};

use crate::{class_counter::ClassCounter, question::Question};

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

    pub fn get_label_sums(data: &Vec<Vec<i32>>) -> (f32, f32) {
        let mut sum_of_labels = 0.0;
        let mut sum_of_squared_labels = 0.0;
        data.iter().for_each(|row| {
            let label = row[row.len() - 1] as f32;
            sum_of_labels += label;
            sum_of_squared_labels += label * label;
        });
        (sum_of_labels, sum_of_squared_labels)
    }
}

pub fn get_class_counts(data: &Vec<Vec<i32>>, number_of_classes: u32) -> ClassCounter {
    let mut class_counter = ClassCounter::new(number_of_classes);
    data.iter().for_each(|row| {
        let class = row[row.len() - 1];
        class_counter.counts[class as usize] += 1;
    });
    class_counter
}

pub fn partition(data: &Vec<Vec<i32>>, question: &Question) -> (Vec<Vec<i32>>, Vec<Vec<i32>>) {
    let mut true_rows = vec![];
    let mut false_rows = vec![];

    data.iter().for_each(|row| {
        if (question.solve(row)) {
            true_rows.push(row.clone());
        } else {
            false_rows.push(row.clone());
        }
    });
    (false_rows, true_rows)
}

#[cfg(test)]
mod tests {
    use crate::{calculations::gini::*, question};

    use super::*;

    #[test]
    fn test_get_class_counts_one_row_and_class() {
        //set up data
        let number_classes = 1;
        let data = vec![vec![1, 2, 0]];
        let class_counts = get_class_counts(&data, number_classes);
        assert_eq!(class_counts.counts.get(0).unwrap(), &1);
    }

    #[test]
    fn test_get_class_counts_multiple_rows_and_classes() {
        //set up data
        let number_classes = 2;
        let data = vec![vec![1, 2, 0], vec![1, 2, 0], vec![1, 2, 1]];
        let class_counts = get_class_counts(&data, number_classes);
        assert_eq!(class_counts.counts.get(0).unwrap(), &2);
        assert_eq!(class_counts.counts.get(1).unwrap(), &1);
    }

    #[test]
    fn test_gini_calculation_no_impurity() {
        let number_classes = 1;
        let data = vec![vec![1, 2, 0], vec![1, 2, 0], vec![1, 2, 0]];
        let class_counts = get_class_counts(&data, number_classes);
        let gini_result = calculate_gini(&class_counts, data.len() as f32);
        assert_eq!(gini_result, 0.0);
    }

    #[test]
    fn test_gini_calculation_has_impurity() {
        let number_classes = 3;
        let data = vec![vec![1, 2, 0], vec![1, 2, 1], vec![1, 2, 2]];
        let class_counts = get_class_counts(&data, number_classes);
        let gini_result = calculate_gini(&class_counts, data.len() as f32);
        assert_eq!(gini_result, 1.0 - (1.0 / 3.0));
    }

    #[test]
    fn test_partition_all_false() {
        let data = vec![vec![1, 2, 0], vec![1, 2, 1], vec![1, 2, 2]];
        let question = Question::new(0, false, 2);
        let partitioned_data = partition(&data, &question);
        println!("{:?}", partitioned_data);
        assert_eq!(partitioned_data.0.len(), 0);
        assert_eq!(partitioned_data.1.len(), 3);
    }

    #[test]
    fn test_partition_all_true() {
        let data = vec![vec![1, 2, 1], vec![1, 2, 2], vec![1, 2, 3]];
        let question = Question::new(0, false, 0);
        let partitioned_data = partition(&data, &question);
        println!("{:?}", partitioned_data);
        assert_eq!(partitioned_data.0.len(), 3);
        assert_eq!(partitioned_data.1.len(), 0);
    }

    #[test]
    fn test_partition_even() {
        let data = vec![vec![1, 2, 1], vec![2, 2, 2], vec![4, 2, 3], vec![5, 2, 3]];
        let question = Question::new(0, false, 3);
        let partitioned_data = partition(&data, &question);
        println!("{:?}", partitioned_data);
        assert_eq!(partitioned_data.0.len(), 2);
        assert_eq!(partitioned_data.1.len(), 2);
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
