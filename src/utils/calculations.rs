use crate::{
    class_counter::{self, ClassCounter},
    question::Question,
};
use core::num;
use std::hash::*;

pub fn calculate_loss(
    number_of_rows: f32,
    true_rows_count: f32,
    class_counts_left: &ClassCounter<i32, i32>,
    class_counts_right: &ClassCounter<i32, i32>,
) -> f32 {
    let false_rows_count = number_of_rows - true_rows_count;
    let gini_left = gini(class_counts_left, false_rows_count);
    let gini_right = gini(class_counts_right, true_rows_count);
    let sum = (false_rows_count * gini_left) + (true_rows_count * gini_right);
    let result = sum / number_of_rows;
    result
}

pub fn get_class_counts(data: &Vec<Vec<i32>>) -> ClassCounter<i32, i32> {
    let mut class_counter = ClassCounter::new();
    data.iter().for_each(|row| {
        let class = row[row.len() - 1];
        let count = class_counter.map.entry(class).or_insert(0);
        *count += 1;
    });
    class_counter
}

pub fn gini(class_counts: &ClassCounter<i32, i32>, number_of_rows: f32) -> f32 {
    let impurity: f32 = 1.0;
    let mut reduction: f32 = 0.0;
    class_counts.map.iter().for_each(|key_value| {
        let probability_i = *key_value.1 as f32 / number_of_rows;
        reduction += probability_i * probability_i;
    });
    impurity - reduction
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
    (true_rows, false_rows)
}

#[cfg(test)]
mod tests {
    use crate::question;

    use super::*;

    #[test]
    fn test_get_class_counts_one_row_and_class() {
        //set up data
        let data = vec![vec![1, 2, 3]];
        let class_counts = get_class_counts(&data);
        assert_eq!(class_counts.map.get(&3).unwrap(), &1);
    }

    #[test]
    fn test_get_class_counts_multiple_rows_and_classes() {
        //set up data
        let data = vec![vec![1, 2, 3], vec![1, 2, 3], vec![1, 2, 2]];
        let class_counts = get_class_counts(&data);
        assert_eq!(class_counts.map.get(&3).unwrap(), &2);
        assert_eq!(class_counts.map.get(&2).unwrap(), &1);
    }

    #[test]
    fn test_gini_calculation_no_impurity() {
        let data = vec![vec![1, 2, 2], vec![1, 2, 2], vec![1, 2, 2]];
        let class_counts = get_class_counts(&data);
        let gini_result = gini(&class_counts, data.len() as f32);
        assert_eq!(gini_result, 0.0);
    }

    #[test]
    fn test_gini_calculation_has_impurity() {
        let data = vec![vec![1, 2, 1], vec![1, 2, 2], vec![1, 2, 3]];
        let class_counts = get_class_counts(&data);
        let gini_result = gini(&class_counts, data.len() as f32);
        assert_eq!(gini_result, 1.0 - (1.0 / 3.0));
    }

    #[test]
    fn test_partition_all_false() {
        let data = vec![vec![1, 2, 1], vec![1, 2, 2], vec![1, 2, 3]];
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
}
