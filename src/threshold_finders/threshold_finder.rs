mod threshold_finder_gini;
mod threshold_finder_variance;

use crate::{calculations::get_class_counts, question::Question};

#[derive(Debug)]
struct BestThresholdResult {
    loss: f32,
    threshold_value: f32,
}

#[derive(Debug)]
pub struct BestSplitResult {
    pub gain: f32,
    pub question: Question,
}

pub mod gini {
    use super::*;
    use crate::calculations::gini::calculate_gini;
    
    pub fn find_best_split(data: &Vec<Vec<i32>>, number_of_classes: u32) -> BestSplitResult {
        let mut best_gain = 0.0;
        let mut best_question = Question::new(0, false, 0);

        let class_counts_all = get_class_counts(data, number_of_classes);
        let gini_all = calculate_gini(&class_counts_all, data.len() as f32);

        let last_feature_column_index = data[0].len() - 1;
        for i in (0..last_feature_column_index) {
            let best_threshold_for_feature =
                threshold_finder_gini::determine_best_threshold(&data, i as u32, &class_counts_all);

            let information_gain = gini_all - best_threshold_for_feature.loss;
            if (information_gain > best_gain) {
                best_gain = information_gain;
                best_question.column = i as u32;
                best_question.value = best_threshold_for_feature.threshold_value as i32;
            }
        }

        BestSplitResult {
            gain: best_gain,
            question: best_question,
        }
    }
}

pub mod variance {
    use super::*;
    use crate::calculations::variance_reduction::{calculate_variance, get_label_sums};
    
    pub fn find_best_split(data: &Vec<Vec<i32>>, number_of_classes: u32) -> BestSplitResult {
        let mut best_gain = 0.0;
        let mut best_question = Question::new(0, false, 0);

        let label_sums = get_label_sums(data);
        let sum_of_labels = label_sums.0;
        let sum_of_squared_labels = label_sums.1;
        let total_variance =
            get_total_variance(sum_of_labels, sum_of_squared_labels, data.len() as f32);

        let last_feature_column_index = data[0].len() - 1;
        for i in (0..last_feature_column_index) {
            let best_threshold_for_feature = threshold_finder_variance::determine_best_threshold(
                data,
                i as u32,
                sum_of_squared_labels,
                sum_of_labels,
            );

            let information_gain = total_variance - best_threshold_for_feature.loss;
            if (information_gain > best_gain) {
                best_gain = information_gain;
                best_question.column = i as u32;
                best_question.value = best_threshold_for_feature.threshold_value as i32;
            }
        }

        BestSplitResult {
            gain: best_gain,
            question: best_question,
        }
    }

    fn get_total_variance(
        sum_of_labels: f32,
        sum_of_squared_labels: f32,
        number_of_labels: f32,
    ) -> f32 {
        let mean_of_labels = sum_of_labels / number_of_labels;
        calculate_variance(sum_of_squared_labels, mean_of_labels, number_of_labels)
    }
}

fn get_sorted_feature_tuple_vector(data: &Vec<Vec<i32>>, column: u32) -> Vec<(i32, i32)> {
    let mut feature_tuple_vector = vec![];
    let mut row_index = 0;
    data.iter().for_each(|row| {
        let feature_value = row[column as usize];
        feature_tuple_vector.push((feature_value, row_index));
        row_index += 1;
    });
    feature_tuple_vector.sort_by_key(|tuple| tuple.0);
    feature_tuple_vector
}

#[cfg(test)]
mod tests {
    use super::*;
    mod gini_tests {
        use crate::threshold_finder::{get_sorted_feature_tuple_vector, gini};
        use std::fs;
        #[test]
        fn test_find_best_threshold_dummy() {
            let data = vec![vec![10, 2, 0], vec![6, 2, 0], vec![1, 2, 1]];
            let result = gini::find_best_split(&data, 2);
            assert_eq!(result.question.value, 6);
        }

        #[test]
        fn test_get_sorted_feature_tuple_vector() {
            let data = vec![vec![10, 2, 1], vec![6, 2, 2], vec![1, 2, 3]];
            let column = 0;
            let sorted_feature_tuple_vector = get_sorted_feature_tuple_vector(&data, column);
            println!("{:?}", sorted_feature_tuple_vector);
            assert_eq!(sorted_feature_tuple_vector, vec![(1, 2), (6, 1), (10, 0)])
        }

        #[test]
        fn test_find_first_best_threshold_iris() {
            let _data = fs::read_to_string("./data_arff/iris.arff").expect("Unable to read file");
            let iris: Vec<Vec<i32>> = arff::from_str(&_data).unwrap();
            let result = gini::find_best_split(&iris, 3);
            assert_eq!(result.question.column, 2);
            assert_eq!(result.question.value, 30);
        }
    }

    mod variance_tests {
        use crate::threshold_finder::{get_sorted_feature_tuple_vector, variance};
        use std::fs;
        #[test]
        fn test_find_best_threshold_dummy() {
            let data = vec![vec![10, 2, 0], vec![6, 2, 0], vec![1, 2, 1]];
            let result = variance::find_best_split(&data, 2);
            assert_eq!(result.question.value, 6);
        }

        #[test]
        fn test_get_sorted_feature_tuple_vector() {
            let data = vec![vec![10, 2, 1], vec![6, 2, 2], vec![1, 2, 3]];
            let column = 0;
            let sorted_feature_tuple_vector = get_sorted_feature_tuple_vector(&data, column);
            println!("{:?}", sorted_feature_tuple_vector);
            assert_eq!(sorted_feature_tuple_vector, vec![(1, 2), (6, 1), (10, 0)])
        }

        #[test]
        fn test_find_first_best_threshold_iris() {
            let _data = fs::read_to_string("./data_arff/iris.arff").expect("Unable to read file");
            let iris: Vec<Vec<i32>> = arff::from_str(&_data).unwrap();
            let result = variance::find_best_split(&iris, 3);
            assert_eq!(result.question.column, 2);
            assert_eq!(result.question.value, 30);
        }
    }
}
