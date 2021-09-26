mod threshold_finder_categorical;
mod threshold_finder_numeric;

use crate::{
    calculations::{get_class_counts, gini},
    question::Question,
};

#[derive(Debug)]
struct LastSeen {
    count: i32,
    value: i32,
}

#[derive(Debug)]
struct BestThresholdResult {
    loss: f32,
    threshold_value: f32,
}

#[derive(Debug)]
pub struct BestSplitResult {
    gain: f32,
    question: Question,
}

pub fn find_best_split(data: &Vec<Vec<i32>>) -> BestSplitResult {
    let mut best_gain = 0.0;
    let mut best_question = Question::new(0, false, 0);

    let class_counts_all = get_class_counts(data);
    let gini_all = gini(&class_counts_all, data.len() as f32);

    let last_feature_column_index = data[0].len();
    for i in (0..last_feature_column_index) {
        let best_threshold_for_feature = threshold_finder_numeric::determine_best_numeric_threshold(
            &data,
            i as i32,
            &class_counts_all,
        );

        let information_gain = gini_all - best_threshold_for_feature.loss;
        if (information_gain > best_gain) {
            best_gain = information_gain;
            best_question.column = i as i32;
            best_question.value = best_threshold_for_feature.threshold_value as i32;
        }
    }

    BestSplitResult {
        gain: best_gain,
        question: best_question,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn test_find_best_threshold_dummy() {
        let data = vec![vec![10, 2, 2], vec![6, 2, 2], vec![1, 2, 1]];
        let result = find_best_split(&data);
        assert_eq!(result.question.value, 6);
    }

    #[test]
    fn test_find_first_best_threshold_iris() {
        let _data = fs::read_to_string("./data_arff/iris.arff").expect("Unable to read file");
        let iris: Vec<Vec<i32>> = arff::from_str(&_data).unwrap();
        let result = find_best_split(&iris);
        assert_eq!(result.question.column, 2);
        assert_eq!(result.question.value, 30);
    }
}
