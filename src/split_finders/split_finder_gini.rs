#[path = "threshold_finders/threshold_finder_gini.rs"]
mod threshold_finder_gini;
use super::*;
use crate::{
    calculations::gini::calculate_gini, class_counter::get_class_counts, dataset::DataSet,
    question::Question,
};

pub fn find_best_split(data: &DataSet<i32, i32>, number_of_classes: u32) -> BestSplitResult {
    let mut best_gain = 0.0;
    let mut best_question = Question::new(0, false, 0);

    let class_counts_all = get_class_counts(&data.labels, number_of_classes);
    let gini_all = calculate_gini(&class_counts_all, data.features.len() as f32);

    let number_of_features = data.features[0].len();
    for i in 0..number_of_features {
        let best_threshold_for_feature =
            threshold_finder_gini::determine_best_threshold(data, i as u32, &class_counts_all);

        let information_gain = gini_all - best_threshold_for_feature.loss;
        if information_gain > best_gain {
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

#[cfg(test)]
mod tests {
    use crate::{
        data_reader::read_csv_data, dataset::DataSet, split_finder::get_sorted_feature_tuple_vector,
    };

    #[test]
    fn test_find_best_threshold_dummy() {
        let features = vec![vec![10, 2, 0], vec![6, 2, 0], vec![1, 2, 1]];
        let labels = vec![0, 0, 1];
        let data = DataSet { features, labels };
        let result = super::find_best_split(&data, 2);
        assert_eq!(result.question.value, 6);
    }

    #[test]
    fn test_get_sorted_feature_tuple_vector() {
        let features = vec![vec![10, 2, 1], vec![6, 2, 2], vec![1, 2, 3]];
        let labels = vec![1, 2, 3];
        let data = DataSet { features, labels };
        let column = 0;
        let sorted_feature_tuple_vector = get_sorted_feature_tuple_vector(&data.features, column);
        println!("{:?}", sorted_feature_tuple_vector);
        assert_eq!(sorted_feature_tuple_vector, vec![(1, 2), (6, 1), (10, 0)])
    }

    #[test]
    fn test_find_first_best_threshold_iris() {
        let iris = read_csv_data("./data_arff/iris.csv", false);
        let result = super::find_best_split(&iris, 3);
        assert_eq!(result.question.column, 2);
        assert_eq!(result.question.value, 30);
    }
}
