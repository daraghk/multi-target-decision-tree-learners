#[path = "threshold_finders/threshold_finder_variance.rs"]
mod threshold_finder_variance;
use common::question::Question;

use super::*;
use crate::calculations::variance::*;

pub fn find_best_split(data: &MultiTargetDataSet, number_of_targets: u32) -> BestSplitResult {
    let mut best_gain = 0.0;
    let mut best_question = Question::new(0, 0.);

    let number_of_targets = number_of_targets as usize;
    let total_multi_target_label_metrics =
        get_multi_target_label_metrics(&data.labels, number_of_targets);
    let total_variance_sum = get_total_variance_sum(
        &total_multi_target_label_metrics,
        data.labels.len() as f32,
        number_of_targets,
    );
    
    for(i, v) in data.features[0].iter().enumerate(){
        let best_threshold_for_feature = threshold_finder_variance::determine_best_threshold(
            data,
            i as u32,
            &total_multi_target_label_metrics,
            number_of_targets,
        );

        let information_gain = total_variance_sum - best_threshold_for_feature.loss;
        if information_gain > best_gain {
            best_gain = information_gain;
            best_question.column = i as u32;
            best_question.value = best_threshold_for_feature.threshold_value;
        }
    }

    BestSplitResult {
        gain: best_gain,
        question: best_question,
    }
}

fn get_total_variance_sum(
    total_multi_target_label_metrics: &MultiTargetLabelMetrics,
    number_of_labels: f32,
    number_of_targets: usize,
) -> f32 {
    let total_variance_vector = calculate_variance_vector(
        total_multi_target_label_metrics,
        number_of_labels,
        number_of_targets,
    );
    total_variance_vector.iter().sum()
}

#[cfg(test)]
mod tests {
    use common::{data_reader::read_csv_data_one_hot_multi_target, datasets::MultiTargetDataSet};

    #[test]
    fn test_find_best_split_dummy() {
        let features = vec![vec![10., 2., 0.], vec![6., 2., 0.], vec![1., 2., 1.]];
        let labels = vec![vec![1., 0.], vec![1., 0.], vec![1., 1.]];
        let indices = (0..labels.len()).collect::<Vec<usize>>();
        let data = MultiTargetDataSet {
            features,
            labels,
            indices,
        };
        let result = super::find_best_split(&data, 2);
        assert_eq!(result.question.value, 6.);
    }

    #[test]
    fn test_find_first_best_split_iris() {
        let iris = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let result = super::find_best_split(&iris, 3);
        assert_eq!(result.question.column, 2);
        assert_eq!(result.question.value, 30.);
    }
}
