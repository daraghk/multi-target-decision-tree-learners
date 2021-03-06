use crate::{
    class_counter::ClassCounter,
    leaf::{Leaf, RegressionLeaf},
    node::TreeNode,
};
use common::datasets::MultiTargetDataSet;

pub mod classification {
    use super::*;

    pub fn calculate_accuracy(
        test_data: &MultiTargetDataSet,
        tree_root: &Box<TreeNode<RegressionLeaf>>,
    ) -> f64 {
        let mut accuracy = 0.;
        for i in 0..test_data.feature_rows.len() {
            let prediction = predict_class(&test_data.feature_rows[i], tree_root);
            let actual = &test_data.labels[i];
            if prediction == *actual {
                accuracy += 1.;
            } else {
                //for debugging - print incorrect classifications
                //println!("Prediction: {:?}, Actual: {:?}", prediction, actual);
            }
        }
        accuracy / test_data.feature_rows.len() as f64
    }

    pub fn predict_class(feature_row: &Vec<f64>, node: &Box<TreeNode<RegressionLeaf>>) -> Vec<f64> {
        let leaf = find_leaf_node_for_data(feature_row, node);
        let labels = &leaf.data.as_ref().unwrap().labels;
        let number_of_classes = labels[0].len();
        let leaf_class_counts = get_class_counts_multi_target(labels, number_of_classes);
        let mut max = 0;
        let mut max_class = 0.;
        let mut index = 0.;
        leaf_class_counts.counts.iter().for_each(|count| {
            if *count > max {
                max = *count;
                max_class = index;
            }
            index += 1.;
        });
        let mut result = vec![0.; number_of_classes];
        result[max_class as usize] = 1.;
        result
    }

    fn get_class_counts_multi_target(
        labels: &Vec<Vec<f64>>,
        number_of_classes: usize,
    ) -> ClassCounter {
        let mut class_counter = ClassCounter::new(number_of_classes);
        labels.iter().for_each(|label_vector| {
            for i in 0..label_vector.len() {
                if label_vector[i] == 1. {
                    class_counter.counts[i] += 1;
                }
            }
        });
        class_counter
    }
}

pub mod regression {
    use crate::leaf::RegressionLeafNewPartition;

    use super::*;

    pub fn calculate_overall_mean_squared_error(
        test_data: &MultiTargetDataSet,
        tree_root: &Box<TreeNode<RegressionLeaf>>,
    ) -> f64 {
        let mut total_error = 0.;
        for i in 0..test_data.feature_rows.len() {
            let leaf = find_leaf_node_for_data(&test_data.feature_rows[i], tree_root);
            let prediction = calculate_average_label_vector(leaf.data.as_ref().unwrap());
            let actual = &test_data.labels[i];
            total_error += mean_sum_of_squared_differences_between_vectors(&prediction, actual);
        }
        total_error / test_data.feature_rows.len() as f64
    }

    pub fn calculate_overall_mean_squared_error_new_partition(
        test_data: &MultiTargetDataSet,
        tree_root: &Box<TreeNode<RegressionLeafNewPartition>>,
    ) -> f64 {
        let mut total_error = 0.;
        for i in 0..test_data.feature_rows.len() {
            let leaf = find_leaf_node_for_data(&test_data.feature_rows[i], tree_root);
            let prediction = {
                let data = leaf.data.as_ref().unwrap();
                let labels = &data.labels;
                let label_length = data.labels[0].len();
                let mut average_vector = vec![0.; label_length];
                for i in 0..data.labels.len() {
                    for j in 0..label_length {
                        average_vector[j] += labels[i][j];
                    }
                }
                for j in 0..label_length {
                    average_vector[j] /= labels.len() as f64;
                }
                average_vector
            };
            let actual = &test_data.labels[i];
            total_error += mean_sum_of_squared_differences_between_vectors(&prediction, actual);
        }
        total_error / test_data.feature_rows.len() as f64
    }

    fn calculate_average_label_vector(data: &MultiTargetDataSet) -> Vec<f64> {
        let labels = &data.labels;
        let label_length = data.labels[0].len();
        let mut average_vector = vec![0.; label_length];
        for i in 0..data.labels.len() {
            for j in 0..label_length {
                average_vector[j] += labels[i][j];
            }
        }
        for j in 0..label_length {
            average_vector[j] /= labels.len() as f64;
        }
        average_vector
    }

    fn mean_sum_of_squared_differences_between_vectors(
        prediction: &Vec<f64>,
        actual: &Vec<f64>,
    ) -> f64 {
        let mut sum_of_squared_differences = 0.;
        for i in 0..prediction.len() {
            let error = prediction[i] - actual[i];
            sum_of_squared_differences += f64::powf(error, 2.);
        }
        sum_of_squared_differences / prediction.len() as f64
    }
}

fn find_leaf_node_for_data<'a, L: Leaf>(
    feature_row: &Vec<f64>,
    node: &'a Box<TreeNode<L>>,
) -> &'a L {
    if !node.is_leaf_node() {
        if node.question.solve(feature_row) {
            return find_leaf_node_for_data(feature_row, &node.true_branch.as_ref().unwrap());
        } else {
            return find_leaf_node_for_data(feature_row, &node.false_branch.as_ref().unwrap());
        }
    }
    node.leaf.as_ref().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decision_trees::{RegressionMultiTargetDecisionTree, TreeConfig},
        scorer::classification::{calculate_accuracy, predict_class},
        split_finder::{SplitFinder, SplitMetric},
    };
    use common::data_reader::read_csv_data_one_hot_multi_target;

    #[test]
    fn test_classifier_known_data() {
        let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);

        let tree_config = TreeConfig {
            split_finder,
            use_multi_threading: false,
            number_of_classes: 3,
            max_levels: 8,
        };

        let tree = RegressionMultiTargetDecisionTree::new(data_set, tree_config);
        let row_to_classify = vec![58., 27., 51., 19.];
        let boxed_tree = Box::new(tree.root);
        let predicted_class = predict_class(&row_to_classify, &boxed_tree);
        assert_eq!(predicted_class, vec![0., 0., 1.]);
    }

    #[test]
    fn test_overall_accuracy_on_iris_test_data() {
        let train_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);

        let tree_config = TreeConfig {
            split_finder,
            use_multi_threading: false,
            number_of_classes: 3,
            max_levels: 8,
        };

        let tree = RegressionMultiTargetDecisionTree::new(train_set, tree_config);
        let boxed_tree = Box::new(tree.root);

        let test_set =
            read_csv_data_one_hot_multi_target("./../common/data-files/iris_test.csv", 3);
        let accuracy = calculate_accuracy(&test_set, &boxed_tree);
        assert_eq!(accuracy, 1.0);
    }
}
