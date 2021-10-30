use crate::{leaf::Leaf, node::TreeNode};
use common::datasets::MultiTargetDataSet;

pub fn calculate_accuracy(
    test_data: &MultiTargetDataSet,
    tree_root: &Box<TreeNode>,
    number_of_classes: u32,
) -> f32 {
    let mut accuracy = 0.;
    for i in 0..test_data.features.len() {
        let prediction = predict_class(
            &test_data.features[i],
            tree_root,
            number_of_classes as usize,
        );
        let actual = &test_data.labels[i];
        if prediction == *actual {
            accuracy += 1.;
        } else {
            //for debugging - print incorrect classifications
            //println!("Prediction: {:?}, Actual: {:?}", prediction, actual);
        }
    }
    accuracy / test_data.features.len() as f32
}

pub fn predict_class(
    feature_row: &Vec<f32>,
    node: &Box<TreeNode>,
    number_of_classes: usize,
) -> Vec<f32> {
    let leaf = classify(feature_row, node);
    let mut max = 0;
    let mut max_class = 0.;
    let mut index = 0.;
    leaf.predictions
        .as_ref()
        .unwrap()
        .counts
        .iter()
        .for_each(|count| {
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

fn classify<'a>(feature_row: &Vec<f32>, node: &'a Box<TreeNode>) -> &'a Leaf {
    if !node.is_leaf_node() {
        if node.question.solve(feature_row) {
            return classify(feature_row, &node.true_branch.as_ref().unwrap());
        } else {
            return classify(feature_row, &node.false_branch.as_ref().unwrap());
        }
    }
    node.leaf.as_ref().unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decision_tree::{print_tree, MultiTargetDecisionTree},
        split_finder::{SplitFinder, SplitMetric},
    };
    use common::data_reader::{get_feature_names, read_csv_data_one_hot_multi_target};

    #[test]
    fn test_classifier_known_data() {
        let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = MultiTargetDecisionTree::new(data_set, split_finder, 3, false, false);
        let row_to_classify = vec![58., 27., 51., 19.];
        let boxed_tree = Box::new(tree.root);
        let predicted_class = predict_class(&row_to_classify, &boxed_tree, 3);
        assert_eq!(predicted_class, vec![0., 0., 1.]);
    }

    use super::*;
    #[test]
    fn test_print_classifier_result_unknown_data() {
        let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = MultiTargetDecisionTree::new(data_set, split_finder, 3, false, false);
        let row_to_classify = vec![1., 23., 90., 10.];
        let boxed_tree = Box::new(tree.root);
        let predicted_class = predict_class(&row_to_classify, &boxed_tree, 3);
        println!("{:?}", predicted_class);
    }

    #[test]
    fn test_overall_accuracy_on_iris_training_data() {
        let train_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = MultiTargetDecisionTree::new(train_set, split_finder, 3, false, false);
        let boxed_tree = Box::new(tree.root);

        let test_set =
            read_csv_data_one_hot_multi_target("./../common/data-files/iris_test.csv", 3);
        let accuracy = calculate_accuracy(&test_set, &boxed_tree, 3);
        assert_eq!(accuracy, 1.0);
    }

    #[test]
    fn test_overall_accuracy_on_iris_test_data() {
        let train_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = MultiTargetDecisionTree::new(train_set, split_finder, 3, false, false);
        let boxed_tree = Box::new(tree.root);

        let test_set =
            read_csv_data_one_hot_multi_target("./../common/data-files/iris_test.csv", 3);
        let accuracy = calculate_accuracy(&test_set, &boxed_tree, 3);
        assert_eq!(accuracy, 1.0);
    }

    #[test]
    fn print_iris_tree_for_ref() {
        let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = MultiTargetDecisionTree::new(data_set, split_finder, 3, false, false);
        let feature_names = get_feature_names("./../common/data-files/iris.csv");
        print_tree(Box::new(tree.root), "".to_string(), &feature_names)
    }
}
