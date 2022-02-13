use crate::{
    boosting_ensemble::common_multi_class_boosting_functions::executor_helper_functions::calculate_residuals,
    tree_traverse::find_leaf_node_for_data,
};
use multi_target_decision_tree::{
    decision_trees::{grad_boost_leaf_output::LeafOutputCalculator, AMGBoostTree, TreeConfig},
    leaf::AMGBoostLeaf,
    node::TreeNode,
};

use super::{calculate_approximate_value, GradBoostTrainingData};

pub(crate) fn execute_gradient_boosting_loop(
    training_data: &mut GradBoostTrainingData,
    number_of_iterations: u32,
    tree_config: TreeConfig,
    leaf_output_calculator: LeafOutputCalculator,
    learning_rate: f64,
) -> Vec<Box<TreeNode<AMGBoostLeaf>>> {
    let mut trees = Vec::with_capacity(number_of_iterations as usize);
    //Training data mutable labels gets altered in each iteration, dependency between each iteration, can't parallelise
    for _i in 0..number_of_iterations {
        let residuals = calculate_residuals(training_data);
        let mut learner_data = training_data.data.clone();
        learner_data.labels = residuals;
        let residual_tree = AMGBoostTree::new(learner_data, tree_config, leaf_output_calculator);
        let boxed_residual_tree = Box::new(residual_tree.root);
        update_dataset_labels(training_data, &boxed_residual_tree, learning_rate);
        trees.push(boxed_residual_tree);
    }
    trees
}

fn update_dataset_labels(
    training_data: &mut GradBoostTrainingData,
    boxed_tree_ref: &Box<TreeNode<AMGBoostLeaf>>,
    learning_rate: f64,
) {
    let number_of_classes = training_data.mutable_labels[0].len();
    let number_of_classes_f64 = number_of_classes as f64;

    //TODO this can be done concurrently in theory as different mutable labels are updated in each iteration
    for i in 0..training_data.size {
        let feature_row = &training_data.data.feature_rows[i];
        //TODO for each iteration here in this loop we traverse the current tree being looked at for each data instance
        let (weighted_max_value, weighted_non_max_value, max_value_index) =
            get_weighted_amg_boost_leaf_values(
                feature_row,
                boxed_tree_ref,
                learning_rate,
                number_of_classes_f64,
            );

        //update label values below, first max class then rest
        let current_label = &mut training_data.mutable_labels[i];
        current_label[max_value_index] += weighted_max_value;
        for j in 0..number_of_classes {
            if j != max_value_index {
                current_label[j] += weighted_non_max_value;
            }
        }
    }
}

fn get_weighted_amg_boost_leaf_values(
    feature_row: &[f64],
    boxed_tree_ref: &Box<TreeNode<AMGBoostLeaf>>,
    learning_rate: f64,
    number_of_classes: f64,
) -> (f64, f64, usize) {
    let leaf_data = find_leaf_node_for_data(feature_row, boxed_tree_ref);
    let max_value = leaf_data.max_value.unwrap();
    let weighted_max_value = learning_rate * max_value;
    let weighted_non_max_value =
        learning_rate * calculate_approximate_value(max_value, number_of_classes);
    let max_value_index = leaf_data.class.unwrap();
    (weighted_max_value, weighted_non_max_value, max_value_index)
}
