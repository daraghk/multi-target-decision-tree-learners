use std::collections::HashMap;

use crate::{
    boosting_ensemble::common_multi_class_boosting_functions::executor_helper_functions::calculate_residuals,
    tree_traverse::find_leaf_node_for_data,
};
use common::data_processor::create_dataset_with_sorted_features;
use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    grad_boost_decision_trees::{
        grad_boost_leaf_output::{LeafOutputCalculator, LeafOutputType},
        AMGBoostTree,
    },
    leaf::AMGBoostLeaf,
    node::TreeNode,
};

use super::{calculate_approximate_value, GradBoostTrainingData};

//TODO THIS NEEDS CLEANUP FOR UPDATING MUTABLE LABELS ETC
pub(crate) fn execute_gradient_boosting_loop(
    training_data: &mut GradBoostTrainingData,
    number_of_iterations: u32,
    tree_config: TreeConfig,
    learning_rate: f64,
) -> Vec<Box<TreeNode<AMGBoostLeaf>>> {
    let mut trees = Vec::with_capacity(number_of_iterations as usize);
    let leaf_output_calculator =
        LeafOutputCalculator::new(LeafOutputType::MultiClassClassification);
    let processed_data = create_dataset_with_sorted_features(&training_data.data);
    for _i in 0..number_of_iterations {
        //TODO how to remove / make faster?
        //TODO how to get rid of clone - processed_data never changes - at least the feature cols don't
        let mut learner_data = processed_data.clone();

        let residuals = calculate_residuals(training_data);
        learner_data.labels = residuals;

        let residuals = calculate_residuals(training_data);
        learner_data.labels = residuals;

        let residual_tree = AMGBoostTree::new(learner_data, tree_config, leaf_output_calculator);
        let boxed_residual_tree = Box::new(residual_tree.root);
        {
            let boxed_tree_ref = &boxed_residual_tree;
            let number_of_classes = training_data.mutable_labels[0].len();
            let number_of_classes_f64 = number_of_classes as f64;

            //TODO this can be done concurrently in theory as different mutable labels are updated in each iteration
            for i in 0..training_data.size {
                let feature_row = &training_data.data.feature_rows[i];
                //TODO for each iteration here in this loop we traverse the current tree being looked at for each data instance
                let (weighted_max_value, weighted_non_max_value, max_value_index) = {
                    let leaf_data = find_leaf_node_for_data(feature_row, boxed_tree_ref);
                    let max_value = leaf_data.max_value.unwrap();
                    let weighted_max_value = learning_rate * max_value;
                    let weighted_non_max_value = learning_rate
                        * calculate_approximate_value(max_value, number_of_classes_f64);
                    let max_value_index = leaf_data.class.unwrap();
                    (weighted_max_value, weighted_non_max_value, max_value_index)
                };

                //update label values below, first max class then rest
                let current_label = &mut training_data.mutable_labels[i];
                current_label[max_value_index] += weighted_max_value;
                for j in 0..number_of_classes {
                    if j != max_value_index {
                        current_label[j] += weighted_non_max_value;
                    }
                }
            }
        };
        trees.push(boxed_residual_tree);
    }
    trees
}

fn traverse_tree_to_create_map_of_index_to_leaf_output<'a>(
    node: &'a Box<TreeNode<AMGBoostLeaf>>,
    map_data_indices_to_weighted_leaf_output: &mut HashMap<usize, (f64, f64, usize)>,
    number_of_classes: f64,
) {
    if !node.is_leaf_node() {
        if node.true_branch.is_some() {
            traverse_tree_to_create_map_of_index_to_leaf_output(
                &node.true_branch.as_ref().unwrap(),
                map_data_indices_to_weighted_leaf_output,
                number_of_classes,
            );
        }
        if node.false_branch.is_some() {
            traverse_tree_to_create_map_of_index_to_leaf_output(
                &node.false_branch.as_ref().unwrap(),
                map_data_indices_to_weighted_leaf_output,
                number_of_classes,
            );
        }
    }

    let leaf = node.leaf.as_ref().unwrap();
    let max_value = leaf.max_value.unwrap();
    let weighted_max_value = 0.1 * max_value;
    let weighted_non_max_value = 0.1 * calculate_approximate_value(max_value, number_of_classes);
    let max_value_index = leaf.class.unwrap();
    let weighted_leaf_outputs = (weighted_max_value, weighted_non_max_value, max_value_index);

    let data_points_in_leaf = leaf.data.as_ref().unwrap();
    let data_indices_in_leaf: Vec<usize> = data_points_in_leaf.sorted_feature_columns[0]
        .iter()
        .map(|(_value, index)| *index)
        .collect();
    data_indices_in_leaf.iter().for_each(|index| {
        map_data_indices_to_weighted_leaf_output.insert(*index, weighted_leaf_outputs.clone());
    });
}
