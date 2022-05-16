use crate::boosting_ensemble::common_multi_class_boosting_functions::executor_helper_functions::calculate_residuals;

use common::datasets::MultiTargetDataSetSortedFeatures;
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

pub(crate) fn execute_gradient_boosting_loop(
    training_data: &mut GradBoostTrainingData,
    number_of_iterations: u32,
    tree_config: TreeConfig,
    learning_rate: f64,
) -> Vec<Box<TreeNode<AMGBoostLeaf>>> {
    let mut trees = Vec::with_capacity(number_of_iterations as usize);
    let leaf_output_calculator =
        LeafOutputCalculator::new(LeafOutputType::MultiClassClassification);
    let number_of_classes = training_data.mutable_labels[0].len() as f64;
    let mut current_residual_learner_data = MultiTargetDataSetSortedFeatures {
        sorted_feature_columns: training_data.data.sorted_feature_columns.clone(),
        labels: training_data.data.labels.clone(),
    };
    for _i in 0..number_of_iterations {
        let residuals = calculate_residuals(training_data);
        current_residual_learner_data.labels = residuals;
        let residual_tree = AMGBoostTree::new(
            &current_residual_learner_data,
            tree_config,
            leaf_output_calculator,
        );
        let boxed_residual_tree = Box::new(residual_tree.root);
        let mut map_data_indices_to_weighted_leaf_output = vec![(0., 0., 0); training_data.size];
        traverse_tree_to_create_map_of_index_to_leaf_output(
            &boxed_residual_tree,
            &mut map_data_indices_to_weighted_leaf_output,
            number_of_classes,
            learning_rate,
        );

        for i in 0..training_data.size {
            let weighted_leaf_output = &map_data_indices_to_weighted_leaf_output[i];
            let current_label = &mut training_data.mutable_labels[i];
            let max_value_index = weighted_leaf_output.2;
            let weighted_max_value = weighted_leaf_output.0;
            let weighted_non_max_value = weighted_leaf_output.1;
            current_label[max_value_index] += weighted_max_value;
            for j in 0..number_of_classes as usize {
                if j != max_value_index {
                    current_label[j] += weighted_non_max_value;
                }
            }
        }
        trees.push(boxed_residual_tree);
    }
    trees
}

fn traverse_tree_to_create_map_of_index_to_leaf_output(
    node: &Box<TreeNode<AMGBoostLeaf>>,
    map_data_indices_to_weighted_leaf_output: &mut Vec<(f64, f64, usize)>,
    number_of_classes: f64,
    learning_rate: f64,
) {
    if node.is_leaf_node() {
        let leaf = node.leaf.as_ref().unwrap();
        let max_value = leaf.max_value.unwrap();
        let weighted_max_value = 0.1 * max_value;
        let weighted_non_max_value =
            0.1 * calculate_approximate_value(max_value, number_of_classes);
        let max_value_index = leaf.class.unwrap();
        let weighted_leaf_outputs = (weighted_max_value, weighted_non_max_value, max_value_index);

        let data_indices_in_leaf = &leaf.data_indices;
        data_indices_in_leaf.iter().for_each(|index| {
            map_data_indices_to_weighted_leaf_output[*index] = weighted_leaf_outputs;
        });
    } else {
        if node.true_branch.is_some() {
            traverse_tree_to_create_map_of_index_to_leaf_output(
                node.true_branch.as_ref().unwrap(),
                map_data_indices_to_weighted_leaf_output,
                number_of_classes,
                learning_rate,
            );
        }
        if node.false_branch.is_some() {
            traverse_tree_to_create_map_of_index_to_leaf_output(
                node.false_branch.as_ref().unwrap(),
                map_data_indices_to_weighted_leaf_output,
                number_of_classes,
                learning_rate,
            );
        }
    }
}
