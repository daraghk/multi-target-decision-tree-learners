use common::{
    datasets::MultiTargetDataSetSortedFeatures,
    numerical_calculations::{add_f64_slices_mutating, subtract_f64_slices_as_vector},
};
use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    grad_boost_decision_trees::{
        grad_boost_leaf_output::{LeafOutputCalculator, LeafOutputType},
        GradBoostMultiTargetDecisionTree,
    },
    leaf::GradBoostLeaf,
    node::TreeNode,
};

use crate::boosting_ensemble::boosting_types::GradBoostTrainingData;

pub(super) fn execute_gradient_boosting_loop<'a>(
    training_data: &mut GradBoostTrainingData,
    number_of_iterations: u32,
    tree_config: TreeConfig,
    learning_rate: f64,
) -> Vec<Box<TreeNode<GradBoostLeaf>>> {
    let mut trees = Vec::with_capacity(number_of_iterations as usize);
    let leaf_output_calculator = LeafOutputCalculator::new(LeafOutputType::Regression);
    let mut current_residual_learner_data = MultiTargetDataSetSortedFeatures {
        sorted_feature_columns: training_data.data.sorted_feature_columns.clone(),
        labels: training_data.data.labels.clone(),
    };
    for _i in 0..number_of_iterations {
        let residuals = calculate_residuals(training_data);
        current_residual_learner_data.labels = residuals;
        let residual_tree = GradBoostMultiTargetDecisionTree::new(
            &current_residual_learner_data,
            tree_config,
            leaf_output_calculator,
        );
        let boxed_residual_tree = Box::new(residual_tree.root);
        let mut map_data_indices_to_weighted_leaf_output = vec![vec![]; training_data.size];
        traverse_tree_to_create_map_of_index_to_leaf_output(
            &boxed_residual_tree,
            &mut map_data_indices_to_weighted_leaf_output,
            learning_rate,
        );

        for i in 0..training_data.size {
            let weighted_leaf_output = &map_data_indices_to_weighted_leaf_output[i];
            add_f64_slices_mutating(&mut training_data.mutable_labels[i], weighted_leaf_output);
        }
        trees.push(boxed_residual_tree);
    }
    trees
}

fn calculate_residuals(training_data: &GradBoostTrainingData) -> Vec<Vec<f64>> {
    let mut residuals = Vec::with_capacity(training_data.size);
    (0..training_data.size).into_iter().for_each(|i| {
        let true_data_label = &training_data.data.labels[i];
        let current_data_label = &training_data.mutable_labels[i];
        let residual = subtract_f64_slices_as_vector(true_data_label, current_data_label);
        residuals.push(residual);
    });
    residuals
}

fn traverse_tree_to_create_map_of_index_to_leaf_output(
    node: &Box<TreeNode<GradBoostLeaf>>,
    map_data_indices_to_weighted_leaf_output: &mut Vec<Vec<f64>>,
    learning_rate: f64,
) {
    if node.is_leaf_node() {
        let leaf = node.leaf.as_ref().unwrap();
        let leaf_output = leaf.leaf_output.as_ref().unwrap();
        let weighted_leaf_output = leaf_output
            .into_iter()
            .map(|x| learning_rate * x)
            .collect::<Vec<_>>();
        let data_indices_in_leaf = &leaf.data_indices;
        data_indices_in_leaf.iter().for_each(|index| {
            map_data_indices_to_weighted_leaf_output[*index] = weighted_leaf_output.clone();
        });
    } else {
        if node.true_branch.is_some() {
            traverse_tree_to_create_map_of_index_to_leaf_output(
                &node.true_branch.as_ref().unwrap(),
                map_data_indices_to_weighted_leaf_output,
                learning_rate,
            );
        }
        if node.false_branch.is_some() {
            traverse_tree_to_create_map_of_index_to_leaf_output(
                &node.false_branch.as_ref().unwrap(),
                map_data_indices_to_weighted_leaf_output,
                learning_rate,
            );
        }
    }
}
