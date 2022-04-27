use std::collections::HashMap;

use common::{
    data_processor::create_dataset_with_sorted_features,
    numerical_calculations::add_f64_slices_as_vector,
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

use crate::{
    boosting_ensemble::{
        boosting_types::GradBoostTrainingData,
        common_boosting_functions::update_common::update_dataset_labels,
        common_multi_class_boosting_functions::executor_helper_functions::calculate_residuals,
    },
    tree_traverse::find_leaf_node_for_data,
};

pub(crate) fn execute_gradient_boosting_loop(
    training_data: &mut GradBoostTrainingData,
    number_of_iterations: u32,
    tree_config: TreeConfig,
    learning_rate: f64,
) -> Vec<(Box<TreeNode<GradBoostLeaf>>, Vec<Vec<f64>>)> {
    let mut trees_and_residuals = Vec::with_capacity(number_of_iterations as usize);
    let leaf_output_calculator =
        LeafOutputCalculator::new(LeafOutputType::MultiClassClassification);
    let processed_data = create_dataset_with_sorted_features(&training_data.data);
    for _i in 0..number_of_iterations {
        //TODO how to remove / make faster?
        //TODO how to get rid of clone - processed_data never changes - at least the feature cols don't
        let mut learner_data = processed_data.clone();
        let residuals = calculate_residuals(training_data);
        learner_data.labels = residuals.iter().map(|label| label).collect();

        let residual_tree = GradBoostMultiTargetDecisionTree::new(
            learner_data,
            tree_config,
            leaf_output_calculator,
        );
        let boxed_residual_tree = Box::new(residual_tree.root);
        let mut map_data_indices_to_weighted_leaf_output = HashMap::new();
        traverse_tree_to_create_map_of_index_to_leaf_output(
            &boxed_residual_tree,
            &mut map_data_indices_to_weighted_leaf_output,
        );
        {
            for i in 0..training_data.size {
                let weighted_leaf_output =
                    map_data_indices_to_weighted_leaf_output.get(&i).unwrap();
                training_data.mutable_labels[i] = add_f64_slices_as_vector(
                    &training_data.mutable_labels[i],
                    &weighted_leaf_output,
                );
            }
        };
        trees_and_residuals.push((boxed_residual_tree, residuals));
    }
    trees_and_residuals
}

fn traverse_tree_to_create_map_of_index_to_leaf_output<'a>(
    node: &'a Box<TreeNode<GradBoostLeaf>>,
    map_data_indices_to_weighted_leaf_output: &mut HashMap<usize, Vec<f64>>,
) {
    if !node.is_leaf_node() {
        if node.true_branch.is_some() {
            traverse_tree_to_create_map_of_index_to_leaf_output(
                &node.true_branch.as_ref().unwrap(),
                map_data_indices_to_weighted_leaf_output,
            );
        }
        if node.false_branch.is_some() {
            traverse_tree_to_create_map_of_index_to_leaf_output(
                &node.false_branch.as_ref().unwrap(),
                map_data_indices_to_weighted_leaf_output,
            );
        }
    }

    let leaf = node.leaf.as_ref().unwrap();
    let leaf_output = leaf.leaf_output.as_ref().unwrap();
    let weighted_leaf_output = leaf_output.into_iter().map(|x| 0.1 * x).collect::<Vec<_>>();
    let data_points_in_leaf = leaf.data.as_ref().unwrap();
    let data_indices_in_leaf: Vec<usize> = data_points_in_leaf.sorted_feature_columns[0]
        .iter()
        .map(|(_value, index)| *index)
        .collect();
    data_indices_in_leaf.iter().for_each(|index| {
        map_data_indices_to_weighted_leaf_output.insert(*index, weighted_leaf_output.clone());
    });
}
