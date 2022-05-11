use common::{
    data_processor::create_dataset_with_sorted_features,
    datasets::MultiTargetDataSetSortedFeatures,
    numerical_calculations::{add_f64_slices_mutating},
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
        common_multi_class_boosting_functions::executor_helper_functions::calculate_residuals,
    },
};

pub(crate) fn execute_gradient_boosting_loop(
    training_data: &mut GradBoostTrainingData,
    number_of_iterations: u32,
    tree_config: TreeConfig,
    learning_rate: f64,
) -> Vec<Box<TreeNode<GradBoostLeaf>>> {
    let mut trees = Vec::with_capacity(number_of_iterations as usize);
    let leaf_output_calculator =
        LeafOutputCalculator::new(LeafOutputType::MultiClassClassification);
    let processed_data = create_dataset_with_sorted_features(&training_data.data);
    for _i in 0..number_of_iterations {
        let residuals = calculate_residuals(training_data);
        let learner_data = MultiTargetDataSetSortedFeatures {
            sorted_feature_columns: processed_data.sorted_feature_columns.clone(),
            labels: residuals,
        };

        let residual_tree = GradBoostMultiTargetDecisionTree::new(
            learner_data,
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
            // println!("{:?} {:?}", i, weighted_leaf_output);
            add_f64_slices_mutating(&mut training_data.mutable_labels[i], weighted_leaf_output);
        }
        trees.push(boxed_residual_tree);
    }
    trees
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