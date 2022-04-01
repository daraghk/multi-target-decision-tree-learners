use common::vector_calculations::subtract_vectors;
use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    grad_boost_decision_trees::{
        grad_boost_leaf_output::{LeafOutputCalculator, LeafOutputType},
        GradBoostMultiTargetDecisionTree,
    },
    leaf::GradBoostLeaf,
    node::TreeNode,
};
use rayon::prelude::*;

use crate::boosting_ensemble::{
    boosting_types::GradBoostTrainingData,
    common_boosting_functions::update_common::update_dataset_labels,
};

pub(super) fn execute_gradient_boosting_loop(
    training_data: &mut GradBoostTrainingData,
    number_of_iterations: u32,
    tree_config: TreeConfig,
    learning_rate: f64,
) -> Vec<Box<TreeNode<GradBoostLeaf>>> {
    let mut trees = Vec::with_capacity(number_of_iterations as usize);
    let leaf_output_calculator = LeafOutputCalculator::new(LeafOutputType::Regression);
    for _i in 0..number_of_iterations {
        let residuals = calculate_residuals(training_data);
        let mut learner_data = training_data.data.clone();
        learner_data.labels = residuals;
        let residual_tree = GradBoostMultiTargetDecisionTree::new(
            learner_data,
            tree_config,
            leaf_output_calculator,
        );
        let boxed_residual_tree = Box::new(residual_tree.root);
        update_dataset_labels(training_data, &boxed_residual_tree, learning_rate);
        trees.push(boxed_residual_tree);
    }
    trees
}

fn calculate_residuals(training_data: &GradBoostTrainingData) -> Vec<Vec<f64>> {
    let indices: Vec<_> = (0..training_data.size).collect();
    let residuals = indices
        .par_iter()
        .map(|i| {
            let true_data_label = &training_data.data.labels[*i];
            let current_data_label = &training_data.mutable_labels[*i];
            let residual = subtract_vectors(true_data_label, current_data_label);
            residual
        })
        .collect::<Vec<_>>();
    residuals
}
