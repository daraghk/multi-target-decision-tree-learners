use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    grad_boost_decision_trees::{
        grad_boost_leaf_output::{LeafOutputCalculator, LeafOutputType},
        GradBoostMultiTargetDecisionTree,
    },
    leaf::GradBoostLeaf,
    node::TreeNode,
};

use crate::boosting_ensemble::{
    boosting_types::GradBoostTrainingData,
    common_boosting_functions::update_common::update_dataset_labels,
    common_multi_class_boosting_functions::executor_helper_functions::calculate_residuals,
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
