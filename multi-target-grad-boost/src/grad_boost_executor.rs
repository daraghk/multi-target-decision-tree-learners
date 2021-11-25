use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{add_vectors, subtract_vectors},
};
use multi_target_decision_tree::{
    decision_trees::{GradBoostMultiTargetDecisionTree, TreeConfig},
    leaf::GradBoostLeaf,
    node::TreeNode,
};

use crate::tree_traverse::find_leaf_node_for_data;

pub fn update_dataset_labels_with_initial_guess(
    mutable_data: &mut MultiTargetDataSet,
    initial_guess: &Vec<f64>,
) {
    for i in 0..mutable_data.labels.len() {
        mutable_data.labels[i] = initial_guess.clone();
    }
}

pub fn execute_gradient_boosting_loop(
    true_data: MultiTargetDataSet,
    mutable_data: &mut MultiTargetDataSet,
    number_of_iterations: u32,
    tree_config: TreeConfig,
    learning_rate: f64,
) -> Vec<Box<TreeNode<GradBoostLeaf>>> {
    let mut trees = vec![];
    for _i in 0..number_of_iterations {
        let residuals = calculate_residuals(&true_data, mutable_data);
        let mut learner_data = mutable_data.clone();
        learner_data.labels = residuals;
        let residual_tree = GradBoostMultiTargetDecisionTree::new(learner_data, tree_config);
        let boxed_residual_tree = Box::new(residual_tree.root);
        update_dataset_labels(mutable_data, &boxed_residual_tree, learning_rate);
        trees.push(boxed_residual_tree);
    }
    trees
}

fn calculate_residuals(
    true_data: &MultiTargetDataSet,
    mutable_data: &MultiTargetDataSet,
) -> Vec<Vec<f64>> {
    let label_length = mutable_data.labels[0].len();
    let mut residuals = vec![];
    for i in 0..mutable_data.labels.len() {
        let original_index = mutable_data.indices[i];
        let true_data_label = &true_data.labels[original_index];
        let current_data_label = &mutable_data.labels[i];
        let residual = subtract_vectors(true_data_label, current_data_label);
        residuals.push(residual);
    }
    residuals
}

fn update_dataset_labels(
    mutable_data: &mut MultiTargetDataSet,
    boxed_tree_ref: &Box<TreeNode<GradBoostLeaf>>,
    learning_rate: f64,
) {
    for i in 0..mutable_data.labels.len() {
        let leaf_data = find_leaf_node_for_data(&mutable_data.features[i], boxed_tree_ref);
        let leaf_output = leaf_data.leaf_output.as_ref().unwrap();
        let weighted_leaf_output = leaf_output
            .into_iter()
            .map(|x| learning_rate * x)
            .collect::<Vec<_>>();
        mutable_data.labels[i] = add_vectors(&mutable_data.labels[i], &weighted_leaf_output);
    }
}
