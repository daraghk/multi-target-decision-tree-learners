use common::{datasets::MultiTargetDataSet, vector_calculations::add_vectors};
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
    number_of_iterations: u8,
    tree_config: TreeConfig,
    learning_rate: f64,
) -> Vec<Box<TreeNode<GradBoostLeaf>>> {
    let mut trees = vec![];
    for _i in 0..number_of_iterations {
        let dataset_to_grow_tree = mutable_data.clone();
        let decision_tree =
            GradBoostMultiTargetDecisionTree::new(&true_data, dataset_to_grow_tree, tree_config);
        let boxed_tree = Box::new(decision_tree.root);
        update_dataset_labels(&true_data, mutable_data, &boxed_tree, learning_rate);
        println!("{:?}", mutable_data.labels[10]);
        trees.push(boxed_tree);
    }
    trees
}

fn update_dataset_labels(
    true_data: &MultiTargetDataSet,
    mutable_data: &mut MultiTargetDataSet,
    boxed_tree_ref: &Box<TreeNode<GradBoostLeaf>>,
    learning_rate: f64,
) {
    for i in 0..mutable_data.labels.len() {
        let leaf_data = find_leaf_node_for_data(&mutable_data.features[i], boxed_tree_ref);
        let leaf_output = leaf_data.leaf_output.as_ref().unwrap();
        let leaf_output = leaf_output
            .into_iter()
            .map(|x| learning_rate * x)
            .collect::<Vec<_>>();
        mutable_data.labels[i] = add_vectors(&mutable_data.labels[i], &leaf_output);
    }
}
