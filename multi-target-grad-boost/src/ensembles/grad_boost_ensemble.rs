use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{add_vectors, calculate_average_vector, subtract_vectors},
};
use multi_target_decision_tree::{
    decision_trees::{
        grad_boost_leaf_output::LeafOutputCalculator, GradBoostMultiTargetDecisionTree, TreeConfig,
    },
    leaf::GradBoostLeaf,
    node::TreeNode,
};

use crate::tree_traverse::find_leaf_node_for_data;

pub mod ensemble_multi_class;
pub mod ensemble_regression;

pub trait GradientBoostedEnsemble {
    fn train(
        data: MultiTargetDataSet,
        tree_config: TreeConfig,
        leaf_output_calculator: LeafOutputCalculator,
        number_of_iterations: u32,
        learning_rate: f64,
    ) -> Self;
    fn predict(&self, feature_row: &[f64]) -> Vec<f64>;
    fn calculate_all_predictions(&self, test_set: &MultiTargetDataSet) -> Vec<Vec<f64>>;
    fn calculate_score(&self, test_set: &MultiTargetDataSet) -> f64;
}

mod update_common {
    use super::*;

    pub fn update_dataset_labels_with_initial_guess(
        mutable_data: &mut MultiTargetDataSet,
        initial_guess: &Vec<f64>,
    ) {
        for i in 0..mutable_data.labels.len() {
            mutable_data.labels[i] = initial_guess.clone();
        }
    }

    pub fn update_dataset_labels(
        mutable_data: &mut MultiTargetDataSet,
        boxed_tree_ref: &Box<TreeNode<GradBoostLeaf>>,
        learning_rate: f64,
    ) {
        for i in 0..mutable_data.labels.len() {
            let leaf_data = find_leaf_node_for_data(&mutable_data.feature_rows[i], boxed_tree_ref);
            let leaf_output = leaf_data.leaf_output.as_ref().unwrap();
            let weighted_leaf_output = leaf_output
                .into_iter()
                .map(|x| learning_rate * x)
                .collect::<Vec<_>>();
            mutable_data.labels[i] = add_vectors(&mutable_data.labels[i], &weighted_leaf_output);
        }
    }
}

mod predict_common {
    use super::*;

    pub fn predict_instance(
        test_feature_row: &[f64],
        trees: &Vec<Box<TreeNode<GradBoostLeaf>>>,
        initial_guess: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        let test_instance_leaf_outputs =
            collect_leaf_outputs_for_test_instance(test_feature_row, trees, learning_rate);
        let mut sum_of_leaf_outputs = initial_guess.clone().to_owned();
        for leaf_output in test_instance_leaf_outputs {
            for i in 0..sum_of_leaf_outputs.len() {
                sum_of_leaf_outputs[i] += leaf_output[i];
            }
        }
        sum_of_leaf_outputs
    }

    fn collect_leaf_outputs_for_test_instance(
        test_feature_row: &[f64],
        trees: &Vec<Box<TreeNode<GradBoostLeaf>>>,
        learning_rate: f64,
    ) -> Vec<Vec<f64>> {
        let mut leaf_outputs = vec![];
        for i in 0..trees.len() {
            let leaf = find_leaf_node_for_data(test_feature_row, &trees[i]);
            let leaf_output = &*leaf.leaf_output.as_ref().unwrap();
            let weighted_leaf_output = leaf_output
                .into_iter()
                .map(|x| learning_rate * x)
                .collect::<Vec<_>>();
            leaf_outputs.push(weighted_leaf_output.clone());
        }
        leaf_outputs
    }
}
