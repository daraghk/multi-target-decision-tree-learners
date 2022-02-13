use multi_target_decision_tree::{leaf::AMGBoostLeaf, node::TreeNode};

use crate::tree_traverse::find_leaf_node_for_data;

use super::calculate_approximate_value;

pub fn predict_instance(
    test_feature_row: &[f64],
    trees: &Vec<Box<TreeNode<AMGBoostLeaf>>>,
    initial_guess: &[f64],
    learning_rate: f64,
    number_of_classes: usize,
) -> Vec<f64> {
    let test_instance_leaf_outputs = collect_leaf_outputs_for_test_instance(
        test_feature_row,
        trees,
        learning_rate,
        number_of_classes,
    );
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
    trees: &Vec<Box<TreeNode<AMGBoostLeaf>>>,
    learning_rate: f64,
    number_of_classes: usize,
) -> Vec<Vec<f64>> {
    let mut leaf_outputs = vec![];
    for i in 0..trees.len() {
        let leaf = find_leaf_node_for_data(test_feature_row, &trees[i]);
        let leaf_output = construct_approximate_leaf_output(leaf, number_of_classes);
        let weighted_leaf_output = leaf_output
            .into_iter()
            .map(|x| learning_rate * x)
            .collect::<Vec<_>>();
        leaf_outputs.push(weighted_leaf_output.clone());
    }
    leaf_outputs
}

fn construct_approximate_leaf_output(
    leaf_data: &AMGBoostLeaf,
    number_of_classes: usize,
) -> Vec<f64> {
    let max_value = leaf_data.max_value.unwrap();
    let max_value_class = leaf_data.class.unwrap();
    let non_max_value = calculate_approximate_value(max_value, number_of_classes as f64);
    let mut approximated_leaf_output = vec![non_max_value; number_of_classes as usize];
    approximated_leaf_output[max_value_class] = max_value;
    approximated_leaf_output
}
