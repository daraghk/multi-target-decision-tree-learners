use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{
        add_vectors, mean_sum_of_squared_differences_between_vectors, subtract_vectors,
    },
};

use crate::{grad_boost_ensemble::GradientBoostedEnsemble, tree_traverse::find_leaf_node_for_data};

pub fn calculate_mean_squared_error(
    test_data_labels: &Vec<Vec<f64>>,
    predictions: &Vec<Vec<f64>>,
) -> f64 {
    let mut total_error = 0.;
    let number_of_labels = test_data_labels.len();
    for i in 0..number_of_labels {
        total_error +=
            mean_sum_of_squared_differences_between_vectors(&test_data_labels[i], &predictions[i]);
    }
    total_error / number_of_labels as f64
}

pub fn predict_instance(
    test_feature_row: &[f64],
    grad_boost_ensemble: &GradientBoostedEnsemble,
    initial_guess: &[f64],
    learning_rate: f64,
) -> Vec<f64> {
    let test_instance_leaf_outputs = collect_leaf_outputs_for_test_instance(
        test_feature_row,
        grad_boost_ensemble,
        learning_rate,
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
    grad_boost_ensemble: &GradientBoostedEnsemble,
    learning_rate: f64,
) -> Vec<Vec<f64>> {
    let mut leaf_outputs = vec![];
    for i in 0..grad_boost_ensemble.trees.len() {
        let leaf = find_leaf_node_for_data(test_feature_row, &grad_boost_ensemble.trees[i]);
        let leaf_output = &*leaf.leaf_output.as_ref().unwrap();
        let weighted_leaf_output = leaf_output
            .into_iter()
            .map(|x| learning_rate * x)
            .collect::<Vec<_>>();
        leaf_outputs.push(weighted_leaf_output.clone());
    }
    leaf_outputs
}
