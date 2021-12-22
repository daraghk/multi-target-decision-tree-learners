use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{
        calculate_average_vector, mean_sum_of_squared_differences_between_vectors, subtract_vectors,
    },
};
use multi_target_decision_tree::{
    decision_trees::{
        grad_boost_leaf_output::LeafOutputCalculator, GradBoostMultiTargetDecisionTree, TreeConfig,
    },
    leaf::GradBoostLeaf,
    node::TreeNode,
};

use super::{
    update_common::{update_dataset_labels, update_dataset_labels_with_initial_guess},
    GradientBoostedEnsemble,
};

use crate::grad_boost_ensemble::predict_common::predict_instance;

pub struct GradientBoostedEnsembleRegression {
    pub trees: Vec<Box<TreeNode<GradBoostLeaf>>>,
    initial_guess: Vec<f64>,
    learning_rate: f64,
}

impl GradientBoostedEnsemble for GradientBoostedEnsembleRegression {
    fn train(
        data: MultiTargetDataSet,
        tree_config: TreeConfig,
        leaf_output_calculator: LeafOutputCalculator,
        number_of_iterations: u32,
        learning_rate: f64,
    ) -> GradientBoostedEnsembleRegression {
        let mut mutable_data = data.clone();
        let initial_guess = calculate_average_vector(&mutable_data.labels);
        let number_of_classes = mutable_data.labels[0].len() as f64;
        update_dataset_labels_with_initial_guess(&mut mutable_data, &initial_guess);
        let trees = execute_gradient_boosting_loop(
            data,
            &mut mutable_data,
            number_of_iterations,
            tree_config,
            leaf_output_calculator,
            learning_rate,
        );
        Self {
            trees,
            initial_guess,
            learning_rate,
        }
    }

    fn predict(&self, feature_row: &[f64]) -> Vec<f64> {
        let result = predict_instance(
            feature_row,
            &self.trees,
            &self.initial_guess,
            self.learning_rate,
        );
        result
    }

    fn calculate_all_predictions(&self, test_set: &MultiTargetDataSet) -> Vec<Vec<f64>> {
        let number_of_test_instances = test_set.feature_rows.len();
        let mut predictions = Vec::with_capacity(number_of_test_instances);
        for i in 0..number_of_test_instances {
            let test_feature_row = &test_set.feature_rows[i];
            let test_label_original = &test_set.labels[i];
            let prediction = self.predict(test_feature_row);
            predictions.push(prediction);
        }
        predictions
    }

    fn calculate_score(&self, test_set: &MultiTargetDataSet) -> f64 {
        let predictions = self.calculate_all_predictions(test_set);
        let error = calculate_mean_squared_error(&test_set.labels, &predictions);
        error
    }
}

fn execute_gradient_boosting_loop(
    true_data: MultiTargetDataSet,
    mutable_data: &mut MultiTargetDataSet,
    number_of_iterations: u32,
    tree_config: TreeConfig,
    leaf_output_calculator: LeafOutputCalculator,
    learning_rate: f64,
) -> Vec<Box<TreeNode<GradBoostLeaf>>> {
    let mut trees = Vec::with_capacity(number_of_iterations as usize);
    for _i in 0..number_of_iterations {
        let residuals = calculate_residuals(&true_data, mutable_data);
        let mut learner_data = mutable_data.clone();
        learner_data.labels = residuals;
        let residual_tree = GradBoostMultiTargetDecisionTree::new(
            learner_data,
            tree_config,
            leaf_output_calculator,
        );
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

fn calculate_mean_squared_error(
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
