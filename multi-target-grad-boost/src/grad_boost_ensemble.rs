use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{add_vectors, calculate_average_vector, subtract_vectors},
};
use multi_target_decision_tree::{
    decision_trees::{GradBoostMultiTargetDecisionTree, TreeConfig},
    leaf::GradBoostLeaf,
    node::TreeNode,
};

use crate::{
    grad_boost_executor::{
        execute_gradient_boosting_loop, update_dataset_labels_with_initial_guess,
    },
    grad_boost_predict::{calculate_mean_squared_error, predict_instance},
};

pub struct GradientBoostedEnsemble {
    pub trees: Vec<Box<TreeNode<GradBoostLeaf>>>,
    initial_guess: Vec<f64>,
    learning_rate: f64,
}

impl GradientBoostedEnsemble {
    pub fn train(
        data: MultiTargetDataSet,
        tree_config: TreeConfig,
        number_of_iterations: u32,
        learning_rate: f64,
    ) -> GradientBoostedEnsemble {
        let mut mutable_data = data.clone();
        let initial_guess = calculate_average_vector(&mutable_data.labels);
        update_dataset_labels_with_initial_guess(&mut mutable_data, &initial_guess);
        let trees = execute_gradient_boosting_loop(
            data,
            &mut mutable_data,
            number_of_iterations,
            tree_config,
            learning_rate,
        );
        Self {
            trees,
            initial_guess,
            learning_rate,
        }
    }

    pub fn predict(&self, feature_row: &[f64]) -> Vec<f64> {
        let result = predict_instance(feature_row, self, &self.initial_guess, self.learning_rate);
        result
    }

    pub fn calculate_all_predictions(&self, test_set: &MultiTargetDataSet) -> Vec<Vec<f64>> {
        let number_of_test_instances = test_set.features.len();
        let mut predictions = Vec::with_capacity(number_of_test_instances);
        for i in 0..number_of_test_instances {
            let test_feature_row = &test_set.features[i];
            let test_label_original = &test_set.labels[i];
            let prediction = self.predict(test_feature_row);
            predictions.push(prediction);
        }
        predictions
    }

    pub fn calculate_error(&self, test_set: &MultiTargetDataSet) -> f64 {
        let predictions = self.calculate_all_predictions(test_set);
        let error = calculate_mean_squared_error(&test_set.labels, &predictions);
        error
    }
}
