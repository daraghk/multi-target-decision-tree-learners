use crate::boosting_ensemble::boosting_loop::boosting_loop;
use common::datasets::MultiTargetDataSet;
use multi_target_decision_tree::decision_trees::{
    grad_boost_leaf_output::LeafOutputCalculator, TreeConfig,
};

use self::multi_class_boost_executor_functions::execute_gradient_boosting_loop;

use super::{
    boosting_types::{BoostingEnsembleType, BoostingExecutor, MultiClassBoostModel},
    common_boosting_functions::predict_common::predict_instance,
    common_multi_class_boosting_functions::predict_helper_functions::{
        calculate_accuracy_from_predictions, get_binary_prediction,
    },
    GradientBoostedEnsemble,
};

pub mod multi_class_boost_executor_functions;

impl GradientBoostedEnsemble for MultiClassBoostModel {
    fn train(
        data: MultiTargetDataSet,
        tree_config: TreeConfig,
        leaf_output_calculator: LeafOutputCalculator,
        number_of_iterations: u32,
        learning_rate: f64,
    ) -> Self {
        let boosting_executor = BoostingExecutor {
            ensemble_type: BoostingEnsembleType::MultiClassBoost,
            loop_executor_function: execute_gradient_boosting_loop,
        };
        let boosting_model = boosting_loop(
            data,
            tree_config,
            leaf_output_calculator,
            number_of_iterations,
            learning_rate,
            boosting_executor,
        );
        Self {
            trees: boosting_model.trees,
            initial_guess: boosting_model.initial_guess,
            learning_rate: boosting_model.learning_rate,
        }
    }

    fn predict(&self, feature_row: &[f64]) -> Vec<f64> {
        let prediction = predict_instance(
            feature_row,
            &self.trees,
            &self.initial_guess,
            self.learning_rate,
        );
        get_binary_prediction(&prediction)
    }

    fn calculate_all_predictions(&self, test_set: &MultiTargetDataSet) -> Vec<Vec<f64>> {
        let number_of_test_instances = test_set.feature_rows.len();
        let mut predictions = Vec::with_capacity(number_of_test_instances);
        for i in 0..number_of_test_instances {
            let test_feature_row = &test_set.feature_rows[i];
            let prediction = self.predict(test_feature_row);
            predictions.push(prediction);
        }
        predictions
    }

    fn calculate_score(&self, test_set: &MultiTargetDataSet) -> f64 {
        let predictions = self.calculate_all_predictions(test_set);
        calculate_accuracy_from_predictions(&predictions, test_set)
    }
}
