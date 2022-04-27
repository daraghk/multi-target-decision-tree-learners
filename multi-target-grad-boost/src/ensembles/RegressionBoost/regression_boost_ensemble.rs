use common::{
    datasets::MultiTargetDataSet,
    numerical_calculations::mean_sum_of_squared_differences_between_f64_slices,
};
use multi_target_decision_tree::decision_trees::TreeConfig;

use self::regression_boost_executor_functions::execute_gradient_boosting_loop;

use super::{
    boosting_loop::boosting_loop,
    boosting_types::{BoostingEnsembleType, BoostingExecutor, RegressionBoostModel},
    common_boosting_functions::predict_common::predict_instance,
    GradientBoostedEnsemble,
};

mod regression_boost_executor_functions;

impl GradientBoostedEnsemble for RegressionBoostModel<'_> {
    fn train(
        data: MultiTargetDataSet,
        tree_config: TreeConfig,
        number_of_iterations: u32,
        learning_rate: f64,
    ) -> Self {
        let boosting_executor = BoostingExecutor {
            ensemble_type: BoostingEnsembleType::RegressionBoost,
            loop_executor_function: execute_gradient_boosting_loop,
        };
        let boosting_model = boosting_loop(
            data,
            tree_config,
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

fn calculate_mean_squared_error(
    test_data_labels: &Vec<Vec<f64>>,
    predictions: &Vec<Vec<f64>>,
) -> f64 {
    let mut total_error = 0.;
    let number_of_labels = test_data_labels.len();
    for i in 0..number_of_labels {
        total_error += mean_sum_of_squared_differences_between_f64_slices(
            &test_data_labels[i],
            &predictions[i],
        );
    }
    total_error / number_of_labels as f64
}
