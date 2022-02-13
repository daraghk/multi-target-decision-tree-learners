use common::datasets::MultiTargetDataSet;
use multi_target_decision_tree::decision_trees::{
    grad_boost_leaf_output::LeafOutputCalculator, TreeConfig,
};

#[path = "./MultiClassBoosting/AMGBoost/amg_boost_ensemble.rs"]
pub mod amg_boost_ensemble;
#[path = "./MultiClassBoosting/common_multi_class_boosting_functions.rs"]
pub mod common_multi_class_boosting_functions;
#[path = "./MultiClassBoosting/MultiClassBoost/multi_class_boost_ensemble.rs"]
pub mod multi_class_boost_ensemble;
#[path = "./RegressionBoost/regression_boost_ensemble.rs"]
pub mod regression_boost_ensemble;

pub mod boosting_loop;
pub mod boosting_types;
pub mod common_boosting_functions;

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
