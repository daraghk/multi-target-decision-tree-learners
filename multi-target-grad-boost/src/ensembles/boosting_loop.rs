use common::{datasets::MultiTargetDataSet, vector_calculations::calculate_average_vector};
use multi_target_decision_tree::{
    decision_trees::{grad_boost_leaf_output::LeafOutputCalculator, TreeConfig},
    leaf::Leaf,
};

use super::{
    boosting_types::{
        BoostingEnsembleType, BoostingExecutor, BoostingResult, GradBoostTrainingData,
    },
    common_boosting_functions::update_common::update_dataset_labels_with_initial_guess,
};

pub fn boosting_loop<T: Leaf>(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    leaf_output_calculator: LeafOutputCalculator,
    number_of_iterations: u32,
    learning_rate: f64,
    boosting_executor: BoostingExecutor<T>,
) -> BoostingResult<T> {
    let mutable_labels = data.labels.clone();
    let training_data_size = data.labels.len();
    let mut training_data = GradBoostTrainingData {
        data,
        mutable_labels,
        size: training_data_size,
    };
    let initial_guess = determine_initial_guess(&training_data, boosting_executor.ensemble_type);
    update_dataset_labels_with_initial_guess(&mut training_data.mutable_labels, &initial_guess);
    let trees = (boosting_executor.loop_executor_function)(
        &mut training_data,
        number_of_iterations,
        tree_config,
        leaf_output_calculator,
        learning_rate,
    );
    BoostingResult {
        trees,
        initial_guess,
        learning_rate,
    }
}

fn determine_initial_guess(
    training_data: &GradBoostTrainingData,
    ensemble_type: BoostingEnsembleType,
) -> Vec<f64> {
    let number_of_classes = training_data.mutable_labels[0].len() as f64;
    let initial_guess = match ensemble_type {
        BoostingEnsembleType::AMGBoost => vec![1. / number_of_classes; number_of_classes as usize],
        BoostingEnsembleType::MultiClassBoost => {
            vec![1. / number_of_classes; number_of_classes as usize]
        }
        BoostingEnsembleType::RegressionBoost => {
            calculate_average_vector(&training_data.data.labels)
        }
    };
    initial_guess
}
