pub mod predict_helper_functions {
    use common::datasets::MultiTargetDataSet;

    pub fn get_binary_prediction(prediction: &[f64]) -> Vec<f64> {
        let mut max_index = 0;
        let mut max_seen = f64::NEG_INFINITY;
        prediction.iter().enumerate().for_each(|(index, element)| {
            if *element > max_seen {
                max_seen = *element;
                max_index = index;
            }
        });
        let mut binary_prediction_vector = vec![0.; prediction.len()];
        binary_prediction_vector[max_index] = 1.;
        binary_prediction_vector
    }

    pub fn calculate_accuracy_from_predictions(
        predictions: &[Vec<f64>],
        test_set: &MultiTargetDataSet,
    ) -> f64 {
        let mut correct_count = 0.;
        predictions
            .iter()
            .enumerate()
            .for_each(|(index, prediction)| {
                let actual = &test_set.labels[index];
                if prediction == actual {
                    correct_count += 1.;
                }
            });
        correct_count / predictions.len() as f64
    }
}

pub mod executor_helper_functions {
    use common::numerical_calculations::{
        multiply_f64_slice_by_f64_scalar, subtract_f64_slices_as_vector,
    };
    use rayon::prelude::*;

    use crate::boosting_ensemble::boosting_types::GradBoostTrainingData;

    pub fn calculate_residuals(training_data: &GradBoostTrainingData) -> Vec<Vec<f64>> {
        //iterate over training data and use 'true label' and current to calculate residuals
        let indices: Vec<_> = (0..training_data.size).collect();
        let residuals = indices
            .par_iter()
            .map(|i| {
                let true_label = &training_data.data.labels[*i];
                let current_label = &training_data.mutable_labels[*i];
                let probabilities = calculate_probabilities_of_predictions(current_label);
                let residual = subtract_f64_slices_as_vector(true_label, &probabilities);
                residual
            })
            .collect::<Vec<_>>();
        residuals
    }

    //current label ~ current prediction
    fn calculate_probabilities_of_predictions(current_label: &[f64]) -> Vec<f64> {
        let exponential_of_current_data_label: Vec<f64> =
            current_label.iter().map(|element| element.exp()).collect();
        let sum_of_exponentials_of_predictions: f64 =
            exponential_of_current_data_label.iter().sum();
        let probabilties = multiply_f64_slice_by_f64_scalar(
            1. / sum_of_exponentials_of_predictions,
            &exponential_of_current_data_label,
        );
        probabilties
    }
}
