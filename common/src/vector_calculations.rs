pub fn add_vectors(first: &[f64], second: &[f64]) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(&first_element, &second_element)| first_element + second_element)
        .collect()
}

pub fn subtract_vectors(first: &[f64], second: &[f64]) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(&first_element, &second_element)| first_element - second_element)
        .collect()
}

pub fn calculate_average_vector(vector_of_vectors: &Vec<Vec<f64>>) -> Vec<f64> {
    let inner_vector_length = vector_of_vectors[0].len();
    let mut average_vector = vec![0.; inner_vector_length];
    for i in 0..vector_of_vectors.len() {
        for j in 0..inner_vector_length {
            average_vector[j] += vector_of_vectors[i][j];
        }
    }
    for j in 0..inner_vector_length {
        average_vector[j] /= vector_of_vectors.len() as f64;
    }
    average_vector
}

pub fn mean_sum_of_squared_differences_between_vectors(prediction: &[f64], actual: &[f64]) -> f64 {
    assert_eq!(prediction.len(), actual.len());
    let squared_differences: Vec<f64> = prediction
        .iter()
        .zip(actual)
        .map(|(&prediction_element, &actual_element)| {
            let difference = prediction_element - actual_element;
            f64::powf(difference, 2.)
        })
        .collect();

    let sum_of_squared_differences: f64 = squared_differences.iter().sum();
    sum_of_squared_differences / prediction.len() as f64
}
