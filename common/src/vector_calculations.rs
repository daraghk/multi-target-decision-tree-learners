pub fn add_vectors(first: &Vec<f64>, second: &Vec<f64>) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    let mut result = vec![0.; first.len()];
    for i in 0..first.len() {
        result[i] = first[i] + second[i];
    }
    result
}

pub fn subtract_vectors(first: &Vec<f64>, second: &Vec<f64>) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    let mut result = vec![0.; first.len()];
    for i in 0..first.len() {
        result[i] = first[i] - second[i];
    }
    result
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

pub fn mean_sum_of_squared_differences_between_vectors(
    prediction: &Vec<f64>,
    actual: &Vec<f64>,
) -> f64 {
    let mut sum_of_squared_differences = 0.;
    for i in 0..prediction.len() {
        let error = prediction[i] - actual[i];
        sum_of_squared_differences += f64::powf(error, 2.);
    }
    sum_of_squared_differences / prediction.len() as f64
}
