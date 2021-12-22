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

pub fn sum_of_vectors(vector_of_vectors: &Vec<Vec<f64>>) -> Vec<f64> {
    let length_of_inner_vectors = vector_of_vectors[0].len();
    let mut sum_vector = vec![0.; length_of_inner_vectors];
    vector_of_vectors.iter().for_each(|inner_vector| {
        sum_vector = add_vectors(&sum_vector, &inner_vector);
    });
    sum_vector
}

pub fn multiply_vectors(first: &[f64], second: &[f64]) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(&first_element, &second_element)| first_element * second_element)
        .collect()
}

pub fn divide_vectors(first: &[f64], second: &[f64]) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(&first_element, &second_element)| first_element / second_element)
        .collect()
}

pub fn multiply_vector_by_scalar(scalar: f64, vector: &[f64]) -> Vec<f64> {
    vector.iter().map(|element| element * scalar).collect()
}

#[cfg(test)]
mod tests {
    use super::sum_of_vectors;

    #[test]
    fn test_sum_of_vectors() {
        let vector_of_vectors = vec![vec![1., 2., 3.], vec![1., 2., 3.]];
        let result = sum_of_vectors(&vector_of_vectors);
        println!("{:?}", result);
        assert_eq!(result, vec![2., 4., 6.]);
    }
}
