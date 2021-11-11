pub fn add_vectors(first: &Vec<f32>, second: &Vec<f32>) -> Vec<f32> {
    assert_eq!(first.len(), second.len());
    let mut result = vec![0.; first.len()];
    for i in 0..first.len() {
        result[i] = first[i] + second[i];
    }
    result
}

pub fn subtract_vectors(first: &Vec<f32>, second: &Vec<f32>) -> Vec<f32> {
    assert_eq!(first.len(), second.len());
    let mut result = vec![0.; first.len()];
    for i in 0..first.len() {
        result[i] = first[i] - second[i];
    }
    result
}

pub fn calculate_average_vector(vector_of_vectors: &Vec<Vec<f32>>) -> Vec<f32> {
    let inner_vector_length = vector_of_vectors[0].len();
    let mut average_vector = vec![0.; inner_vector_length];
    for i in 0..vector_of_vectors.len() {
        for j in 0..inner_vector_length {
            average_vector[j] += vector_of_vectors[i][j];
        }
    }
    for j in 0..inner_vector_length {
        average_vector[j] /= vector_of_vectors.len() as f32;
    }
    average_vector
}
