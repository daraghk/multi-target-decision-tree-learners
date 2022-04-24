pub fn add_f64_slices_as_vector(first: &[f64], second: &[f64]) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(&first_element, &second_element)| first_element + second_element)
        .collect()
}

pub fn subtract_f64_slices_as_vector(first: &[f64], second: &[f64]) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(&first_element, &second_element)| first_element - second_element)
        .collect()
}

pub fn calculate_average_f64_vector(vector_of_vectors: &Vec<Vec<f64>>) -> Vec<f64> {
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

pub fn calculate_average_f64_vector_from_refs(vector_of_vectors: &Vec<&Vec<f64>>) -> Vec<f64> {
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

pub fn mean_sum_of_squared_differences_between_f64_slices(first: &[f64], second: &[f64]) -> f64 {
    assert_eq!(first.len(), second.len());
    let squared_differences: Vec<f64> = first
        .iter()
        .zip(second)
        .map(|(&prediction_element, &actual_element)| {
            let difference = prediction_element - actual_element;
            f64::powf(difference, 2.)
        })
        .collect();

    let sum_of_squared_differences: f64 = squared_differences.iter().sum();
    sum_of_squared_differences / first.len() as f64
}

pub fn sum_of_f64_vectors(vector_of_vectors: &Vec<Vec<f64>>) -> Vec<f64> {
    let length_of_inner_vectors = vector_of_vectors[0].len();
    let mut sum_vector = vec![0.; length_of_inner_vectors];
    vector_of_vectors.iter().for_each(|inner_vector| {
        sum_vector = add_f64_slices_as_vector(&sum_vector, &inner_vector);
    });
    sum_vector
}

pub fn sum_of_f64_vectors_from_refs(vector_of_vectors: &Vec<&Vec<f64>>) -> Vec<f64> {
    let length_of_inner_vectors = vector_of_vectors[0].len();
    let mut sum_vector = vec![0.; length_of_inner_vectors];
    vector_of_vectors.iter().for_each(|inner_vector| {
        sum_vector = add_f64_slices_as_vector(&sum_vector, &inner_vector);
    });
    sum_vector
}

pub fn multiply_f64_slices_as_vector(first: &[f64], second: &[f64]) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(&first_element, &second_element)| first_element * second_element)
        .collect()
}

pub fn divide_f64_slices_as_vector(first: &[f64], second: &[f64]) -> Vec<f64> {
    assert_eq!(first.len(), second.len());
    first
        .iter()
        .zip(second)
        .map(|(&first_element, &second_element)| first_element / second_element)
        .collect()
}

pub fn multiply_f64_slice_by_f64_scalar(scalar: f64, vector: &[f64]) -> Vec<f64> {
    vector.iter().map(|element| element * scalar).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_math_functions() {
        let vec_one = vec![1., 2., 3.];
        let vec_two = vec![1., 2., 3.];
        let sum_of_vecs = add_f64_slices_as_vector(&vec_one, &vec_two);
        assert_eq!(sum_of_vecs, [2., 4., 6.]);
        assert_eq!(sum_of_vecs, multiply_f64_slice_by_f64_scalar(2., &vec_one));
        assert_eq!(
            divide_f64_slices_as_vector(&vec_one, &vec_two),
            vec![1., 1., 1.]
        );
        assert_eq!(
            multiply_f64_slices_as_vector(&vec_one, &vec_two),
            vec![1., 4., 9.]
        );

        let vector_of_vectors = vec![vec_one.clone(), vec_two.clone()];
        assert_eq!(calculate_average_f64_vector(&vector_of_vectors), vec_one);
        assert_eq!(
            mean_sum_of_squared_differences_between_f64_slices(&vec_one, &vec_two),
            0.
        );
        assert_eq!(sum_of_f64_vectors(&vector_of_vectors), vec![2., 4., 6.]);
    }
}
