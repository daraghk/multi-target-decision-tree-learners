use common::{
    datasets::MultiTargetDataSet,
    numerical_calculations::{
        add_f64_slices_as_vector, calculate_average_f64_vector, divide_f64_slices_as_vector,
        multiply_f64_slice_by_f64_scalar, sum_of_f64_vectors,
    },
};

#[derive(Clone, Copy)]
pub enum LeafOutputType {
    Regression,
    MultiClassClassification,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct LeafOutputCalculator {
    leaf_output_type: LeafOutputType,
    pub calculate_leaf_output: fn(&MultiTargetDataSet) -> Vec<f64>,
}

impl LeafOutputCalculator {
    pub fn new(leaf_output_type: LeafOutputType) -> Self {
        Self {
            leaf_output_type,
            calculate_leaf_output: match leaf_output_type {
                LeafOutputType::Regression => calculate_leaf_output_squared_loss,
                LeafOutputType::MultiClassClassification => calculate_leaf_output_multi_class_loss,
            },
        }
    }
}

pub fn calculate_leaf_output_squared_loss(leaf_data: &MultiTargetDataSet) -> Vec<f64> {
    let average_residuals = calculate_average_f64_vector(&leaf_data.labels);
    average_residuals
}

pub fn calculate_leaf_output_multi_class_loss(leaf_data: &MultiTargetDataSet) -> Vec<f64> {
    let numerator = sum_of_f64_vectors(&leaf_data.labels);
    let denominator = calculate_denominator_term_for_leaf_output(&leaf_data.labels);
    let numerator_over_denominator = divide_f64_slices_as_vector(&numerator, &denominator);
    let number_of_classes = leaf_data.labels[0].len() as f64;
    let scalar = (number_of_classes - 1.) / number_of_classes;
    let result = multiply_f64_slice_by_f64_scalar(scalar, &numerator_over_denominator);
    result
}

fn calculate_denominator_term_for_leaf_output(vector_of_vectors: &Vec<Vec<f64>>) -> Vec<f64> {
    let length_of_inner_vectors = vector_of_vectors[0].len();
    let mut sum_vector = vec![0.; length_of_inner_vectors];
    vector_of_vectors.iter().for_each(|inner_vector| {
        let term: Vec<f64> = inner_vector
            .iter()
            .map(|element| {
                let element_abs = element.abs();
                element_abs * (1. - element_abs)
            })
            .collect();
        sum_vector = add_f64_slices_as_vector(&sum_vector, &term);
    });
    sum_vector
}

#[cfg(test)]
mod tests {
    use common::{
        data_reader::read_csv_data_one_hot_multi_target,
        numerical_calculations::{
            divide_f64_slices_as_vector, multiply_f64_slice_by_f64_scalar, sum_of_f64_vectors,
        },
    };

    use super::{
        calculate_denominator_term_for_leaf_output, calculate_leaf_output_multi_class_loss,
    };

    #[test]
    fn test_leaf_output_multi_class_loss() {
        let vector_of_vectors = vec![vec![0.333, 0.333, 0.333], vec![0.333, 0.333, 0.333]];
        let numerator = sum_of_f64_vectors(&vector_of_vectors);
        let denominator = calculate_denominator_term_for_leaf_output(&vector_of_vectors);
        let scalar = 2. / 3.;
        let division = divide_f64_slices_as_vector(&numerator, &denominator);
        let result = multiply_f64_slice_by_f64_scalar(scalar, &division);
        println!("{:?}", result);
    }
}
