// pub(super) fn calculate_loss(
//     left_variance_vector: Vec<f32>,
//     right_variance_vector: Vec<f32>,
//     left_size: f32,
//     right_size: f32,
// ) -> f32 {
//     // let total_size = left_size + right_size;
//     // ((left_size / total_size) * left_variance_vector)
//     //     + ((right_size / total_size) * right_variance_vector)
//     0.0
// }

// pub(super) fn calculate_variance(
//     sum_of_squared_labels_vector: Vec<f32>,
//     mean_of_labels_vector: Vec<f32>,
//     number_of_labels: f32,
// ) -> f32 {
//     // if number_of_labels == 0.0 {
//     //     return 0.0;
//     // }
//     // let left = sum_of_squared_labels_vector;
//     // let right = number_of_labels * (mean_of_labels_vector * mean_of_labels_vector);
//     // let variance = (left - right) / number_of_labels;
//     // variance
//     0.0
// }

// //TODO: needs to be improved, inefficient looping O(number_of_labels * number_of_targets)
// pub(super) fn get_label_sums_vectors(
//     labels: &Vec<Vec<i32>>,
//     number_of_targets: usize,
// ) -> (Vec<f32>, Vec<f32>) {
//     let mut sum_of_labels_vector = vec![0.0; number_of_targets];
//     let mut sum_of_squared_labels_vector = vec![0.0; number_of_targets];
//     labels.iter().for_each(|label_vector| {
//         let label_value_vector = label_vector;
//         let mut index: usize = 0;
//         label_value_vector.iter().for_each(|component| {
//             let label_value = label_value_vector[index] as f32;
//             sum_of_labels_vector[index] += label_value;
//             sum_of_squared_labels_vector[index] += label_value * label_value;
//             index += 1;
//         });
//     });
//     (sum_of_labels_vector, sum_of_squared_labels_vector)
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     #![feature(core)]
//     use std::simd::f32x4;

//     #[test]
//     fn test_get_label_sums() {
//         let labels = vec![vec![1, 3, 4], vec![12, 5, 3], vec![3, 5, 7]];
//         let label_sum_vectors = super::get_label_sums_vectors(&labels, 3);
//         println!("{:?}", label_sum_vectors);
//     }
// }
