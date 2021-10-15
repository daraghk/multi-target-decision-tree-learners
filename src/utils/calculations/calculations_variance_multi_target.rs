use super::variance_multi_target::MultiTargetLabelMetrics;

pub(super) fn calculate_loss_vector(
    left_variance_vector: Vec<f32>,
    right_variance_vector: Vec<f32>,
    left_size: f32,
    right_size: f32,
    number_of_targets: usize,
) -> Vec<f32> {
    let mut loss_vector = vec![0.0; number_of_targets];
    let total_size = left_size + right_size;
    let left_weight = left_size / total_size;
    let right_weight = right_size / total_size;
    for i in 0..number_of_targets {
        loss_vector[i] =
            (left_weight * left_variance_vector[i]) + (right_weight * right_variance_vector[i]);
    }
    loss_vector
}

pub(super) fn calculate_variance_vector(
    multi_target_label_metrics: &MultiTargetLabelMetrics,
    number_of_labels: f32,
    number_of_targets: usize,
) -> Vec<f32> {
    let mut variance_result_vector = vec![0.0; number_of_targets];
    for i in 0..number_of_targets {
        let mean = multi_target_label_metrics.mean_of_labels_vector[i];
        let left = multi_target_label_metrics.sum_of_squared_labels_vector[i];
        let right = number_of_labels * (mean * mean);
        let variance = (left - right) / number_of_labels;
        variance_result_vector[i] = variance;
    }
    variance_result_vector
}

pub(super) fn get_multi_target_label_metrics(
    labels: &Vec<Vec<i32>>,
    number_of_targets: usize,
) -> MultiTargetLabelMetrics {
    let label_sum_vectors = get_label_sum_vectors(labels, number_of_targets);
    let sum_of_labels_vector = label_sum_vectors.0;
    let sum_of_squared_labels_vector = label_sum_vectors.1;
    let number_of_labels = labels.len() as f32;
    let mut mean_of_labels_vector =
        get_mean_of_labels_vector(number_of_labels, number_of_targets, &sum_of_labels_vector);
    MultiTargetLabelMetrics {
        sum_of_labels_vector,
        sum_of_squared_labels_vector,
        mean_of_labels_vector,
    }
}

fn get_label_sum_vectors(labels: &Vec<Vec<i32>>, number_of_targets: usize) -> (Vec<f32>, Vec<f32>) {
    let mut sum_of_labels_vector = vec![0.0; number_of_targets];
    let mut sum_of_squared_labels_vector = vec![0.0; number_of_targets];
    labels.iter().for_each(|label_vector| {
        for i in 0..label_vector.len() {
            let label_value = label_vector[i] as f32;
            sum_of_labels_vector[i] += label_value;
            sum_of_squared_labels_vector[i] += label_value * label_value;
        }
    });
    (sum_of_labels_vector, sum_of_squared_labels_vector)
}

fn get_mean_of_labels_vector(
    number_of_labels: f32,
    number_of_targets: usize,
    sum_of_labels_vector: &Vec<f32>,
) -> Vec<f32> {
    let mut mean_of_labels_vector = vec![0.0; number_of_targets];
    for i in 0..number_of_targets {
        mean_of_labels_vector[i] = sum_of_labels_vector[i] / number_of_labels;
    }
    mean_of_labels_vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_label_sums() {
        let labels = vec![vec![1, 3, 4], vec![12, 5, 3], vec![3, 5, 7]];
        let label_metrics = super::get_multi_target_label_metrics(&labels, 3);
        println!("{:?}", label_metrics);
    }

    #[test]
    fn test_calculate_variance_vector() {
        let labels = vec![vec![1, 3, 4], vec![2, 5, 3], vec![3, -5, 7]];
        let number_of_targets = 3;
        let label_metrics = super::get_multi_target_label_metrics(&labels, number_of_targets);
        let variance_vector = calculate_variance_vector(
            &label_metrics,
            label_metrics.sum_of_labels_vector.len() as f32,
            number_of_targets,
        );
        println!("{:?}", label_metrics);
        println!("{:?}", variance_vector);
        assert_eq!(variance_vector[0], 2.0 / 3.0);
    }
}
