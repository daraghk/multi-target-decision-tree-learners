mod calculations_variance;

pub mod variance {
    use super::calculations_variance;

    #[derive(Debug)]
    pub struct MultiTargetLabelMetrics {
        pub sum_of_labels_vector: Vec<f64>,
        pub sum_of_squared_labels_vector: Vec<f64>,
        pub mean_of_labels_vector: Vec<f64>,
    }

    pub fn calculate_loss_vector(
        left_variance_vector: Vec<f64>,
        right_variance_vector: Vec<f64>,
        left_size: f64,
        right_size: f64,
        number_of_targets: usize,
    ) -> Vec<f64> {
        calculations_variance::calculate_loss_vector(
            left_variance_vector,
            right_variance_vector,
            left_size,
            right_size,
            number_of_targets,
        )
    }

    pub fn calculate_variance_vector(
        multi_target_label_metrics: &MultiTargetLabelMetrics,
        number_of_labels: f64,
        number_of_targets: usize,
    ) -> Vec<f64> {
        calculations_variance::calculate_variance_vector(
            multi_target_label_metrics,
            number_of_labels,
            number_of_targets,
        )
    }

    pub fn get_multi_target_label_metrics(
        labels: &Vec<Vec<f64>>,
        number_of_targets: usize,
    ) -> MultiTargetLabelMetrics {
        calculations_variance::get_multi_target_label_metrics(labels, number_of_targets)
    }
}
