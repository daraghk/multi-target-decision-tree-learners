mod calculations_gini;
mod calculations_variance;
mod calculations_variance_multi_target;

pub mod gini {
    use super::calculations_gini;
    use crate::class_counter::ClassCounter;
    pub fn calculate_loss(
        number_of_rows: f32,
        true_rows_count: f32,
        class_counts_left: &ClassCounter,
        class_counts_right: &ClassCounter,
    ) -> f32 {
        calculations_gini::calculate_loss(
            number_of_rows,
            true_rows_count,
            class_counts_left,
            class_counts_right,
        )
    }

    pub fn calculate_gini(class_counts: &ClassCounter, number_of_rows: f32) -> f32 {
        calculations_gini::calculate_gini(class_counts, number_of_rows)
    }
}

pub mod variance {
    use super::calculations_variance;
    pub fn calculate_loss(
        left_variance: f32,
        right_variance: f32,
        left_size: f32,
        right_size: f32,
    ) -> f32 {
        calculations_variance::calculate_loss(left_variance, right_variance, left_size, right_size)
    }

    pub fn calculate_variance(
        sum_of_squared_labels: f32,
        mean_of_labels: f32,
        number_of_labels: f32,
    ) -> f32 {
        calculations_variance::calculate_variance(
            sum_of_squared_labels,
            mean_of_labels,
            number_of_labels,
        )
    }

    pub fn get_label_sums(labels: &Vec<i32>) -> (f32, f32) {
        calculations_variance::get_label_sums(labels)
    }
}

pub mod variance_multi_target {
    use super::calculations_variance_multi_target;

    #[derive(Debug)]
    pub struct MultiTargetLabelMetrics {
        pub sum_of_labels_vector: Vec<f32>,
        pub sum_of_squared_labels_vector: Vec<f32>,
        pub mean_of_labels_vector: Vec<f32>,
    }

    pub fn calculate_loss_vector(
        left_variance_vector: Vec<f32>,
        right_variance_vector: Vec<f32>,
        left_size: f32,
        right_size: f32,
        number_of_targets: usize,
    ) -> Vec<f32> {
        calculations_variance_multi_target::calculate_loss_vector(
            left_variance_vector,
            right_variance_vector,
            left_size,
            right_size,
            number_of_targets,
        )
    }

    pub fn calculate_variance_vector(
        multi_target_label_metrics: &MultiTargetLabelMetrics,
        number_of_labels: f32,
        number_of_targets: usize,
    ) -> Vec<f32> {
        calculations_variance_multi_target::calculate_variance_vector(
            multi_target_label_metrics,
            number_of_labels,
            number_of_targets,
        )
    }

    pub fn get_multi_target_label_metrics(
        labels: &Vec<Vec<i32>>,
        number_of_targets: usize,
    ) -> MultiTargetLabelMetrics {
        calculations_variance_multi_target::get_multi_target_label_metrics(
            labels,
            number_of_targets,
        )
    }
}
