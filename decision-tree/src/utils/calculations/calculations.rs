mod calculations_gini;
mod calculations_variance;

pub mod gini {
    use super::calculations_gini;
    use crate::class_counter::ClassCounter;
    pub fn calculate_loss(
        number_of_rows: f64,
        true_rows_count: f64,
        class_counts_left: &ClassCounter,
        class_counts_right: &ClassCounter,
    ) -> f64 {
        calculations_gini::calculate_loss(
            number_of_rows,
            true_rows_count,
            class_counts_left,
            class_counts_right,
        )
    }

    pub fn calculate_gini(class_counts: &ClassCounter, number_of_rows: f64) -> f64 {
        calculations_gini::calculate_gini(class_counts, number_of_rows)
    }
}

pub mod variance {
    use super::calculations_variance;
    pub fn calculate_loss(
        left_variance: f64,
        right_variance: f64,
        left_size: f64,
        right_size: f64,
    ) -> f64 {
        calculations_variance::calculate_loss(left_variance, right_variance, left_size, right_size)
    }

    pub fn calculate_variance(
        sum_of_squared_labels: f64,
        mean_of_labels: f64,
        number_of_labels: f64,
    ) -> f64 {
        calculations_variance::calculate_variance(
            sum_of_squared_labels,
            mean_of_labels,
            number_of_labels,
        )
    }

    pub fn get_label_sums(labels: &Vec<f64>) -> (f64, f64) {
        calculations_variance::get_label_sums(labels)
    }
}
