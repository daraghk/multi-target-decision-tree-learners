use crate::class_counter::ClassCounter;

pub(super) fn calculate_loss(
    number_of_rows: f32,
    true_rows_count: f32,
    class_counts_left: &ClassCounter,
    class_counts_right: &ClassCounter,
) -> f32 {
    let false_rows_count = number_of_rows - true_rows_count;
    let gini_left = calculate_gini(class_counts_left, false_rows_count);
    let gini_right = calculate_gini(class_counts_right, true_rows_count);
    let sum = (false_rows_count * gini_left) + (true_rows_count * gini_right);
    let result = sum / number_of_rows;
    result
}

pub(super) fn calculate_gini(class_counts: &ClassCounter, number_of_rows: f32) -> f32 {
    let impurity: f32 = 1.0;
    let mut reduction: f32 = 0.0;
    class_counts.counts.iter().for_each(|class_count| {
        let probability_i = *class_count as f32 / number_of_rows;
        reduction += probability_i * probability_i;
    });
    impurity - reduction
}

#[cfg(test)]
mod tests {
    use crate::class_counter::get_class_counts;

    use super::*;

    #[test]
    fn test_get_class_counts_one_row_and_class() {
        //set up data
        let number_classes = 1;
        let data = vec![0.];
        let class_counts = get_class_counts(&data, number_classes);
        assert_eq!(class_counts.counts.get(0).unwrap(), &1);
    }

    #[test]
    fn test_get_class_counts_multiple_rows_and_classes() {
        //set up data
        let number_classes = 2;
        let data = vec![0., 0., 1.];
        let class_counts = get_class_counts(&data, number_classes);
        assert_eq!(class_counts.counts.get(0).unwrap(), &2);
        assert_eq!(class_counts.counts.get(1).unwrap(), &1);
    }

    #[test]
    fn test_gini_calculation_no_impurity() {
        let number_classes = 1;
        let data = vec![vec![1., 2.], vec![1., 2.], vec![1., 2.]];
        let class_counts = get_class_counts(&vec![0., 0., 0.], number_classes);
        let gini_result = calculate_gini(&class_counts, data.len() as f32);
        assert_eq!(gini_result, 0.0);
    }

    #[test]
    fn test_gini_calculation_has_impurity() {
        let number_classes = 3;
        let data = vec![vec![1., 2., 0.], vec![1., 2., 1.], vec![1., 2., 2.]];
        let class_counts = get_class_counts(&vec![0., 1., 2.], number_classes);
        let gini_result = calculate_gini(&class_counts, data.len() as f32);
        assert_eq!(gini_result, 1.0 - (1.0 / 3.0));
    }
}
