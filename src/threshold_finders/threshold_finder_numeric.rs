use crate::calculations::calculate_loss;
use crate::class_counter::ClassCounter;
use crate::threshold_finder::BestThresholdResult;
use crate::threshold_finder::LastSeen;

pub (super) fn determine_best_numeric_threshold(
    data: &Vec<Vec<i32>>,
    column: u32,
    class_counts_all: &ClassCounter<i32, i32>,
) -> BestThresholdResult {
    let mut best_result_container = BestThresholdResult {
        loss: f32::INFINITY,
        threshold_value: 0.0,
    };
    let number_of_rows = data.len() as u32;

    let mut class_counts_right: ClassCounter<i32, i32> = ClassCounter::new();
    class_counts_right.map = class_counts_all.map.clone();
    let mut class_counts_left: ClassCounter<i32, i32> = ClassCounter::new();

    let sorted_feature_data = get_sorted_feature_tuple_vector(data, column);
    let mut true_rows_count = number_of_rows;
    let mut last_seen = LastSeen {
        count: 0,
        value: sorted_feature_data.get(0).unwrap().0,
    };

    //iterate through the sorted feature, updating counts and determine the best loss
    sorted_feature_data.iter().for_each(|tuple| {
        let feature_val = tuple.0;
        if (feature_val == last_seen.value) {
            last_seen.count += 1;
        } else {
            true_rows_count -= last_seen.count;

            update_class_counts_left(
                &mut class_counts_left,
                &class_counts_right,
                &class_counts_all,
            );

            let loss = calculate_loss(
                number_of_rows as f32,
                true_rows_count as f32,
                &class_counts_left,
                &class_counts_right,
            );
            if (loss < best_result_container.loss) {
                best_result_container.loss = loss;
                best_result_container.threshold_value = feature_val as f32;
            }
            last_seen.count = 1;
            last_seen.value = feature_val;
        }

        //always decrement correct class for a feature value
        update_class_counts_right(tuple, data, &mut class_counts_right);
    });

    best_result_container
}

fn update_class_counts_left(
    class_counts_left: &mut ClassCounter<i32, i32>,
    class_counts_right: &ClassCounter<i32, i32>,
    class_counts_all: &ClassCounter<i32, i32>,
) {
    class_counts_all.map.iter().for_each(|kv| {
        let class = *kv.0;
        class_counts_left.map.insert(
            class,
            class_counts_all.map[&class] - class_counts_right.map[&class],
        );
    });
}

fn update_class_counts_right(
    tuple: &(i32, i32),
    data: &Vec<Vec<i32>>,
    class_counts_right: &mut ClassCounter<i32, i32>,
) {
    let real_row_index = tuple.1;
    let row = &data[real_row_index as usize];
    let class = row[row.len() - 1];
    class_counts_right
        .map
        .insert(class, class_counts_right.map[&class] - 1);
}

fn get_sorted_feature_tuple_vector(data: &Vec<Vec<i32>>, column: u32) -> Vec<(i32, i32)> {
    let mut feature_tuple_vector = vec![];
    let mut row_index = 0;
    data.iter().for_each(|row| {
        let feature_value = row[column as usize];
        feature_tuple_vector.push((feature_value, row_index));
        row_index += 1;
    });
    feature_tuple_vector.sort_by_key(|tuple| tuple.0);
    feature_tuple_vector
}

#[cfg(test)]
mod tests {
    use crate::calculations::get_class_counts;

    use super::*;

    #[test]
    fn test_get_sorted_feature_tuple_vector() {
        let data = vec![vec![10, 2, 1], vec![6, 2, 2], vec![1, 2, 3]];
        let column = 0;
        let sorted_feature_tuple_vector = get_sorted_feature_tuple_vector(&data, column);
        println!("{:?}", sorted_feature_tuple_vector);
        assert_eq!(sorted_feature_tuple_vector, vec![(1, 2), (6, 1), (10, 0)])
    }

    #[test]
    fn test_best_threshold_for_particular_feature() {
        let data = vec![vec![10, 2, 2], vec![6, 2, 2], vec![1, 2, 1]];
        let column = 0;
        let class_counts = get_class_counts(&data);
        let best = determine_best_numeric_threshold(&data, column, &class_counts);
        assert_eq!(best.loss, 0.0);
        assert_eq!(best.threshold_value, 6.0);
    }
}
