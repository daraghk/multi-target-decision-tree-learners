pub fn get_sorted_feature_tuple_vector(features: &Vec<Vec<f64>>, column: u32) -> Vec<(f64, i32)> {
    let mut feature_tuple_vector = vec![];
    let mut row_index = 0;
    features.iter().for_each(|row| {
        let feature_value = row[column as usize];
        feature_tuple_vector.push((feature_value, row_index));
        row_index += 1;
    });
    //feature_tuple_vector.sort_by_key(|tuple| tuple.0);
    feature_tuple_vector.sort_by(|a, b| a.partial_cmp(&b).unwrap()); // Panics on NaN

    feature_tuple_vector
}

#[cfg(test)]
mod tests {
    use common::datasets::DataSet;

    use super::*;
    #[test]
    fn test_get_sorted_feature_tuple_vector() {
        let features = vec![vec![10., 2., 1.], vec![6., 2., 2.], vec![-1., 2., 3.]];
        let labels = vec![1., 2., 3.];
        let data = DataSet { features, labels };
        let column = 0;
        let sorted_feature_tuple_vector = get_sorted_feature_tuple_vector(&data.features, column);
        println!("{:?}", sorted_feature_tuple_vector);
        assert_eq!(
            sorted_feature_tuple_vector,
            vec![(-1., 2), (6., 1), (10., 0)]
        )
    }
}
