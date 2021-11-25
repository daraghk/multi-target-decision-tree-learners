pub fn get_sorted_feature_tuple_vector(features: &Vec<Vec<f64>>, column: u32) -> Vec<(f64, usize)> {
    let mut feature_tuple_vector = vec![];
    for (i, row) in features.iter().enumerate() {
        let feature_value = row[column as usize];
        feature_tuple_vector.push((feature_value, i));
    }
    feature_tuple_vector.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    feature_tuple_vector
}

#[cfg(test)]
mod tests {
    use common::datasets::MultiTargetDataSet;

    use super::*;
    #[test]
    fn test_get_sorted_feature_tuple_vector() {
        let features = vec![vec![10., 2., 1.], vec![6., 2., 2.], vec![-1., 2., 3.]];
        let labels = vec![vec![0.], vec![0.], vec![0.]];
        let indices = (0..labels.len()).collect::<Vec<usize>>();
        let data = MultiTargetDataSet {
            features,
            labels,
            indices,
        };
        let column = 0;
        let sorted_feature_tuple_vector = get_sorted_feature_tuple_vector(&data.features, column);
        println!("{:?}", sorted_feature_tuple_vector);
        assert_eq!(
            sorted_feature_tuple_vector,
            vec![(-1., 2), (6., 1), (10., 0)]
        )
    }
}
