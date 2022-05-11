pub fn get_sorted_feature_tuple_vector(feature_column: &[f64]) -> Vec<(f64, usize)> {
    let mut feature_tuple_vector = vec![];
    for (i, feature_value) in feature_column.iter().enumerate() {
        feature_tuple_vector.push((*feature_value, i));
    }
    feature_tuple_vector.sort_by(|a, b| a.partial_cmp(b).unwrap());
    feature_tuple_vector
}

#[cfg(test)]
mod tests {
    use crate::{data_reader::create_feature_columns, datasets::MultiTargetDataSet};

    use super::*;

    #[test]
    fn test_get_sorted_feature_tuple_vector() {
        let features = vec![vec![10., 2., 1.], vec![6., 2., 2.], vec![-1., 2., 3.]];
        let labels = vec![vec![0.], vec![0.], vec![0.]];
        let columns = create_feature_columns(&features);

        let data = MultiTargetDataSet {
            feature_rows: features,
            feature_columns: columns,
            labels,
        };
        let column = 0;
        let sorted_feature_tuple_vector =
            get_sorted_feature_tuple_vector(&data.feature_columns[column]);
        println!("{:?}", sorted_feature_tuple_vector);
        assert_eq!(
            sorted_feature_tuple_vector,
            vec![(-1., 2), (6., 1), (10., 0)]
        )
    }
}
