use common::{
    data_reader::create_feature_columns, datasets::MultiTargetDataSet, question::Question,
};

pub fn partition(
    data: &MultiTargetDataSet,
    question: &Question,
) -> (MultiTargetDataSet, MultiTargetDataSet) {
    let mut true_rows = vec![];
    let mut false_rows = vec![];
    let mut true_labels = vec![];
    let mut false_labels = vec![];

    &data
        .feature_rows
        .iter()
        .zip(&data.labels)
        .enumerate()
        .for_each(|(index, (feature_vector, label_vector))| {
            if question.solve(feature_vector) {
                true_rows.push(feature_vector.clone());
                true_labels.push(label_vector.clone());
            } else {
                false_rows.push(feature_vector.clone());
                false_labels.push(label_vector.clone());
            }
        });

    let false_columns = create_feature_columns(&false_rows);
    let true_columns = create_feature_columns(&true_rows);

    let false_data = MultiTargetDataSet {
        feature_rows: false_rows,
        feature_columns: false_columns,
        labels: false_labels,
    };

    let true_data = MultiTargetDataSet {
        feature_rows: true_rows,
        feature_columns: true_columns,
        labels: true_labels,
    };

    (false_data, true_data)
}
