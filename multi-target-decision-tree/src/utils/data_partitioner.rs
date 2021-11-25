use common::{datasets::MultiTargetDataSet, question::Question};

pub fn partition(
    data: &MultiTargetDataSet,
    question: &Question,
) -> (MultiTargetDataSet, MultiTargetDataSet) {
    let mut true_rows = vec![];
    let mut false_rows = vec![];
    let mut true_labels = vec![];
    let mut false_labels = vec![];
    let mut true_indices = vec![];
    let mut false_indices = vec![];

    &data.features.iter().zip(&data.labels).enumerate().for_each(
        |(index, (feature_vector, label_vector))| {
            let current_data_index = *data.indices.get(index).unwrap();
            if question.solve(feature_vector) {
                true_rows.push(feature_vector.clone());
                true_labels.push(label_vector.clone());
                true_indices.push(current_data_index);
            } else {
                false_rows.push(feature_vector.clone());
                false_labels.push(label_vector.clone());
                false_indices.push(current_data_index);
            }
        },
    );

    let false_data = MultiTargetDataSet {
        features: false_rows,
        labels: false_labels,
        indices: false_indices,
    };

    let true_data = MultiTargetDataSet {
        features: true_rows,
        labels: true_labels,
        indices: true_indices,
    };

    (false_data, true_data)
}
