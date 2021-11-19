use common::{
    data_reader::read_csv_data_multi_target,
    datasets::MultiTargetDataSet,
    vector_calculations::{
        add_vectors, calculate_average_vector, mean_sum_of_squared_differences_between_vectors,
        subtract_vectors,
    },
};
use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    split_finder::{SplitFinder, SplitMetric},
};
use multi_target_grad_boost::grad_boost_ensemble::{
    self, find_leaf_node_for_data, GradientBoostedEnsemble,
};

#[test]
fn test_gradient_boosting() {
    let true_data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );

    let split_finder = SplitFinder::new(SplitMetric::Variance);
    println!("{:?}", true_data.labels[10]);

    let initial_guess = calculate_average_vector(&true_data.labels);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 4,
        max_levels: 12,
    };

    let grad_boost_ensemble =
        grad_boost_ensemble::GradientBoostedEnsemble::train(true_data, tree_config, 100);

    let test_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_test_mt.csv",
        "./../common/data-files/multi-target/labels_test_mt.csv",
    );

    let number_of_test_instances = test_set.features.len();
    let mut predictions = vec![];
    for i in 0..number_of_test_instances {
        let test_feature_row = &test_set.features[i];
        let test_label_original = &test_set.labels[i];
        let prediction = predict(test_feature_row, &grad_boost_ensemble, &initial_guess);
        let difference = subtract_vectors(test_label_original, &prediction);
        println!(
            "{:?}: Original, {:?}: Result, {:?}: Difference",
            test_label_original, prediction, difference
        );
        predictions.push(prediction);
    }

    assert_eq!(predictions.len(), test_set.labels.len());
    let mean_squared_error = calculate_overall_mean_squared_error(&test_set.labels, &predictions);
    let root_mean_squared_error = f32::sqrt(mean_squared_error);
    println!("{:?}", mean_squared_error);
    println!("{:?}", root_mean_squared_error);
}

fn calculate_overall_mean_squared_error(
    test_data_labels: &Vec<Vec<f32>>,
    predictions: &Vec<Vec<f32>>,
) -> f32 {
    let mut total_error = 0.;
    let number_of_labels = test_data_labels.len();
    for i in 0..number_of_labels {
        total_error +=
            mean_sum_of_squared_differences_between_vectors(&test_data_labels[i], &predictions[i]);
    }
    total_error / number_of_labels as f32
}

fn predict(
    test_feature_row: &Vec<f32>,
    grad_boost_ensemble: &GradientBoostedEnsemble,
    initial_guess: &Vec<f32>,
) -> Vec<f32> {
    let test_instance_leaf_outputs =
        collect_leaf_outputs_for_test_instance(test_feature_row, grad_boost_ensemble);
    let mut sum_of_leaf_outputs = initial_guess.clone();
    for leaf_output in test_instance_leaf_outputs {
        let sum = add_vectors(&leaf_output, &sum_of_leaf_outputs);
        sum_of_leaf_outputs = sum;
    }
    sum_of_leaf_outputs
}

fn collect_leaf_outputs_for_test_instance(
    test_feature_row: &Vec<f32>,
    grad_boost_ensemble: &GradientBoostedEnsemble,
) -> Vec<Vec<f32>> {
    let mut leaf_outputs = vec![];
    for i in 0..grad_boost_ensemble.trees.len() {
        let leaf = find_leaf_node_for_data(test_feature_row, &grad_boost_ensemble.trees[i]);
        let leaf_output = &*leaf.leaf_output.as_ref().unwrap();
        let leaf_output = leaf_output.into_iter().map(|x| 0.1 * x).collect::<Vec<_>>();
        leaf_outputs.push(leaf_output);
    }
    leaf_outputs
}
