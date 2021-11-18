use common::{data_reader::read_csv_data_multi_target, vector_calculations::add_vectors};
use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    split_finder::{SplitFinder, SplitMetric},
};
use multi_target_grad_boost::grad_boost_ensemble::{self, find_leaf_node_for_data};

#[test]
fn test_gradient_boosting() {
    let true_data = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_train_mt.csv",
        "./../common/data-files/multi-target/labels_train_mt.csv",
    );

    let split_finder = SplitFinder::new(SplitMetric::Variance);
    println!("{:?}", true_data.labels[10]);

    let tree_config = TreeConfig {
        split_finder,
        use_multi_threading: false,
        number_of_classes: 4,
        max_levels: 24,
    };

    let mut grad_boost_ensemble =
        grad_boost_ensemble::GradientBoostedEnsemble::train(true_data, tree_config, 20);
    //print_output_diff_between_true_and_final(&box_true_data, &mutable_data);

    let test_set = read_csv_data_multi_target(
        "./../common/data-files/multi-target/features_test_mt.csv",
        "./../common/data-files/multi-target/labels_test_mt.csv",
    );

    let feature_row_test = &test_set.features[100].clone();
    //for each tree traverse to leaf
    //add leaf leaf output to sum vector
    let mut result_vector = vec![0.; test_set.labels.len()];
    for i in 0..grad_boost_ensemble.trees.len(){
        let tree_output = find_leaf_node_for_data(feature_row_test, &grad_boost_ensemble.trees[i]);
        result_vector = add_vectors(&result_vector, &tree_output.leaf_output.as_ref().unwrap());
    }
    println!("{:?}", test_set.labels[100]);
    println!("{:?}", result_vector);
}
