use std::{sync::Arc, thread};

use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{calculate_average_vector, subtract_vectors},
};

use crate::{data_partitioner::partition, leaf::GradBoostLeaf, node::TreeNode};

use super::TreeConfig;

pub(crate) fn build_grad_boost_regression_tree(
    true_data: &MultiTargetDataSet,
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    current_level: u32,
) -> TreeNode<GradBoostLeaf> {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf_output = calculate_average_leaf_residuals(&true_data, &data);
        let leaf = GradBoostLeaf { leaf_output: Some(leaf_output) };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let left_tree =
            build_grad_boost_regression_tree(true_data, left_data, tree_config, new_level);
        let right_tree =
            build_grad_boost_regression_tree(true_data, right_data, tree_config, new_level);
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

// pub(crate) fn build_grad_boost_regression_tree_using_multiple_threads(
//     true_data: &Box<MultiTargetDataSet>,
//     data: MultiTargetDataSet,
//     tree_config: TreeConfig,
//     current_level: u32,
// ) -> TreeNode<GradBoostLeaf> {
//     let split_result =
//         (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
//     if split_result.gain == 0.0 || current_level == tree_config.max_levels {
//         let leaf_output = calculate_average_leaf_residuals(&true_data, &data);
//         let leaf = GradBoostLeaf { leaf_output: Some(leaf_output) };
//         return TreeNode::leaf_node(split_result.question, leaf);
//     } else {
//         let partitioned_data = partition(&data, &split_result.question);
//         let left_data = partitioned_data.1;
//         let right_data = partitioned_data.0;

//         let new_level = current_level + 1;

//         let arc_true_data = Arc::new(true_data.to_owned());
//         let arc_true_data_clone = arc_true_data.clone();
//         let left_tree_handle = thread::spawn(move || {
//             return build_grad_boost_regression_tree_using_multiple_threads(
//                 &arc_true_data_clone,
//                 left_data,
//                 tree_config,
//                 new_level,
//             );
//         });

//         let right_tree_handle = thread::spawn(move || {
//             return build_grad_boost_regression_tree_using_multiple_threads(
//                 &arc_true_data,
//                 right_data,
//                 tree_config,
//                 new_level,
//             );
//         });

//         let left_tree = left_tree_handle.join().unwrap();
//         let right_tree = right_tree_handle.join().unwrap();

//         TreeNode::new(
//             split_result.question,
//             Box::new(left_tree),
//             Box::new(right_tree),
//         )
//     }
// }

fn calculate_average_leaf_residuals(
    true_data: &MultiTargetDataSet,
    leaf_data: &MultiTargetDataSet,
) -> Vec<f64> {
    let label_length = leaf_data.labels[0].len();
    let mut residuals = vec![vec![0.; label_length]];
    for i in 0..leaf_data.labels.len() {
        let original_index = leaf_data.indices[i];
        let true_data_label = &true_data.labels[original_index];
        let current_data_label = &leaf_data.labels[i];
        residuals.push(subtract_vectors(true_data_label, current_data_label));
    }
    let average_of_residuals = calculate_average_vector(&residuals);
    average_of_residuals
}
