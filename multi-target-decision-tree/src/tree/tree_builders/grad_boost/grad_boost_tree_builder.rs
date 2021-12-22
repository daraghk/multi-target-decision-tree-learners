use rayon::prelude::*;
use std::{sync::Arc, thread};

use crate::{data_partitioner::partition, leaf::GradBoostLeaf, node::TreeNode};
use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{
        add_vectors, calculate_average_vector, divide_vectors, multiply_vector_by_scalar,
        subtract_vectors, sum_of_vectors,
    },
};

use super::{LeafOutputCalculator, TreeConfig};

pub(crate) fn build_grad_boost_regression_tree(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    leaf_output_calculator: LeafOutputCalculator,
    current_level: u32,
) -> TreeNode<GradBoostLeaf> {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf_output = (leaf_output_calculator.calculate_leaf_output)(&data);
        let leaf = GradBoostLeaf {
            leaf_output: Some(leaf_output),
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let left_tree = build_grad_boost_regression_tree(
            left_data,
            tree_config,
            leaf_output_calculator,
            new_level,
        );
        let right_tree = build_grad_boost_regression_tree(
            right_data,
            tree_config,
            leaf_output_calculator,
            new_level,
        );
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

pub(crate) fn build_grad_boost_regression_tree_using_multiple_threads(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    leaf_output_calculator: LeafOutputCalculator,
    current_level: u32,
) -> TreeNode<GradBoostLeaf> {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf_output = (leaf_output_calculator.calculate_leaf_output)(&data);
        let leaf = GradBoostLeaf {
            leaf_output: Some(leaf_output),
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let (left_tree, right_tree) = rayon::join(
            || {
                return build_grad_boost_regression_tree_using_multiple_threads(
                    left_data,
                    tree_config,
                    leaf_output_calculator,
                    new_level,
                );
            },
            || {
                return build_grad_boost_regression_tree_using_multiple_threads(
                    right_data,
                    tree_config,
                    leaf_output_calculator,
                    new_level,
                );
            },
        );
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}
