use rayon::prelude::*;
use std::{sync::Arc, thread};

use crate::{data_partitioner::partition, leaf::GradBoostLeaf, node::TreeNode};
use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{
        add_vectors, calculate_average_vector, divide_vectors, subtract_vectors, sum_of_vectors,
    },
};

use super::TreeConfig;

pub(crate) fn build_grad_boost_regression_tree(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    current_level: u32,
) -> TreeNode<GradBoostLeaf> {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf_output = calculate_leaf_output_squared_loss(&data);
        let leaf = GradBoostLeaf {
            leaf_output: Some(leaf_output),
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let left_tree = build_grad_boost_regression_tree(left_data, tree_config, new_level);
        let right_tree = build_grad_boost_regression_tree(right_data, tree_config, new_level);
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
    current_level: u32,
) -> TreeNode<GradBoostLeaf> {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf_output = calculate_leaf_output_squared_loss(&data);
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
                    new_level,
                );
            },
            || {
            return build_grad_boost_regression_tree_using_multiple_threads(
                right_data,
                tree_config,
                new_level,
            )}
        );
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

fn calculate_leaf_output_squared_loss(leaf_data: &MultiTargetDataSet) -> Vec<f64> {
    let average_residuals = calculate_average_vector(&leaf_data.labels);
    average_residuals
}

fn calculate_leaf_output_multi_class_loss(leaf_data: &MultiTargetDataSet) -> Vec<f64> {
    let numerator = sum_of_vectors(&leaf_data.labels);
    let denominator = calculate_denominator_term_for_leaf_output(&leaf_data.labels);
    divide_vectors(&numerator, &denominator)
}

fn calculate_denominator_term_for_leaf_output(vector_of_vectors: &Vec<Vec<f64>>) -> Vec<f64> {
    let length_of_inner_vectors = vector_of_vectors[0].len();
    let mut sum_vector = vec![0.; length_of_inner_vectors];
    vector_of_vectors.iter().for_each(|inner_vector| {
        let term: Vec<f64> = inner_vector
            .iter()
            .map(|element| {
                let element_abs = element.abs();
                element_abs * (2. - element_abs)
            })
            .collect();
        sum_vector = add_vectors(&sum_vector, &term);
    });
    sum_vector
}

#[cfg(test)]
mod tests {
    use super::calculate_denominator_term_for_leaf_output;

    #[test]
    fn test_leaf_output_denominator_multi_class_loss() {
        let vector_of_vectors = vec![vec![1., 2., 3.], vec![1., 2., 3.]];
        let result = calculate_denominator_term_for_leaf_output(&vector_of_vectors);
        println!("{:?}", result);
    }
}
