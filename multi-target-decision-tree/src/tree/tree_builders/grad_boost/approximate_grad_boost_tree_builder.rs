use rayon::prelude::*;

use super::{LeafOutputCalculator, TreeConfig};
use crate::{leaf::AMGBoostLeaf, node::TreeNode, split_finder};
use common::{
    data_processor,
    datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures},
};

pub(crate) fn build_approximate_grad_boost_regression_tree(
    data: MultiTargetDataSetSortedFeatures,
    all_labels: &Vec<Vec<f64>>,
    tree_config: TreeConfig,
    leaf_output_calculator: LeafOutputCalculator,
    current_level: u32,
) -> TreeNode<AMGBoostLeaf> {
    let number_of_cols = data.sorted_feature_columns.len();
    let number_of_targets = data.labels[0].len() as u32;
    let split_result = split_finder::split_finder_variance::find_best_split(
        &data,
        all_labels,
        number_of_targets,
        number_of_cols,
    );
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf_output = (leaf_output_calculator.calculate_leaf_output)(&data.labels);
        let (max_value, class) = find_max_value_and_index_from_vector(&leaf_output);
        let leaf = AMGBoostLeaf {
            max_value: Some(max_value),
            class: Some(class),
            data: Some(data),
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let split_column = split_result.question.column as usize;
        let split_value = split_result.question.value;
        let partitioned_data =
            data_processor::partition(&data, split_column, split_value, all_labels);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let left_tree = build_approximate_grad_boost_regression_tree(
            left_data,
            all_labels,
            tree_config,
            leaf_output_calculator,
            new_level,
        );
        let right_tree = build_approximate_grad_boost_regression_tree(
            right_data,
            all_labels,
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

pub(crate) fn build_approximate_grad_boost_regression_tree_using_multiple_threads(
    data: MultiTargetDataSetSortedFeatures,
    all_labels: &Vec<Vec<f64>>,
    tree_config: TreeConfig,
    leaf_output_calculator: LeafOutputCalculator,
    current_level: u32,
) -> TreeNode<AMGBoostLeaf> {
    let number_of_cols = data.sorted_feature_columns.len();
    let number_of_targets = data.labels[0].len() as u32;
    let split_result = split_finder::split_finder_variance::find_best_split(
        &data,
        all_labels,
        number_of_targets,
        number_of_cols,
    );
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf_output = (leaf_output_calculator.calculate_leaf_output)(&data.labels);
        let (max_value, class) = find_max_value_and_index_from_vector(&leaf_output);
        let leaf = AMGBoostLeaf {
            max_value: Some(max_value),
            class: Some(class),
            data: Some(data),
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let split_column = split_result.question.column as usize;
        let split_value = split_result.question.value;
        let partitioned_data =
            data_processor::partition(&data, split_column, split_value, all_labels);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let (left_tree, right_tree) = rayon::join(
            || {
                return build_approximate_grad_boost_regression_tree_using_multiple_threads(
                    left_data,
                    all_labels,
                    tree_config,
                    leaf_output_calculator,
                    new_level,
                );
            },
            || {
                return build_approximate_grad_boost_regression_tree_using_multiple_threads(
                    right_data,
                    all_labels,
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

fn find_max_value_and_index_from_vector(numbers: &[f64]) -> (f64, usize) {
    let mut max_index = 0;
    let mut max_seen = f64::NEG_INFINITY;
    numbers.iter().enumerate().for_each(|(index, element)| {
        if *element > max_seen {
            max_seen = *element;
            max_index = index;
        }
    });
    (max_seen, max_index)
}
