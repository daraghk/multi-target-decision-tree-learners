use rayon::prelude::*;

use crate::{
    leaf::{GradBoostLeaf, RegressionLeaf},
    node::TreeNode,
    split_finder,
};
use common::{
    data_processor,
    datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures},
};

use super::{LeafOutputCalculator, TreeConfig};
pub(crate) fn build_grad_boost_regression_tree(
    data: &MultiTargetDataSetSortedFeatures,
    all_labels: &Vec<Vec<f64>>,
    tree_config: TreeConfig,
    leaf_output_calculator: LeafOutputCalculator,
    current_level: u32,
) -> TreeNode<GradBoostLeaf> {
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
        let data_indices_in_leaf: Vec<usize> = data.sorted_feature_columns[0]
            .iter()
            .map(|(_value, index)| *index)
            .collect();
        let leaf = GradBoostLeaf {
            leaf_output: Some(leaf_output),
            data_indices: data_indices_in_leaf,
        };
        TreeNode::leaf_node(split_result.question, leaf)
    } else {
        let split_column = split_result.question.column as usize;
        let split_value = split_result.question.value;
        let partitioned_data =
            data_processor::partition(&data, split_column, split_value, all_labels);
        let left_data = partitioned_data.0;
        let right_data = partitioned_data.1;

        let new_level = current_level + 1;
        let left_tree = build_grad_boost_regression_tree(
            &left_data,
            all_labels,
            tree_config,
            leaf_output_calculator,
            new_level,
        );
        let right_tree = build_grad_boost_regression_tree(
            &right_data,
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

pub(crate) fn build_grad_boost_regression_tree_using_multiple_threads(
    data: &MultiTargetDataSetSortedFeatures,
    all_labels: &Vec<Vec<f64>>,
    tree_config: TreeConfig,
    leaf_output_calculator: LeafOutputCalculator,
    current_level: u32,
) -> TreeNode<GradBoostLeaf> {
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
        let data_indices_in_leaf: Vec<usize> = data.sorted_feature_columns[0]
            .iter()
            .map(|(_value, index)| *index)
            .collect();
        let leaf = GradBoostLeaf {
            leaf_output: Some(leaf_output),
            data_indices: data_indices_in_leaf,
        };
        TreeNode::leaf_node(split_result.question, leaf)
    } else {
        let split_column = split_result.question.column as usize;
        let split_value = split_result.question.value;
        let partitioned_data =
            data_processor::partition(&data, split_column, split_value, all_labels);
        let left_data = partitioned_data.0;
        let right_data = partitioned_data.1;

        let new_level = current_level + 1;
        let (left_tree, right_tree) = rayon::join(
            || {
                build_grad_boost_regression_tree_using_multiple_threads(
                    &left_data,
                    all_labels,
                    tree_config,
                    leaf_output_calculator,
                    new_level,
                )
            },
            || {
                build_grad_boost_regression_tree_using_multiple_threads(
                    &right_data,
                    all_labels,
                    tree_config,
                    leaf_output_calculator,
                    new_level,
                )
            },
        );
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}
