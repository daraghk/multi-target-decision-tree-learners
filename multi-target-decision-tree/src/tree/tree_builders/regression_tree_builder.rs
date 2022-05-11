use common::{
    data_processor,
    datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures},
};

use crate::{leaf::RegressionLeaf, node::TreeNode, split_finder};

use super::TreeConfig;

pub(crate) fn build_regression_tree(
    data: MultiTargetDataSetSortedFeatures,
    all_labels: &Vec<Vec<f64>>,
    tree_config: TreeConfig,
    current_level: u32,
) -> TreeNode<RegressionLeaf> {
    let number_of_cols = data.sorted_feature_columns.len();
    let number_of_targets = data.labels[0].len() as u32;
    let split_result = split_finder::split_finder_variance::find_best_split(
        &data,
        all_labels,
        number_of_targets,
        number_of_cols,
    );
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf = RegressionLeaf { data: Some(data) };
        TreeNode::leaf_node(split_result.question, leaf)
    } else {
        let split_column = split_result.question.column as usize;
        let split_value = split_result.question.value;
        let partitioned_data =
            data_processor::partition(&data, split_column, split_value, all_labels);
        let left_data = partitioned_data.0;
        let right_data = partitioned_data.1;

        let new_level = current_level + 1;
        let left_tree = build_regression_tree(left_data, all_labels, tree_config, new_level);
        let right_tree = build_regression_tree(right_data, all_labels, tree_config, new_level);
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

pub(crate) fn build_regression_tree_using_multiple_threads(
    data: MultiTargetDataSetSortedFeatures,
    all_labels: &Vec<Vec<f64>>,
    tree_config: TreeConfig,
    current_level: u32,
) -> TreeNode<RegressionLeaf> {
    let number_of_cols = data.sorted_feature_columns.len();
    let number_of_targets = data.labels[0].len() as u32;
    let split_result = split_finder::split_finder_variance::find_best_split(
        &data,
        all_labels,
        number_of_targets,
        number_of_cols,
    );
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf = RegressionLeaf { data: Some(data) };
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
                build_regression_tree_using_multiple_threads(
                    left_data,
                    all_labels,
                    tree_config,
                    new_level,
                )
            },
            || {
                build_regression_tree_using_multiple_threads(
                    right_data,
                    all_labels,
                    tree_config,
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

#[cfg(test)]
mod tests {
    use common::data_reader::read_csv_data_multi_target;
    use std::time::Instant;

    use crate::{
        decision_trees::RegressionMultiTargetDecisionTree,
        scorer::regression::calculate_overall_mean_squared_error,
        split_finder::{SplitFinder, SplitMetric},
    };

    use super::*;
    #[test]
    fn test_build_tree_regression() {
        let data_original = read_csv_data_multi_target(
            "./../common/data-files/multi-target/features_train_mt.csv",
            "./../common/data-files/multi-target/labels_train_mt.csv",
        );

        let data = data_processor::create_dataset_with_sorted_features(&data_original);
        let split_finder = SplitFinder::new(SplitMetric::Variance);

        let tree_config = TreeConfig {
            split_finder,
            use_multi_threading: true,
            number_of_classes: 10,
            max_levels: 8,
        };

        let before = Instant::now();
        let tree = RegressionMultiTargetDecisionTree::new(data, tree_config);
        println!("Elapsed time: {:.2?}", before.elapsed());

        let boxed_tree = Box::new(tree.root);
        let test_set = read_csv_data_multi_target(
            "./../common/data-files/multi-target/features_test_mt.csv",
            "./../common/data-files/multi-target/labels_test_mt.csv",
        );
        let score = calculate_overall_mean_squared_error(&test_set, &boxed_tree);
        let rmse = f64::sqrt(score);
        println!("{}", score);
        println!("{}", rmse);
    }
}
