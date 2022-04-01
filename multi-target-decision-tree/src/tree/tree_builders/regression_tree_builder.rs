use common::datasets::MultiTargetDataSet;

use crate::{data_partitioner::partition, leaf::RegressionLeaf, node::TreeNode};

use super::TreeConfig;

pub(crate) fn build_regression_tree(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    current_level: u32,
) -> TreeNode<RegressionLeaf> {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf = RegressionLeaf { data: Some(data) };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let left_tree = build_regression_tree(left_data, tree_config, new_level);
        let right_tree = build_regression_tree(right_data, tree_config, new_level);
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

pub(crate) fn build_regression_tree_using_multiple_threads(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    current_level: u32,
) -> TreeNode<RegressionLeaf> {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf = RegressionLeaf { data: Some(data) };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let (left_tree, right_tree) = rayon::join(
            || {
                return build_regression_tree_using_multiple_threads(
                    left_data,
                    tree_config,
                    new_level,
                );
            },
            || {
                return build_regression_tree_using_multiple_threads(
                    right_data,
                    tree_config,
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

#[cfg(test)]
mod tests {
    use crate::{
        decision_trees::{RegressionMultiTargetDecisionTree, TreeConfig},
        printer::print_tree_regression,
        split_finder::{SplitFinder, SplitMetric},
    };
    use common::data_reader::{get_feature_names, read_csv_data_multi_target};

    #[test]
    fn test_build_tree_regression() {
        let data_set = read_csv_data_multi_target(
            "/Users/daraghking/Documents/Thesis/Code/Rust/grad_boost_mcc/tree_learner/common/data-files/multi-target/features_train_mt.csv",
            "/Users/daraghking/Documents/Thesis/Code/Rust/grad_boost_mcc/tree_learner/common/data-files/multi-target/labels_train_mt.csv",
        );
        let split_finder = SplitFinder::new(SplitMetric::Variance);

        let tree_config = TreeConfig {
            split_finder,
            use_multi_threading: true,
            number_of_classes: 10,
            max_levels: 8,
        };

        let tree = RegressionMultiTargetDecisionTree::new(data_set, tree_config);
        let feature_names =
            get_feature_names(            "/Users/daraghking/Documents/Thesis/Code/Rust/grad_boost_mcc/tree_learner/common/data-files/multi-target/features_train_mt.csv"
        );
        print_tree_regression(&Box::new(tree.root), "".to_string(), &feature_names);
    }
}
