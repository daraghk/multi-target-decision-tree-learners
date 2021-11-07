use std::thread;

use common::datasets::MultiTargetDataSet;

use crate::{
    class_counter::get_class_counts_multi_target,
    data_partitioner::partition,
    leaf::Leaf,
    node::TreeNode,
    split_finder::{self, SplitFinder},
};

pub struct MultiTargetDecisionTree {
    pub root: TreeNode,
}

#[derive(Copy, Clone)]
pub struct TreeConfig {
    pub split_finder: SplitFinder,
    pub use_multi_threading: bool,
    pub is_regression_tree: bool,
    pub number_of_classes: u32,
    pub max_levels: u32,
}

impl MultiTargetDecisionTree {
    pub fn new(data: MultiTargetDataSet, tree_config: TreeConfig) -> Self {
        Self {
            root: match tree_config.is_regression_tree {
                true => match tree_config.use_multi_threading {
                    true => build_tree_regression_using_multiple_threads(data, tree_config, 0),
                    false => build_tree_regression(data, tree_config, 0),
                },
                false => match tree_config.use_multi_threading {
                    true => build_tree_using_multiple_threads(data, tree_config),
                    false => build_tree(data, tree_config),
                },
            },
        }
    }
}

pub fn build_tree(data: MultiTargetDataSet, tree_config: TreeConfig) -> TreeNode {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 {
        let class_counts =
            get_class_counts_multi_target(&data.labels, tree_config.number_of_classes);
        let leaf = Leaf {
            predictions: Some(class_counts),
            data: None,
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let left_tree = build_tree(left_data, tree_config);
        let right_tree = build_tree(right_data, tree_config);
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

fn build_tree_using_multiple_threads(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
) -> TreeNode {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 {
        let predictions =
            get_class_counts_multi_target(&data.labels, tree_config.number_of_classes);
        let leaf = Leaf {
            predictions: Some(predictions),
            data: None,
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let left_tree_handle = thread::spawn(move || {
            return build_tree_using_multiple_threads(left_data, tree_config);
        });

        let right_tree_handle = thread::spawn(move || {
            return build_tree_using_multiple_threads(right_data, tree_config);
        });

        let left_tree = left_tree_handle.join().unwrap();
        let right_tree = right_tree_handle.join().unwrap();

        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

pub fn build_tree_regression(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    current_level: u32,
) -> TreeNode {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf = Leaf {
            predictions: None,
            data: Some(data),
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let left_tree = build_tree_regression(left_data, tree_config, new_level);
        let right_tree = build_tree_regression(right_data, tree_config, new_level);
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

pub fn build_tree_regression_using_multiple_threads(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    current_level: u32,
) -> TreeNode {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 || current_level == tree_config.max_levels {
        let leaf = Leaf {
            predictions: None,
            data: Some(data),
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let new_level = current_level + 1;
        let left_tree_handle = thread::spawn(move || {
            return build_tree_regression_using_multiple_threads(left_data, tree_config, new_level);
        });

        let right_tree_handle = thread::spawn(move || {
            return build_tree_regression_using_multiple_threads(
                right_data,
                tree_config,
                new_level,
            );
        });

        let left_tree = left_tree_handle.join().unwrap();
        let right_tree = right_tree_handle.join().unwrap();

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
        printer::{print_tree, print_tree_regression},
        split_finder::{SplitFinder, SplitMetric},
    };
    use common::data_reader::{
        get_feature_names, read_csv_data_multi_target, read_csv_data_one_hot_multi_target,
    };

    use super::*;
    #[test]
    fn test_build_tree() {
        let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);

        let tree_config = TreeConfig {
            split_finder,
            use_multi_threading: true,
            is_regression_tree: false,
            number_of_classes: 3,
            max_levels: 0,
        };

        let tree = MultiTargetDecisionTree::new(data_set, tree_config);
        let feature_names = get_feature_names("./../common/data-files/iris.csv");
        println!("{:?}", feature_names);
        print_tree(Box::new(tree.root), "".to_string(), &feature_names);
    }

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
            is_regression_tree: true,
            number_of_classes: 4,
            max_levels: 8,
        };

        let tree = MultiTargetDecisionTree::new(data_set, tree_config);
        let feature_names =
            get_feature_names(            "/Users/daraghking/Documents/Thesis/Code/Rust/grad_boost_mcc/tree_learner/common/data-files/multi-target/features_train_mt.csv"
        );
        print_tree_regression(Box::new(tree.root), "".to_string(), &feature_names);
    }
}
