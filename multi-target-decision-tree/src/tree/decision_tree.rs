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

impl MultiTargetDecisionTree {
    pub fn new(
        data: MultiTargetDataSet,
        split_finder: SplitFinder,
        number_of_classes: u32,
        use_multi_threading: bool,
        is_regression_tree: bool,
    ) -> Self {
        Self {
            root: match is_regression_tree {
                true => build_tree_regression(data, split_finder, number_of_classes),
                false => match use_multi_threading {
                    true => {
                        build_tree_using_multiple_threads(data, split_finder, number_of_classes)
                    }
                    false => build_tree(data, split_finder, number_of_classes),
                },
            },
        }
    }
}

pub fn build_tree(
    data: MultiTargetDataSet,
    split_finder: SplitFinder,
    number_of_classes: u32,
) -> TreeNode {
    let split_result = (split_finder.find_best_split)(&data, number_of_classes);
    if split_result.gain == 0.0 {
        let class_counts = get_class_counts_multi_target(&data.labels, number_of_classes);
        let leaf = Leaf {
            predictions: Some(class_counts),
            data: None,
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let left_tree = build_tree(left_data, split_finder, number_of_classes);
        let right_tree = build_tree(right_data, split_finder, number_of_classes);
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

fn build_tree_using_multiple_threads(
    data: MultiTargetDataSet,
    split_finder: SplitFinder,
    number_of_classes: u32,
) -> TreeNode {
    let split_result = (split_finder.find_best_split)(&data, number_of_classes);
    if split_result.gain == 0.0 {
        let predictions = get_class_counts_multi_target(&data.labels, number_of_classes);
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
            return build_tree_using_multiple_threads(left_data, split_finder, number_of_classes);
        });

        let right_tree_handle = thread::spawn(move || {
            return build_tree_using_multiple_threads(right_data, split_finder, number_of_classes);
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
    split_finder: SplitFinder,
    number_of_classes: u32,
) -> TreeNode {
    let split_result = (split_finder.find_best_split)(&data, number_of_classes);
    if split_result.gain == 0.0 {
        let leaf = Leaf {
            predictions: None,
            data: Some(data),
        };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let left_tree = build_tree_regression(left_data, split_finder, number_of_classes);
        let right_tree = build_tree_regression(right_data, split_finder, number_of_classes);
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
        let tree = MultiTargetDecisionTree::new(data_set, split_finder, 3, false, false);
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
        let tree = MultiTargetDecisionTree::new(data_set, split_finder, 2, false, true);
        let feature_names =
            get_feature_names(            "/Users/daraghking/Documents/Thesis/Code/Rust/grad_boost_mcc/tree_learner/common/data-files/multi-target/features_train_mt.csv"
        );
        print_tree_regression(Box::new(tree.root), "".to_string(), &feature_names);
    }
}
