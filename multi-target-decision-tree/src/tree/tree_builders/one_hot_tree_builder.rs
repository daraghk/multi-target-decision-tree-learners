use std::thread;

use common::datasets::MultiTargetDataSet;

use crate::{
    class_counter::get_class_counts_multi_target, data_partitioner::partition,
    leaf::OneHotMultiClassLeaf, node::TreeNode,
};

use super::TreeConfig;

pub(crate) fn build_tree(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
) -> TreeNode<OneHotMultiClassLeaf> {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 {
        let class_counts =
            get_class_counts_multi_target(&data.labels, tree_config.number_of_classes);
        let leaf = OneHotMultiClassLeaf {
            class_counts: Some(class_counts),
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

pub(crate) fn build_tree_using_multiple_threads(
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
) -> TreeNode<OneHotMultiClassLeaf> {
    let split_result =
        (tree_config.split_finder.find_best_split)(&data, tree_config.number_of_classes);
    if split_result.gain == 0.0 {
        let class_counts =
            get_class_counts_multi_target(&data.labels, tree_config.number_of_classes);
        let leaf = OneHotMultiClassLeaf {
            class_counts: Some(class_counts),
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

#[cfg(test)]
mod tests {
    use crate::{
        decision_trees::OneHotMultiTargetDecisionTree,
        printer::print_tree,
        split_finder::{SplitFinder, SplitMetric},
    };
    use common::data_reader::{get_feature_names, read_csv_data_one_hot_multi_target};

    use super::*;
    #[test]
    fn test_build_tree() {
        let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);

        let tree_config = TreeConfig {
            split_finder,
            use_multi_threading: true,
            number_of_classes: 3,
            max_levels: 0,
        };

        let tree = OneHotMultiTargetDecisionTree::new(data_set.clone(), tree_config);
        let feature_names = get_feature_names("./../common/data-files/iris.csv");
        println!("{:?}", feature_names);
        print_tree(Box::new(tree.root), "".to_string(), &feature_names);
    }
}
