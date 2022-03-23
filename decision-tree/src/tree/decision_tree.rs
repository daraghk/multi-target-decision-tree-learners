use std::thread;

use common::datasets::DataSet;

use crate::{
    class_counter::get_class_counts, data_partitioner::partition, leaf::Leaf, node::TreeNode,
    split_finder::SplitFinder,
};

pub struct DecisionTree {
    pub root: TreeNode,
}

impl DecisionTree {
    pub fn new(
        data: DataSet,
        split_finder: SplitFinder,
        number_of_classes: u32,
        use_multi_threading: bool,
    ) -> Self {
        Self {
            root: match use_multi_threading {
                true => build_tree_using_multiple_threads(data, split_finder, number_of_classes),
                false => build_tree(data, split_finder, number_of_classes),
            },
        }
    }
}

fn build_tree(data: DataSet, split_finder: SplitFinder, number_of_classes: u32) -> TreeNode {
    let split_result = (split_finder.find_best_split)(&data, number_of_classes);
    if split_result.gain == 0.0 {
        let predictions = get_class_counts(&data.labels, number_of_classes);
        let leaf = Leaf { predictions };
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
    data: DataSet,
    split_finder: SplitFinder,
    number_of_classes: u32,
) -> TreeNode {
    let split_result = (split_finder.find_best_split)(&data, number_of_classes);
    if split_result.gain == 0.0 {
        let predictions = get_class_counts(&data.labels, number_of_classes);
        let leaf = Leaf { predictions };
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

pub fn print_tree(root: Box<TreeNode>, spacing: String, feature_names: &Vec<String>) {
    if root.leaf.is_some() {
        let leaf_ref = &root.leaf.unwrap();
        println!("{} Predict:{:?}", spacing, leaf_ref.predictions);
        return;
    }
    println!(
        "{}",
        format!(
            "{} {:?}",
            spacing.clone(),
            root.question
                .to_string(feature_names.get(root.question.column as usize).unwrap())
        )
    );
    println!("{}", spacing.clone() + "--> True: ");
    print_tree(
        root.true_branch.unwrap(),
        spacing.clone() + "    ",
        feature_names,
    );

    println!("{}", spacing.clone() + "--> False: ");
    print_tree(
        root.false_branch.unwrap(),
        spacing.clone() + "    ",
        feature_names,
    );
}

#[cfg(test)]
mod tests {
    use crate::split_finder::SplitMetric;
    use common::data_reader::{get_feature_names, read_csv_data};

    use super::*;
    #[test]
    fn test_build_tree() {
        let data_set = read_csv_data("./../common/data-files/iris.csv");
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = DecisionTree::new(data_set, split_finder, 3, false);
        let feature_names = get_feature_names("./../common/data-files/iris.csv");
        print_tree(Box::new(tree.root), "".to_string(), &feature_names)
    }
}
