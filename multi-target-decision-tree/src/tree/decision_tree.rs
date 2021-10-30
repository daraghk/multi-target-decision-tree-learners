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
        let predictions = get_class_counts_multi_target(&data.labels, number_of_classes);
        let leaf = Leaf {
            predictions: Some(predictions),
            regression_pred: None,
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
            regression_pred: None,
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

pub fn print_tree_regression(root: Box<TreeNode>, spacing: String, feature_names: &Vec<String>) {
    if root.leaf.is_some() {
        let leaf_ref = &root.leaf.unwrap();
        println!("{} Predict:{:?}", spacing, leaf_ref.regression_pred);
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
    print_tree_regression(
        root.true_branch.unwrap(),
        spacing.clone() + "    ",
        feature_names,
    );

    println!("{}", spacing.clone() + "--> False: ");
    print_tree_regression(
        root.false_branch.unwrap(),
        spacing.clone() + "    ",
        feature_names,
    );
}

pub fn build_tree_regression(
    data: MultiTargetDataSet,
    split_finder: SplitFinder,
    number_of_classes: u32,
) -> TreeNode {
    let split_result = (split_finder.find_best_split)(&data, number_of_classes);
    if split_result.gain == 0.0 {
        let prediction = calculate_average_vector(&data);
        let leaf = Leaf {
            predictions: None,
            regression_pred: Some(prediction),
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

fn calculate_average_vector(data: &MultiTargetDataSet) -> Vec<f32> {
    let labels = &data.labels;
    let label_length = data.labels[0].len();
    let mut average_vector = vec![0.; label_length];
    for i in 0..data.labels.len() {
        for j in 0..label_length {
            average_vector[j] += labels[i][j];
        }
    }
    for j in 0..label_length {
        average_vector[j] /= labels.len() as f32;
    }
    average_vector
}

#[cfg(test)]
mod tests {
    use crate::split_finder::{SplitFinder, SplitMetric};
    use common::data_reader::{get_feature_names, read_csv_data_multi_target, read_csv_data_one_hot_multi_target};

    use super::*;
    #[test]
    fn test_build_tree() {
        let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = MultiTargetDecisionTree::new(data_set, split_finder, 3, false, false);
        let feature_names = get_feature_names("./../common/data-files/iris.csv");
        print_tree(Box::new(tree.root), "".to_string(), &feature_names);
    }

    #[test]
    fn test_build_tree_regression() {
        let data_set = read_csv_data_multi_target("./../common/data-files/multi-target/features_train_mt.csv", "./../common/data-files/multi-target/labels_train_mt.csv");
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = MultiTargetDecisionTree::new(data_set, split_finder, 2, false, true);
        let feature_names = get_feature_names("./../common/data-files/multi-target/features_train_mt.csv");
        print_tree_regression(Box::new(tree.root), "".to_string(), &feature_names);
    }

    #[test]
    fn calculate_average_vector_test() {
        let data_set = read_csv_data_one_hot_multi_target("./../common/data-files/iris.csv", 3);
        let average_of_labels = calculate_average_vector(&data_set);
        println!("{:?}", average_of_labels);
    }
}
