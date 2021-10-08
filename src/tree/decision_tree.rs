use crate::{class_counter::{get_class_counts}, data_partitioner::partition, dataset::DataSet, split_finder};

use self::{leaf::Leaf, node::TreeNode};

mod leaf;
mod node;

#[derive(Debug)]
pub struct DecisionTree {
    root: TreeNode,
}

impl DecisionTree {
    pub fn new(data: DataSet<i32, i32>, number_of_classes: u32) -> Self {
        Self {
            root: build_tree(data, number_of_classes),
        }
    }
}

pub fn build_tree(data: DataSet<i32, i32>, number_of_classes: u32) -> TreeNode {
    let split_result = split_finder::use_gini::find_best_split(&data, number_of_classes);
    if split_result.gain == 0.0 {
        let predictions = get_class_counts(&data.labels, number_of_classes);
        let leaf = Leaf { predictions };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(&data, &split_result.question);
        let left_data = partitioned_data.1;
        let right_data = partitioned_data.0;

        let left_tree = build_tree(left_data, number_of_classes);
        let right_tree = build_tree(right_data, number_of_classes);
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

fn print_tree(root: Box<TreeNode>, spacing: String) {
    if root.leaf.is_some() {
        let leaf_ref = &root.leaf.unwrap();
        println!("{} Predict:{:?}", spacing, leaf_ref.predictions);
        return;
    }
    println!(
        "{}",
        format!("{} {:?}", spacing.clone(), root.question.to_string())
    );
    println!("{}", spacing.clone() + "--> True: ");
    print_tree(root.true_branch.unwrap(), spacing.clone() + "    ");

    println!("{}", spacing.clone() + "--> False: ");
    print_tree(root.false_branch.unwrap(), spacing.clone() + "    ");
}

#[cfg(test)]
mod tests {
    use crate::data_reader::read_csv_data;

    use super::*;
    #[test]
    fn test_build_tree() {
        let data_set = read_csv_data("./data_arff/iris.csv", false);
        let tree = DecisionTree::new(data_set, 3);
        print_tree(Box::new(tree.root), "".to_string())
    }
}
