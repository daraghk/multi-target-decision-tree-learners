use crate::{
    calculations::{get_class_counts, partition},
    class_counter::ClassCounter,
    threshold_finder,
};

use self::{leaf::Leaf, node::TreeNode};

mod leaf;
mod node;

#[derive(Debug)]
pub struct DecisionTree {
    root: TreeNode,
}

impl DecisionTree {
    pub fn new(data: &Vec<Vec<i32>>) -> Self {
        Self {
            root: build_tree(data),
        }
    }
}

pub fn build_tree(data: &Vec<Vec<i32>>) -> TreeNode {
    let split_result = threshold_finder::find_best_split(data);
    if split_result.gain == 0.0 {
        let predictions = get_class_counts(data);
        let leaf = Leaf { predictions };
        return TreeNode::leaf_node(split_result.question, leaf);
    } else {
        let partitioned_data = partition(data, &split_result.question);
        let left_data = partitioned_data.0;
        let right_data = partitioned_data.1;

        let left_tree = build_tree(&left_data);
        let right_tree = build_tree(&right_data);
        TreeNode::new(
            split_result.question,
            Box::new(left_tree),
            Box::new(right_tree),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use crate::question;

    use super::*;
    #[test]
    fn test_build_tree() {
        let _data = fs::read_to_string("./data_arff/iris.arff").expect("Unable to read file");
        let iris: Vec<Vec<i32>> = arff::from_str(&_data).unwrap();
        let tree = DecisionTree::new(&iris);
        println!("{:?}", tree);
    }
}
