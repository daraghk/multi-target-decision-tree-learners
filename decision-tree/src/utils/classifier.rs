use crate::{leaf::Leaf, node::TreeNode};

fn classify(feature_row: &Vec<f32>, node: Box<TreeNode>) -> Leaf {
    if !node.is_leaf_node() {
        if node.question.solve(feature_row) {
            return classify(feature_row, node.true_branch.unwrap());
        } else {
            return classify(feature_row, node.false_branch.unwrap());
        }
    }
    node.leaf.unwrap()
}

pub fn get_predicted_class(feature_row: &Vec<f32>, node: Box<TreeNode>) -> u32 {
    let leaf = classify(feature_row, node);
    let mut max = 0;
    let mut max_class = 0;
    let mut index = 0;
    leaf.predictions.counts.iter().for_each(|count| {
        if *count > max {
            max = *count;
            max_class = index;
        }
        index += 1;
    });
    max_class
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        decision_tree::{print_tree, DecisionTree},
        split_finder::{SplitFinder, SplitMetric},
    };
    use common::data_reader::read_csv_data;

    #[test]
    fn test_classifier_known_data() {
        let data_set = read_csv_data("./../common/data_files/iris.csv");
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = DecisionTree::new(data_set, split_finder, 3);
        let row_to_classify = vec![58., 27., 51., 19.];
        let predicted_class = get_predicted_class(&row_to_classify, Box::new(tree.root));
        assert_eq!(predicted_class, 2);
    }

    use super::*;
    #[test]
    fn test_classifier_unknown_data() {
        let data_set = read_csv_data("./../common/data_files/iris.csv");
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = DecisionTree::new(data_set, split_finder, 3);
        let row_to_classify = vec![1., 23., 90., 10.];
        let predicted_class = get_predicted_class(&row_to_classify, Box::new(tree.root));
        println!("{:?}", predicted_class);
    }

    #[test]
    fn print_iris_tree_for_ref() {
        let data_set = read_csv_data("./../common/data_files/iris.csv");
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        let tree = DecisionTree::new(data_set, split_finder, 3);
        print_tree(Box::new(tree.root), "".to_string())
    }
}
