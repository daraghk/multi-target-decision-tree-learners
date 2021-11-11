use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{add_vectors, calculate_average_vector, subtract_vectors},
};
use multi_target_decision_tree::{
    decision_trees::{GradBoostMultiTargetDecisionTree, TreeConfig},
    leaf::GradBoostLeaf,
    node::TreeNode,
};

pub mod grad_booster;

fn execute_gradient_boosting_loop(
    true_data: &MultiTargetDataSet,
    mutable_data: &mut MultiTargetDataSet,
    number_of_iterations: u8,
    tree_config: TreeConfig,
) -> Vec<Box<TreeNode<GradBoostLeaf>>> {
    let mut trees = vec![];
    for _i in 0..number_of_iterations {
        let dataset_to_grow_tree = mutable_data.clone();
        let decision_tree =
            GradBoostMultiTargetDecisionTree::new(dataset_to_grow_tree, tree_config);
        let boxed_tree = Box::new(decision_tree.root);
        //gather_leaves(&boxed_tree, &mut leaves);
        update_dataset_labels(&true_data, mutable_data, &boxed_tree);
        println!("{:?}", mutable_data.labels[10]);
        trees.push(boxed_tree);
    }
    trees
}

fn update_dataset_labels(
    true_data: &MultiTargetDataSet,
    mutable_data: &mut MultiTargetDataSet,
    boxed_tree_ref: &Box<TreeNode<GradBoostLeaf>>,
) {
    // for i in 0..mutable_data.labels.len() {
    //     //change to get leaf number so we know what leaf this data falls into
    //     let leaf_data = find_leaf_node_for_data(&mutable_data.features[i], boxed_tree_ref)
    //         .data
    //         .as_ref()
    //         .unwrap();
    //     //use leaf number to retrieve average leaf residuals / leaf output
    //     let average_leaf_residuals = calculate_average_leaf_residuals(true_data, leaf_data);
    //     mutable_data.labels[i] = add_vectors(&mutable_data.labels[i], &average_leaf_residuals);
    // }
}

// fn gather_leaves(node: &Box<TreeNode<GradBoostLeaf>>, leaves: &mut Vec<Leaf>) {
//     if node.is_leaf_node() {
//         leaves.push(node.leaf.as_ref().unwrap().clone());
//         return;
//     } else if node.true_branch.is_some() {
//         gather_leaves(&node.true_branch.as_ref().unwrap(), leaves)
//     }
//     gather_leaves(&node.false_branch.as_ref().unwrap(), leaves)
// }

fn calculate_average_leaf_residuals(
    true_dataset: &MultiTargetDataSet,
    leaf_data: &MultiTargetDataSet,
) -> Vec<f32> {
    let label_length = leaf_data.labels[0].len();
    let mut residuals = vec![vec![0.; label_length]];
    for i in 0..leaf_data.labels.len() {
        let original_index = leaf_data.indices[i];
        let true_data_label = &true_dataset.labels[original_index];
        let current_data_label = &leaf_data.labels[i];
        residuals.push(subtract_vectors(true_data_label, current_data_label));
    }
    let average_of_residuals = calculate_average_vector(&residuals);
    average_of_residuals
}

fn find_leaf_node_for_data<'a>(
    feature_row: &Vec<f32>,
    node: &'a Box<TreeNode<GradBoostLeaf>>,
) -> &'a GradBoostLeaf {
    if !node.is_leaf_node() {
        if node.question.solve(feature_row) {
            return find_leaf_node_for_data(feature_row, &node.true_branch.as_ref().unwrap());
        } else {
            return find_leaf_node_for_data(feature_row, &node.false_branch.as_ref().unwrap());
        }
    }
    node.leaf.as_ref().unwrap()
}

fn update_dataset_labels_with_initial_guess(
    mutable_data: &mut MultiTargetDataSet,
    initial_guess: &Vec<f32>,
) {
    for i in 0..mutable_data.labels.len() {
        mutable_data.labels[i] = initial_guess.clone();
    }
}

fn print_output_diff_between_true_and_final(
    true_data: &MultiTargetDataSet,
    mutable_data: &MultiTargetDataSet,
) {
    for i in 0..mutable_data.labels.len() {
        let diff = subtract_vectors(&mutable_data.labels[i], &true_data.labels[i]);
        println!("{:?}", diff);
    }
}

#[cfg(test)]
mod tests {
    use common::data_reader::read_csv_data_multi_target;
    use multi_target_decision_tree::split_finder::{SplitFinder, SplitMetric};

    use super::*;

    #[test]
    fn execute_gradient_boosting() {
        let true_data = read_csv_data_multi_target(
            "./../common/data-files/multi-target/features_train_mt.csv",
            "./../common/data-files/multi-target/labels_train_mt.csv",
        );
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        println!("{:?}", true_data.labels[10]);
        let mut mutable_data = true_data.clone();
        let initial_guess = calculate_average_vector(&true_data.labels);
        update_dataset_labels_with_initial_guess(&mut mutable_data, &initial_guess);

        let tree_config = TreeConfig {
            split_finder,
            use_multi_threading: false,
            number_of_classes: 4,
            max_levels: 12,
        };

        let mut grad_boost_trees =
            execute_gradient_boosting_loop(&true_data, &mut mutable_data, 20, tree_config);

        print_output_diff_between_true_and_final(&true_data, &mutable_data);
    }

    #[test]
    fn test_gathering_leaves() {
        let true_data = read_csv_data_multi_target(
            "./../common/data-files/multi-target/features_train_mt.csv",
            "./../common/data-files/multi-target/labels_train_mt.csv",
        );
        let split_finder = SplitFinder::new(SplitMetric::Variance);
        println!("{:?}", true_data.labels[10]);
        let mut mutable_data = true_data.clone();
        let initial_guess = calculate_average_vector(&true_data.labels);
        update_dataset_labels_with_initial_guess(&mut mutable_data, &initial_guess);

        let tree_config = TreeConfig {
            split_finder,
            use_multi_threading: false,
            number_of_classes: 4,
            max_levels: 12,
        };

        let mut grad_boost_trees =
            execute_gradient_boosting_loop(&true_data, &mut mutable_data, 12, tree_config);

        //gather_leaves(&grad_boost_trees[10], &mut leaves);
    }
}
