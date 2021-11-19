use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{add_vectors, calculate_average_vector, subtract_vectors},
};
use multi_target_decision_tree::{
    decision_trees::{GradBoostMultiTargetDecisionTree, TreeConfig},
    leaf::GradBoostLeaf,
    node::TreeNode,
};

pub struct GradientBoostedEnsemble {
    pub trees: Vec<Box<TreeNode<GradBoostLeaf>>>,
}

impl GradientBoostedEnsemble {
    pub fn train(
        data: MultiTargetDataSet,
        tree_config: TreeConfig,
        number_of_iterations: u8,
    ) -> GradientBoostedEnsemble {
        let mut mutable_data = data.clone();
        let initial_guess = calculate_average_vector(&mutable_data.labels);
        update_dataset_labels_with_initial_guess(&mut mutable_data, &initial_guess);
        let trees = execute_gradient_boosting_loop(
            data,
            &mut mutable_data,
            number_of_iterations,
            tree_config,
        );
        Self { trees }
    }

    pub fn predict(feature_row: &Vec<f32>, trees: &Vec<Box<TreeNode<GradBoostLeaf>>>) {}

    pub fn test() {}
}

fn execute_gradient_boosting_loop(
    true_data: MultiTargetDataSet,
    mutable_data: &mut MultiTargetDataSet,
    number_of_iterations: u8,
    tree_config: TreeConfig,
) -> Vec<Box<TreeNode<GradBoostLeaf>>> {
    let mut trees = vec![];
    for _i in 0..number_of_iterations {
        let dataset_to_grow_tree = mutable_data.clone();
        let decision_tree =
            GradBoostMultiTargetDecisionTree::new(&true_data, dataset_to_grow_tree, tree_config);
        let boxed_tree = Box::new(decision_tree.root);
        update_dataset_labels(&true_data, mutable_data, &boxed_tree);
        println!("{:?}", mutable_data.labels[10]);
        trees.push(boxed_tree);
    }
    trees
}

fn update_dataset_labels_with_initial_guess(
    mutable_data: &mut MultiTargetDataSet,
    initial_guess: &Vec<f32>,
) {
    for i in 0..mutable_data.labels.len() {
        mutable_data.labels[i] = initial_guess.clone();
    }
}

fn update_dataset_labels(
    true_data: &MultiTargetDataSet,
    mutable_data: &mut MultiTargetDataSet,
    boxed_tree_ref: &Box<TreeNode<GradBoostLeaf>>,
) {
    for i in 0..mutable_data.labels.len() {
        let leaf_data = find_leaf_node_for_data(&mutable_data.features[i], boxed_tree_ref);
        let leaf_output = leaf_data.leaf_output.as_ref().unwrap();
        let leaf_output = leaf_output.into_iter().map(|x| 0.1 * x).collect::<Vec<_>>();
        mutable_data.labels[i] = add_vectors(&mutable_data.labels[i], &leaf_output);
    }
}

pub fn find_leaf_node_for_data<'a>(
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

fn print_output_diff_between_true_and_final(
    true_data: &Box<MultiTargetDataSet>,
    mutable_data: &MultiTargetDataSet,
) {
    for i in 0..mutable_data.labels.len() {
        let diff = subtract_vectors(&mutable_data.labels[i], &true_data.labels[i]);
        println!("{:?}", diff);
    }
}
