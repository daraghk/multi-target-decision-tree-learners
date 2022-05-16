use multi_target_decision_tree::{leaf::Leaf, node::TreeNode};

pub fn find_leaf_node_for_data<'a, L: Leaf>(
    feature_row: &[f64],
    node: &'a TreeNode<L>,
) -> &'a L {
    if !node.is_leaf_node() {
        if node.question.solve(feature_row) {
            return find_leaf_node_for_data(feature_row, node.true_branch.as_ref().unwrap());
        } else {
            return find_leaf_node_for_data(feature_row, node.false_branch.as_ref().unwrap());
        }
    }
    node.leaf.as_ref().unwrap()
}
