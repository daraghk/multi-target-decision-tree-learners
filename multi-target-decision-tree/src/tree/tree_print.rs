use crate::{leaf::RegressionLeaf, node::TreeNode};

pub fn print_tree_regression(
    root: &Box<TreeNode<RegressionLeaf>>,
    spacing: String,
    feature_names: &[String],
) {
    if root.leaf.is_some() {
        let leaf_ref = &root.leaf.as_ref().unwrap();
        println!("{} Predict:{:?}", spacing, leaf_ref.data);
        return;
    }
    println!(
        "{}",
        format!(
            "{} {:?}",
            spacing,
            root.question
                .to_string(&feature_names[root.question.column as usize])
        )
    );
    println!("{}", spacing.clone() + "--> True: ");
    print_tree_regression(
        root.true_branch.as_ref().unwrap(),
        spacing.clone() + "    ",
        feature_names,
    );

    println!("{}", spacing.clone() + "--> False: ");
    print_tree_regression(
        root.false_branch.as_ref().unwrap(),
        spacing + "    ",
        feature_names,
    );
}
