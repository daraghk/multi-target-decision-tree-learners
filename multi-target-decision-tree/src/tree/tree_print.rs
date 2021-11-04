use crate::node::TreeNode;

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
        println!("{} Predict:{:?}", spacing, leaf_ref.data);
        return;
    }
    println!(
        "{}",
        format!(
            "{} {:?}",
            spacing.clone(),
            root.question
                .to_string(&feature_names[root.question.column as usize])
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
