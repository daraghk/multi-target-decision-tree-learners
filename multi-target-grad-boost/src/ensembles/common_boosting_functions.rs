pub mod update_common {
    use common::numerical_calculations::add_f64_slices_as_vector;
    use multi_target_decision_tree::{leaf::GradBoostLeaf, node::TreeNode};

    use crate::{
        boosting_ensemble::boosting_types::GradBoostTrainingData,
        tree_traverse::find_leaf_node_for_data,
    };

    //Common to AMGBoost MultiClassBoost and RegressionBoost
    pub fn update_dataset_labels_with_initial_guess(
        mutable_labels: &mut Vec<Vec<f64>>,
        initial_guess: &Vec<f64>,
    ) {
        for i in 0..mutable_labels.len() {
            mutable_labels[i] = initial_guess.clone();
        }
    }

    //Common to MultiClassBoost and RegressionBoost
    pub fn update_dataset_labels(
        training_data: &mut GradBoostTrainingData,
        boxed_tree_ref: &Box<TreeNode<GradBoostLeaf>>,
        learning_rate: f64,
    ) {
        for i in 0..training_data.size {
            let leaf_data =
                find_leaf_node_for_data(&training_data.data.feature_rows[i], boxed_tree_ref);
            let leaf_output = leaf_data.leaf_output.as_ref().unwrap();
            let weighted_leaf_output = leaf_output
                .into_iter()
                .map(|x| learning_rate * x)
                .collect::<Vec<_>>();
            training_data.mutable_labels[i] =
                add_f64_slices_as_vector(&training_data.mutable_labels[i], &weighted_leaf_output);
        }
    }
}

pub mod predict_common {
    use multi_target_decision_tree::{leaf::GradBoostLeaf, node::TreeNode};

    use crate::tree_traverse::find_leaf_node_for_data;

    //Common to MultiClassBoost and RegressionBoost
    pub fn predict_instance(
        test_feature_row: &[f64],
        trees: &Vec<Box<TreeNode<GradBoostLeaf>>>,
        initial_guess: &[f64],
        learning_rate: f64,
    ) -> Vec<f64> {
        let test_instance_leaf_outputs =
            collect_leaf_outputs_for_test_instance(test_feature_row, trees, learning_rate);
        let mut sum_of_leaf_outputs = initial_guess.clone().to_owned();
        for leaf_output in test_instance_leaf_outputs {
            for i in 0..sum_of_leaf_outputs.len() {
                sum_of_leaf_outputs[i] += leaf_output[i];
            }
        }
        sum_of_leaf_outputs
    }

    //Common to MultiClassBoost and RegressionBoost
    fn collect_leaf_outputs_for_test_instance(
        test_feature_row: &[f64],
        trees: &Vec<Box<TreeNode<GradBoostLeaf>>>,
        learning_rate: f64,
    ) -> Vec<Vec<f64>> {
        let mut leaf_outputs = vec![];
        for i in 0..trees.len() {
            let leaf = find_leaf_node_for_data(test_feature_row, &trees[i]);
            let leaf_output = &*leaf.leaf_output.as_ref().unwrap();
            let weighted_leaf_output = leaf_output
                .into_iter()
                .map(|x| learning_rate * x)
                .collect::<Vec<_>>();
            leaf_outputs.push(weighted_leaf_output.clone());
        }
        leaf_outputs
    }
}
