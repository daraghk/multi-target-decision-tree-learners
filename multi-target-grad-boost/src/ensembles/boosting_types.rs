use common::datasets::MultiTargetDataSet;
use multi_target_decision_tree::{
    decision_trees::{grad_boost_leaf_output::LeafOutputCalculator, TreeConfig},
    leaf::{AMGBoostLeaf, GradBoostLeaf, Leaf},
    node::TreeNode,
};

pub struct BoostingResult<T: Leaf> {
    pub trees: Vec<Box<TreeNode<T>>>,
    pub initial_guess: Vec<f64>,
    pub learning_rate: f64,
}

pub struct AMGBoostModel {
    pub trees: Vec<Box<TreeNode<AMGBoostLeaf>>>,
    pub initial_guess: Vec<f64>,
    pub learning_rate: f64,
}

pub struct MultiClassBoostModel {
    pub trees: Vec<Box<TreeNode<GradBoostLeaf>>>,
    pub initial_guess: Vec<f64>,
    pub learning_rate: f64,
}

pub struct RegressionBoostModel {
    pub trees: Vec<Box<TreeNode<GradBoostLeaf>>>,
    pub initial_guess: Vec<f64>,
    pub learning_rate: f64,
}

pub enum BoostingEnsembleType {
    AMGBoost,
    MultiClassBoost,
    RegressionBoost,
}

pub struct BoostingExecutor<T: Leaf> {
    pub ensemble_type: BoostingEnsembleType,
    pub loop_executor_function: fn(
        training_data: &mut GradBoostTrainingData,
        number_of_iterations: u32,
        tree_config: TreeConfig,
        leaf_output_calculator: LeafOutputCalculator,
        learning_rate: f64,
    ) -> Vec<Box<TreeNode<T>>>,
}

pub struct GradBoostTrainingData {
    pub data: MultiTargetDataSet,
    pub mutable_labels: Vec<Vec<f64>>,
    pub size: usize,
}
