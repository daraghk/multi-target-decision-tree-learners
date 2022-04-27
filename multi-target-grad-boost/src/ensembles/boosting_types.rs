use common::datasets::MultiTargetDataSet;
use multi_target_decision_tree::{
    decision_trees::TreeConfig,
    leaf::{AMGBoostLeaf, GradBoostLeaf, Leaf},
    node::TreeNode,
};

pub struct BoostingResult<T: Leaf> {
    pub trees: Vec<Box<TreeNode<T>>>,
    pub initial_guess: Vec<f64>,
    pub learning_rate: f64,
}

pub struct AMGBoostModel<'a> {
    pub trees: Vec<Box<TreeNode<AMGBoostLeaf<'a>>>>,
    pub initial_guess: Vec<f64>,
    pub learning_rate: f64,
}

pub struct MultiClassBoostModel<'a> {
    pub trees: Vec<Box<TreeNode<GradBoostLeaf<'a>>>>,
    pub initial_guess: Vec<f64>,
    pub learning_rate: f64,
}

pub struct RegressionBoostModel<'a> {
    pub trees: Vec<Box<TreeNode<GradBoostLeaf<'a>>>>,
    pub initial_guess: Vec<f64>,
    pub learning_rate: f64,
}

pub enum BoostingEnsembleType {
    AMGBoost,
    MultiClassBoost,
    RegressionBoost,
}

pub struct BoostingExecutor<'a, T: Leaf> {
    pub ensemble_type: BoostingEnsembleType,
    pub loop_executor_function: fn(
        training_data: &'a mut GradBoostTrainingData,
        number_of_iterations: u32,
        tree_config: TreeConfig,
        learning_rate: f64,
    ) -> Vec<(Box<TreeNode<T>>, Vec<Vec<f64>>)>,
}

pub struct GradBoostTrainingData {
    pub data: MultiTargetDataSet,
    pub mutable_labels: Vec<Vec<f64>>,
    pub size: usize,
}
