use common::datasets::MultiTargetDataSet;

#[path = "tree_builders/one_hot_tree_builder.rs"]
mod one_hot_tree_builder;
#[path = "tree_builders/regression_tree_builder.rs"]
mod regression_tree_builder;

use crate::{
    leaf::{OneHotMultiClassLeaf, RegressionLeaf},
    node::TreeNode,
    split_finder::SplitFinder,
};

#[derive(Copy, Clone)]
pub struct TreeConfig {
    pub split_finder: SplitFinder,
    pub use_multi_threading: bool,
    pub number_of_classes: u32,
    pub max_levels: u32,
}

// Multi target decision tree where each label is a vector, and each label-vector
// is of the form e.g [0, 0 , 1, 0, 0], i.e the label vectors are 'one-hot' encoded.
pub struct OneHotMultiTargetDecisionTree {
    pub root: TreeNode<OneHotMultiClassLeaf>,
}

impl OneHotMultiTargetDecisionTree {
    pub fn new(data: MultiTargetDataSet, tree_config: TreeConfig) -> Self {
        Self {
            root: match tree_config.use_multi_threading {
                true => one_hot_tree_builder::build_tree_using_multiple_threads(data, tree_config),
                false => one_hot_tree_builder::build_tree(data, tree_config),
            },
        }
    }
}

// Multi target decision tree where each label vector, and each label-vector
// is of the form e.g [1.90, 2.56, 828.1, 0.2828], i.e label vectors contain floating numbers, not disrete labellings
pub struct RegressionMultiTargetDecisionTree {
    pub root: TreeNode<RegressionLeaf>,
}

impl RegressionMultiTargetDecisionTree {
    pub fn new(data: MultiTargetDataSet, tree_config: TreeConfig) -> Self {
        Self {
            root: match tree_config.use_multi_threading {
                true => regression_tree_builder::build_regression_tree_using_multiple_threads(
                    data,
                    tree_config,
                    0,
                ),
                false => regression_tree_builder::build_regression_tree(data, tree_config, 0),
            },
        }
    }
}
