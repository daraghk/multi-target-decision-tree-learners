use std::thread;

use common::{
    datasets::MultiTargetDataSet,
    vector_calculations::{calculate_average_vector, subtract_vectors},
};

#[path = "tree_builders/grad_boost_tree_builder.rs"]
mod grad_boost_tree_builder;
#[path = "tree_builders/one_hot_tree_builder.rs"]
mod one_hot_tree_builder;
#[path = "tree_builders/regression_tree_builder.rs"]
mod regression_tree_builder;

use crate::{
    class_counter::get_class_counts_multi_target,
    data_partitioner::partition,
    leaf::{GradBoostLeaf, Leaf, OneHotMultiClassLeaf, RegressionLeaf},
    node::TreeNode,
    split_finder::{self, SplitFinder},
};

#[derive(Copy, Clone)]
pub struct TreeConfig {
    pub split_finder: SplitFinder,
    pub use_multi_threading: bool,
    pub is_regression_tree: bool,
    pub is_grad_boost_tree: bool,
    pub number_of_classes: u32,
    pub max_levels: u32,
}

pub struct OneHotMultiTargetDecisionTree {
    pub root: TreeNode<OneHotMultiClassLeaf>,
}

pub struct RegressionMultiTargetDecisionTree {
    pub root: TreeNode<RegressionLeaf>,
}

pub struct GradBoostMultiTargetDecisionTree {
    pub root: TreeNode<GradBoostLeaf>,
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

impl GradBoostMultiTargetDecisionTree {
    pub fn new(data: MultiTargetDataSet, tree_config: TreeConfig) -> Self {
        Self {
            root: match tree_config.use_multi_threading {
                true => {
                    grad_boost_tree_builder::build_grad_boost_regression_tree_using_multiple_threads(
                        data,
                        tree_config,
                        0,
                    )
                }
                false => {
                    grad_boost_tree_builder::build_grad_boost_regression_tree(data, tree_config, 0)
                }
            },
        }
    }
}
