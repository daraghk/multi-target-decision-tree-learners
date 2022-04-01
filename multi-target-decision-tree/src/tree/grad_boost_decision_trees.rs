use common::datasets::MultiTargetDataSet;

use crate::{
    decision_trees::TreeConfig,
    leaf::{AMGBoostLeaf, GradBoostLeaf},
    node::TreeNode,
};

use self::grad_boost_leaf_output::LeafOutputCalculator;
#[path = "tree_builders/grad_boost/approximate_grad_boost_tree_builder.rs"]
mod approximate_grad_boost_tree_builder;
#[path = "tree_builders/grad_boost/grad_boost_leaf_output.rs"]
pub mod grad_boost_leaf_output;
#[path = "tree_builders/grad_boost/grad_boost_tree_builder.rs"]
mod grad_boost_tree_builder;

// Multi target decision tree where each label is a vector, and each label-vector
// contains floating values. These are used to build a multi-target gradient boosting ensemble.
pub struct GradBoostMultiTargetDecisionTree {
    pub root: TreeNode<GradBoostLeaf>,
}

impl GradBoostMultiTargetDecisionTree {
    pub fn new(
        data: MultiTargetDataSet,
        tree_config: TreeConfig,
        leaf_output_calculator: LeafOutputCalculator,
    ) -> Self {
        Self {
            root: match tree_config.use_multi_threading {
                true => {
                    grad_boost_tree_builder::build_grad_boost_regression_tree_using_multiple_threads(
                        data,
                        tree_config,
                        leaf_output_calculator,
                        0,
                    )
                }
                false => grad_boost_tree_builder::build_grad_boost_regression_tree(
                    data,
                    tree_config,
                    leaf_output_calculator,
                    0,
                ),
            },
        }
    }
}

// Multi target decision tree where each label is a vector, and each label-vector
// contains floating values. These are used to build an approximate multi-target gradient boosting ensemble. (AMGBoost)
pub struct AMGBoostTree {
    pub root: TreeNode<AMGBoostLeaf>,
}

impl AMGBoostTree {
    pub fn new(
        data: MultiTargetDataSet,
        tree_config: TreeConfig,
        leaf_output_calculator: LeafOutputCalculator,
    ) -> Self {
        Self {
            root: match tree_config.use_multi_threading {
                true => {
                    approximate_grad_boost_tree_builder::build_approximate_grad_boost_regression_tree_using_multiple_threads(
                        data,
                        tree_config,
                        leaf_output_calculator,
                        0,
                    )
                }
                false => approximate_grad_boost_tree_builder::build_approximate_grad_boost_regression_tree(
                    data,
                    tree_config,
                    leaf_output_calculator,
                    0,
                ),
            },
        }
    }
}
