use common::{
    data_processor::create_dataset_with_sorted_features,
    datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures},
};

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
        data: &MultiTargetDataSetSortedFeatures,
        tree_config: TreeConfig,
        leaf_output_calculator: LeafOutputCalculator,
    ) -> Self {
        let all_labels = &data.labels.clone();
        Self {
            root: match tree_config.use_multi_threading {
                true => {
                    grad_boost_tree_builder::build_grad_boost_regression_tree_using_multiple_threads(
                        data,
                        all_labels,
                        tree_config,
                        leaf_output_calculator,
                        0,
                    )
                }
                false => grad_boost_tree_builder::build_grad_boost_regression_tree(
                    data,
                    all_labels,
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
        data: &MultiTargetDataSetSortedFeatures,
        tree_config: TreeConfig,
        leaf_output_calculator: LeafOutputCalculator,
    ) -> Self {
        let all_labels = &data.labels.clone();
        Self {
            root: match tree_config.use_multi_threading {
                true => {
                    approximate_grad_boost_tree_builder::build_approximate_grad_boost_regression_tree_using_multiple_threads(
                        data,
                        all_labels,
                        tree_config,
                        leaf_output_calculator,
                        0,
                    )
                }
                false => approximate_grad_boost_tree_builder::build_approximate_grad_boost_regression_tree(
                    data,
                    all_labels,
                    tree_config,
                    leaf_output_calculator,
                    0,
                ),
            },
        }
    }
}
