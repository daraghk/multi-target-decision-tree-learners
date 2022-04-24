use common::datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures};

#[path = "tree_builders/regression_tree_builder.rs"]
mod regression_tree_builder;

use crate::{
    leaf::{RegressionLeaf, RegressionLeafNewPartition},
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

// Multi target decision tree where each label vector, and each label-vector
// is of the form e.g [1.90, 2.56, 828.1, 0.2828], i.e label vectors contain floating numbers (and also discrete labellings)
pub struct RegressionMultiTargetDecisionTree {
    pub root: TreeNode<RegressionLeaf>,
}

pub struct RegressionMultiTargetDecisionTreeNewPartition<'a> {
    pub root: TreeNode<RegressionLeafNewPartition<'a>>,
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

impl<'a> RegressionMultiTargetDecisionTreeNewPartition<'a>{
    pub fn new(data: MultiTargetDataSetSortedFeatures<'a>, tree_config: TreeConfig) -> Self {
        let all_labels = &data.labels.clone();
        Self {
            root: regression_tree_builder::build_regression_tree_new_partition(
                data,
                all_labels,
                tree_config,
                0,
            ),
        }
    }
}
