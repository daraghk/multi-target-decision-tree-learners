use common::datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures};

pub trait Leaf {}

impl Leaf for RegressionLeaf {}

impl Leaf for RegressionLeafNewPartition<'_> {}

impl Leaf for GradBoostLeaf {}

impl Leaf for AMGBoostLeaf {}

#[derive(Debug, Clone)]
pub struct RegressionLeaf {
    pub data: Option<MultiTargetDataSet>,
}

#[derive(Debug, Clone)]
pub struct RegressionLeafNewPartition<'a> {
    pub data: Option<MultiTargetDataSetSortedFeatures<'a>>,
}

#[derive(Debug, Clone)]
pub struct GradBoostLeaf {
    pub leaf_output: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct AMGBoostLeaf {
    pub max_value: Option<f64>,
    pub class: Option<usize>,
}
