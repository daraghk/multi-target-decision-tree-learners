use common::datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures};

pub trait Leaf {}

impl Leaf for RegressionLeaf {}

impl Leaf for GradBoostLeaf {}

impl Leaf for AMGBoostLeaf {}

#[derive(Debug, Clone)]
pub struct RegressionLeaf {
    pub data: Option<MultiTargetDataSetSortedFeatures>,
}

#[derive(Debug, Clone)]
pub struct GradBoostLeaf {
    pub data: Option<MultiTargetDataSetSortedFeatures>,
    pub leaf_output: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct AMGBoostLeaf {
    pub data: Option<MultiTargetDataSetSortedFeatures>,
    pub max_value: Option<f64>,
    pub class: Option<usize>,
}
