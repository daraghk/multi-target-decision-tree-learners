use common::datasets::{MultiTargetDataSet, MultiTargetDataSetSortedFeatures};

pub trait Leaf {}

impl Leaf for RegressionLeaf<'_> {}

impl Leaf for GradBoostLeaf<'_> {}

impl Leaf for AMGBoostLeaf<'_> {}

#[derive(Debug, Clone)]
pub struct RegressionLeaf<'a> {
    pub data: Option<MultiTargetDataSetSortedFeatures<'a>>,
}

#[derive(Debug, Clone)]
pub struct GradBoostLeaf<'a> {
    pub data: Option<MultiTargetDataSetSortedFeatures<'a>>,
    pub leaf_output: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct AMGBoostLeaf<'a> {
    pub data: Option<MultiTargetDataSetSortedFeatures<'a>>,
    pub max_value: Option<f64>,
    pub class: Option<usize>,
}
