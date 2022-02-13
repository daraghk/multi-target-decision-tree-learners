use common::datasets::MultiTargetDataSet;

use crate::class_counter::ClassCounter;

pub trait Leaf {}

impl Leaf for OneHotMultiClassLeaf {}

impl Leaf for RegressionLeaf {}

impl Leaf for GradBoostLeaf {}

impl Leaf for AMGBoostLeaf {}
#[derive(Debug, Clone)]
pub struct OneHotMultiClassLeaf {
    pub class_counts: Option<ClassCounter>,
}

#[derive(Debug, Clone)]
pub struct RegressionLeaf {
    pub data: Option<MultiTargetDataSet>,
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
