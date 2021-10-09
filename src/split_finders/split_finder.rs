use crate::question::Question;
use crate::dataset::DataSet;

mod split_finder_gini;
mod split_finder_variance;

#[derive(Debug)]
struct BestThresholdResult {
    loss: f32,
    threshold_value: f32,
}

#[derive(Debug)]
pub struct BestSplitResult {
    pub gain: f32,
    pub question: Question,
}

#[derive(Clone, Copy)]
pub enum SplitMetric{
    Gini,
    Variance
}

pub struct SplitFinder{
    split_metric: SplitMetric,
    pub find_best_split: fn(&DataSet<i32, i32>, u32) -> BestSplitResult
}

impl SplitFinder{
    pub fn new(metric: SplitMetric) -> Self{
        Self{
            split_metric: metric,
            find_best_split: match metric{
                SplitMetric::Gini => use_gini::find_best_split,
                SplitMetric::Variance => use_variance::find_best_split
            }
        }
    }
}

mod use_gini{
    use super::*;
    pub fn find_best_split(data: &DataSet<i32, i32>, number_of_classes: u32)-> super::BestSplitResult {
        split_finder_gini::find_best_split(data, number_of_classes)
    }
}

mod use_variance{
    use super::*;
    pub fn find_best_split(data: &DataSet<i32, i32>, number_of_classes: u32)-> super::BestSplitResult {
        split_finder_variance::find_best_split(data, number_of_classes)
    }
}