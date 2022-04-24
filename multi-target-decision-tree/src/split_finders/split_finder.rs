use common::{datasets::MultiTargetDataSet, results::BestSplitResult};
pub mod split_finder_variance;

#[derive(Clone, Copy)]
pub enum SplitMetric {
    Variance,
}

#[allow(unused)]
#[derive(Clone, Copy)]
pub struct SplitFinder {
    split_metric: SplitMetric,
    pub find_best_split: fn(&MultiTargetDataSet, u32) -> BestSplitResult,
}

impl SplitFinder {
    pub fn new(metric: SplitMetric) -> Self {
        Self {
            split_metric: metric,
            find_best_split: split_finder_variance::find_best_split,
        }
    }
}
