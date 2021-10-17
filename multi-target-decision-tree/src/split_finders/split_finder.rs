use common::{datasets::MultiTargetDataSet, results::BestSplitResult};
mod split_finder_variance;

pub enum SplitMetricMultiTarget {
    Variance,
}

pub struct SplitFinderMultiTarget {
    split_metric: SplitMetricMultiTarget,
    pub find_best_split: fn(&MultiTargetDataSet, u32) -> BestSplitResult,
}

impl SplitFinderMultiTarget {
    pub fn new(metric: SplitMetricMultiTarget) -> Self {
        Self {
            split_metric: metric,
            find_best_split: split_finder_variance::find_best_split,
        }
    }
}
