use common::{datasets::DataSet, results::BestSplitResult};

mod split_finder_gini;
mod split_finder_variance;

#[derive(Clone, Copy)]
pub enum SplitMetric {
    Gini,
    Variance,
}

pub struct SplitFinder {
    split_metric: SplitMetric,
    pub find_best_split: fn(&DataSet, u32) -> BestSplitResult,
}

impl SplitFinder {
    pub fn new(metric: SplitMetric) -> Self {
        Self {
            split_metric: metric,
            find_best_split: match metric {
                SplitMetric::Gini => split_finder_gini::find_best_split,
                SplitMetric::Variance => split_finder_variance::find_best_split,
            },
        }
    }
}