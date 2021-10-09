#[path = "threshold_finders/threshold_finder_variance.rs"]
mod threshold_finder_variance;
use super::*;
use crate::{calculations::variance::*, dataset::DataSet};


pub fn find_best_split(data: &DataSet<i32, i32>, number_of_classes: u32) -> BestSplitResult {
    todo!()
}
