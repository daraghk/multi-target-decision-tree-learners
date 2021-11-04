use common::datasets::MultiTargetDataSet;

use crate::class_counter::ClassCounter;

#[derive(Debug)]
pub struct Leaf {
    pub predictions: Option<ClassCounter>,
    pub data: Option<MultiTargetDataSet>,
}
