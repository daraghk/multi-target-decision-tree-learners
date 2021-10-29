use crate::class_counter::ClassCounter;

#[derive(Debug)]
pub struct Leaf {
    pub predictions: Option<ClassCounter>,
    pub regression_pred: Option<Vec<f32>>,
}