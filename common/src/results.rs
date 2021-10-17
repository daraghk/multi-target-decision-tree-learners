use crate::question::Question;

#[derive(Debug)]
pub struct BestThresholdResult {
    pub loss: f32,
    pub threshold_value: f32,
}

#[derive(Debug)]
pub struct BestSplitResult {
    pub gain: f32,
    pub question: Question,
}