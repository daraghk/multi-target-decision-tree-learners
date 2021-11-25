use crate::question::Question;

#[derive(Debug, Clone, Copy)]
pub struct BestThresholdResult {
    pub loss: f64,
    pub threshold_value: f64,
}

#[derive(Debug)]
pub struct BestSplitResult {
    pub gain: f64,
    pub question: Question,
}
