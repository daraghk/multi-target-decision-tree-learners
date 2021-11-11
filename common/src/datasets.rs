#[derive(Debug, Clone)]
pub struct DataSet {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct MultiTargetDataSet {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<Vec<f32>>,
    pub indices: Vec<usize>,
}
