use std::u32;

#[derive(Debug, Clone)]
pub struct DataSet {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MultiTargetDataSet {
    pub feature_rows: Vec<Vec<f64>>,
    pub labels: Vec<Vec<f64>>,
    pub feature_columns: Vec<Vec<f64>>,
}
