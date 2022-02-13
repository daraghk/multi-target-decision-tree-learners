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
    // pub indices: Vec<usize>,
}

pub struct ChunkedMultiTargetDataSet {
    //every feature_chunk_size elements corresponds to a row
    pub features: Vec<f64>,
    //every label_chunk_size elements corresponds to a label vector
    pub labels: Vec<f64>,
    pub indices: Vec<usize>,
    feature_chunk_size: u32,
    label_chunk_size: u32,
}
