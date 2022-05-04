// Used to read in single-target datasets that are to have their labels one-hot encoded
#[derive(Debug, Clone)]
pub struct SingleTargetDataSet {
    pub features: Vec<Vec<f64>>,
    pub labels: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct MultiTargetDataSet {
    pub feature_rows: Vec<Vec<f64>>,
    pub labels: Vec<Vec<f64>>,
    pub feature_columns: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct MultiTargetDataSetSortedFeatures {
    pub labels: Vec<Vec<f64>>,
    pub sorted_feature_columns: Vec<Vec<(f64, usize)>>,
}
