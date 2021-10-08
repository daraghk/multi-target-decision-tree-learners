//T defines the Type of the feature data values
//K defines the Type of the labels, if K = Vec<_> then label is mutli-target
#[derive(Debug)]
pub struct DataSet<T, K> {
    pub features: Vec<Vec<T>>,
    pub labels: Vec<K>,
}