#[derive(Debug, Clone)]
pub struct DataSet {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<f32>,
}

#[derive(Debug)]
pub struct MultiTargetDataSet {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub enum Data {
    SingleTarget(DataSet),
    MultiTarget(MultiTargetDataSet),
}

#[cfg(test)]
mod tests {
    use crate::data_reader::read_csv_data;

    use super::Data;

    #[test]
    fn test_build_tree() {
        let data_set = read_csv_data("./../common/data_files/iris.csv");
        let single_target = Data::SingleTarget(data_set);
        match single_target {
            Data::SingleTarget(_) => {
                println!("{:?}", single_target);
            }
            _ => {}
        }
    }
}
