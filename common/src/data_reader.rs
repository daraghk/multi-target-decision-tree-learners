use std::error::Error;
use std::fs::File;

use crate::datasets::DataSet;
use crate::datasets::MultiTargetDataSet;

pub fn read_csv_data(file_path: &str) -> DataSet {
    let data_set_read = read_data("./../data_arff/iris.csv").unwrap();
    parse_data_into_features_and_labels(data_set_read)
}

pub fn read_csv_data_multi_target(
    file_path: &str,
    number_of_targets: usize,
) -> MultiTargetDataSet {
    let data_set_read = read_data("./../data_files/iris.csv").unwrap();
    let dataset = parse_data_into_features_and_labels(data_set_read);
    let multi_target_labels = create_multi_target_labels(dataset.labels, number_of_targets);
    MultiTargetDataSet {
        features: dataset.features,
        labels: multi_target_labels,
    }
}

//reading in data from csv, presume header included and label is at the end of each record
fn read_data(file_path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut data = vec![];
    let mut reader = csv::Reader::from_reader(file);
    for result in reader.deserialize() {
        let record: Vec<f32> = result?;
        data.push(record);
    }
    Ok(data)
}

fn parse_data_into_features_and_labels(data_set: Vec<Vec<f32>>) -> DataSet {
    let mut features = vec![];
    let mut labels = vec![];
    data_set.iter().for_each(|row| {
        let mut copy = row.clone();
        let label = copy.pop().unwrap();
        labels.push(label);
        features.push(copy);
    });
    DataSet { features, labels }
}

fn create_multi_target_labels(labels: Vec<f32>, number_of_targets: usize) -> Vec<Vec<f32>> {
    let mut multi_target_labels = vec![];
    labels.iter().for_each(|label| {
        let mut multi_target = vec![0.; number_of_targets];
        multi_target[*label as usize] = 1.;
        multi_target_labels.push(multi_target);
    });
    multi_target_labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn print_csv_reading_and_mt_labels() {
        let data_set = read_csv_data("./data_files/iris.csv");
        let mt_labels = create_multi_target_labels(data_set.labels, 3);
        assert_eq!(*mt_labels.get(0).unwrap(), vec![1., 0., 0.]);
        let multi_target_dataset = MultiTargetDataSet {
            features: data_set.features.clone(),
            labels: mt_labels,
        };
        println!("{:?}", multi_target_dataset);
    }

    #[test]
    fn print_csv_reading() {
        let data_set = read_csv_data("./data_files/iris.csv");
        println!("{:?}", data_set);
    }
}
