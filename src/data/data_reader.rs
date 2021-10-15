use std::collections::HashSet;
use std::error::Error;
use std::fs::File;
use std::io;
use std::process;

use csv::StringRecord;

use crate::dataset::DataSet;

pub fn read_csv_data(file_path: &str, has_multi_target_labels: bool) -> DataSet<i32, i32> {
    let data_set_read = read_data("./data_arff/iris.csv").unwrap();
    parse_data_into_features_and_labels(data_set_read)
}

pub fn read_csv_data_multi_target(file_path: &str, has_multi_target_labels: bool) -> DataSet<i32, Vec<i32>> {
    let data_set_read = read_data("./data_arff/iris.csv").unwrap();
    let dataset = parse_data_into_features_and_labels(data_set_read);
    let multi_target_labels = create_multi_target_labels(dataset.labels);
    DataSet{
        features: dataset.features,
        labels: multi_target_labels
    }
}

//reading in data from csv, presume header included and label is at the end of each record
fn read_data(file_path: &str) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut data = vec![];
    let mut reader = csv::Reader::from_reader(file);
    for result in reader.deserialize() {
        let record: Vec<i32> = result?;
        data.push(record);
    }
    Ok(data)
}

fn parse_data_into_features_and_labels(data_set: Vec<Vec<i32>>) -> DataSet<i32, i32>{
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

fn create_multi_target_labels(labels: Vec<i32>) -> Vec<Vec<i32>> {
    let unique_labels: HashSet<i32> = labels.iter().cloned().collect();
    let number_of_labels = unique_labels.len();
    let mut multi_target_labels = vec![];
    labels.iter().for_each(|label| {
        let mut multi_target = vec![0; number_of_labels];
        multi_target[*label as usize] = 1;
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
        let data_set = read_csv_data("./data_arff/iris.csv", false);
        let mt_labels = create_multi_target_labels(data_set.labels);
        assert_eq!(*mt_labels.get(0).unwrap(), vec![1, 0, 0]);
        let multi_target_dataset = DataSet{
            features: data_set.features.clone(),
            labels: mt_labels
        };
        println!("{:?}", multi_target_dataset);
    }
}