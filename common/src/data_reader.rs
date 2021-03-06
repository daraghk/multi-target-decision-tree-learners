use std::error::Error;
use std::fs::File;

use csv::StringRecord;

use crate::datasets::MultiTargetDataSet;
use crate::datasets::SingleTargetDataSet;

pub fn read_csv_data(file_path: &str) -> SingleTargetDataSet {
    let data_set_read = read_data(file_path).unwrap();
    parse_data_into_features_and_labels(data_set_read)
}

pub fn read_csv_data_one_hot_multi_target(
    file_path: &str,
    number_of_targets: usize,
) -> MultiTargetDataSet {
    let data_set_read = read_data(file_path).unwrap();
    let dataset = parse_data_into_features_and_labels(data_set_read);
    let multi_target_labels = create_multi_target_labels(dataset.labels, number_of_targets);

    let mut columns = vec![];
    for col in 0..dataset.features[0].len() {
        let mut column = vec![];
        for row in 0..dataset.features.len() {
            column.push(dataset.features[row][col]);
        }
        columns.push(column);
    }

    MultiTargetDataSet {
        feature_rows: dataset.features,
        feature_columns: columns,
        labels: multi_target_labels,
    }
}

pub fn read_csv_data_multi_target(
    file_path_to_features: &str,
    file_path_to_labels: &str,
) -> MultiTargetDataSet {
    let data_set_features = read_data(file_path_to_features).unwrap();
    let data_set_labels = read_data(file_path_to_labels).unwrap();

    let columns = create_feature_columns(&data_set_features);

    MultiTargetDataSet {
        feature_rows: data_set_features,
        feature_columns: columns,
        labels: data_set_labels,
    }
}

pub fn get_feature_names(file_path: &str) -> Vec<String> {
    let feature_names = get_header_record(file_path).unwrap();
    let mut names_vec = vec![];
    for i in 0..feature_names.len() {
        names_vec.push(feature_names.get(i).unwrap().to_owned());
    }
    names_vec
}

fn get_header_record(file_path: &str) -> Result<StringRecord, Box<dyn Error>> {
    //feature names should be in the header of the csv file
    let file = File::open(file_path)?;
    let mut reader = csv::Reader::from_reader(file);
    let headers = reader.headers()?;
    Ok(headers.to_owned())
}

//reading in data from csv, presume header included and label is at the end of each record
fn read_data(file_path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let error_msg = "Failed to open ".to_owned() + file_path;
    let file = File::open(file_path).expect(&error_msg);
    let mut data = vec![];
    let mut reader = csv::Reader::from_reader(file);
    for result in reader.deserialize() {
        let record: Vec<f64> = result?;
        data.push(record);
    }
    Ok(data)
}

fn parse_data_into_features_and_labels(data_set: Vec<Vec<f64>>) -> SingleTargetDataSet {
    let mut features = vec![];
    let mut labels = vec![];
    data_set.iter().for_each(|row| {
        let mut copy = row.clone();
        let label = copy.pop().unwrap();
        labels.push(label);
        features.push(copy);
    });
    SingleTargetDataSet { features, labels }
}

fn create_multi_target_labels(labels: Vec<f64>, number_of_targets: usize) -> Vec<Vec<f64>> {
    let mut multi_target_labels = vec![];
    labels.iter().for_each(|label| {
        let mut multi_target = vec![0.; number_of_targets];
        multi_target[*label as usize] = 1.;
        multi_target_labels.push(multi_target);
    });
    multi_target_labels
}

pub fn create_feature_columns(data_set_features: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut columns = vec![];
    for col in 0..data_set_features[0].len() {
        let mut column = vec![];
        for row in 0..data_set_features.len() {
            column.push(data_set_features[row][col]);
        }
        columns.push(column);
    }
    columns
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_csv_reading_and_mt_one_hot_labels() {
        let data_set = read_csv_data("./data-files/iris.csv");
        let mt_labels = create_multi_target_labels(data_set.labels, 3);
        assert_eq!(*mt_labels.get(0).unwrap(), vec![1., 0., 0.]);

        let columns = create_feature_columns(&data_set.features);

        let multi_target_dataset = MultiTargetDataSet {
            feature_rows: data_set.features.clone(),
            feature_columns: columns,
            labels: mt_labels,
        };
        println!("{:?}", multi_target_dataset);
    }

    #[test]
    fn print_csv_reading_and_mt_labels() {
        let data_set = read_csv_data_multi_target(
            "./../common/data-files/multi-target/features_train_mt.csv",
            "./../common/data-files/multi-target/labels_train_mt.csv",
        );
        println!("{:?}", &data_set.labels);
    }

    #[test]
    fn print_csv_reading() {
        let data_set = read_csv_data("./data-files/iris.csv");
        println!("{:?}", data_set);
    }

    #[test]
    fn print_get_feature_names() {
        let feature_names = get_header_record("./data-files/iris.csv");
        let names_unwrapped = feature_names.unwrap();
        let mut names_vec = vec![];
        for i in 0..names_unwrapped.len() - 1 {
            names_vec.push(names_unwrapped.get(i).unwrap());
        }
        println!("{:?}", names_vec)
    }
}
