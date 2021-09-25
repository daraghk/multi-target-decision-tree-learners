#![allow(unused)]
use serde::{Deserialize, Serialize};
use std::fs; // imports both the trait and the derive macro

#[path = "utils/calculations.rs"]
mod calculations;
#[path = "utils/class_counter.rs"]
mod class_counter;
#[path = "utils/question.rs"]
mod question;
#[path = "threshold_finders/threshold_finder.rs"]
mod threshold_finder;

fn main() {
    let _data = fs::read_to_string("./data_arff/iris.arff").expect("Unable to read file");
    let unnamed_data: Vec<Vec<i32>> = arff::from_str(&_data).unwrap();
    println!("{:?}", unnamed_data[0]);
}
