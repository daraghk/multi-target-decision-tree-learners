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
#[path = "tree/decision_tree.rs"]
mod decision_tree;
#[path = "data_reader/data.rs"]
mod data;

fn main() {
}
