#![allow(unused)]

#[path = "utils/calculations/calculations.rs"]
mod calculations;
#[path = "utils/class_counter.rs"]
mod class_counter;
#[path = "utils/classifier.rs"]
mod classifier;
#[path = "utils/data_partitioner.rs"]
mod data_partitioner;
#[path = "tree/decision_tree.rs"]
pub mod decision_tree;
#[path = "utils/feature_sorter.rs"]
mod feature_sorter;
#[path = "tree/leaf.rs"]
pub mod leaf;
#[path = "tree/node.rs"]
pub mod node;
#[path = "split_finders/split_finder.rs"]
pub mod split_finder;
// #[path = "benches/bench.rs"]
// mod bench;
