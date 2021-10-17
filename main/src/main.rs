use common::data_reader::{read_csv_data, read_csv_data_multi_target};
use decision_tree::decision_tree::DecisionTree;
use multi_target_decision_tree::decision_tree::MultiTargetDecisionTree;

fn main() {
    let multi_target_dataset = read_csv_data_multi_target("./common/data_files/iris.csv", 3);
    let multi_target_split_finder = multi_target_decision_tree::split_finder::SplitFinder::new(
        multi_target_decision_tree::split_finder::SplitMetric::Variance,
    );
    let multi_target_decision_tree =
        MultiTargetDecisionTree::new(multi_target_dataset, multi_target_split_finder, 3);
    multi_target_decision_tree::decision_tree::print_tree(
        Box::new(multi_target_decision_tree.root),
        "".to_string(),
    );

    let dataset = read_csv_data("./common/data_files/iris.csv");
    let split_finder = decision_tree::split_finder::SplitFinder::new(
        decision_tree::split_finder::SplitMetric::Variance,
    );
    let decision_tree = DecisionTree::new(dataset, split_finder, 3);
    decision_tree::decision_tree::print_tree(Box::new(decision_tree.root), "".to_string());

    let multi_target_dataset_other = read_csv_data_multi_target("./common/data_files/wine.csv", 3);
    let multi_target_split_finder_other =
        multi_target_decision_tree::split_finder::SplitFinder::new(
            multi_target_decision_tree::split_finder::SplitMetric::Variance,
        );
    let multi_target_decision_tree_other = MultiTargetDecisionTree::new(
        multi_target_dataset_other,
        multi_target_split_finder_other,
        3,
    );
    multi_target_decision_tree::decision_tree::print_tree(
        Box::new(multi_target_decision_tree_other.root),
        "".to_string(),
    );
}
