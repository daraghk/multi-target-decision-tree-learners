use common::datasets::MultiTargetDataSet;
use multi_target_decision_tree::decision_tree::TreeConfig;

struct GradientBooster {
    data: MultiTargetDataSet,
    tree_config: TreeConfig,
    number_of_iterations: u32,
}

impl GradientBooster {
    pub fn train_gradient_booster() {}

    pub fn infer_using_gradient_booster() {}
}
