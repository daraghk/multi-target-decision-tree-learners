use super::leaf::Leaf;
use common::question::Question;

#[derive(Debug)]
pub struct TreeNode {
    pub question: Question,
    pub true_branch: Option<Box<TreeNode>>,
    pub false_branch: Option<Box<TreeNode>>,
    pub leaf: Option<Leaf>,
}

impl TreeNode {
    pub fn new(
        question: Question,
        true_branch: Box<TreeNode>,
        false_branch: Box<TreeNode>,
    ) -> Self {
        Self {
            question,
            true_branch: Some(true_branch),
            false_branch: Some(false_branch),
            leaf: None,
        }
    }

    pub fn leaf_node(question: Question, leaf: Leaf) -> Self {
        Self {
            question,
            true_branch: None,
            false_branch: None,
            leaf: Some(leaf),
        }
    }
}
