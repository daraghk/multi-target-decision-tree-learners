use common::question::Question;

use crate::leaf::Leaf;

#[derive(Debug)]
pub struct TreeNode<L: Leaf> {
    pub question: Question,
    pub true_branch: Option<Box<TreeNode<L>>>,
    pub false_branch: Option<Box<TreeNode<L>>>,
    pub leaf: Option<L>,
}

impl<L: Leaf> TreeNode<L> {
    pub fn new(
        question: Question,
        true_branch: Box<TreeNode<L>>,
        false_branch: Box<TreeNode<L>>,
    ) -> Self {
        Self {
            question,
            true_branch: Some(true_branch),
            false_branch: Some(false_branch),
            leaf: None,
        }
    }

    pub fn leaf_node(question: Question, leaf: L) -> Self {
        Self {
            question,
            true_branch: None,
            false_branch: None,
            leaf: Some(leaf),
        }
    }

    pub fn is_leaf_node(&self) -> bool {
        self.true_branch.is_none() && self.false_branch.is_none()
    }
}
