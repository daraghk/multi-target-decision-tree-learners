use crate::question::Question;

#[derive(Debug)]
pub struct TreeNode<'a> {
    question: Question,
    true_branch: &'a Vec<Vec<i32>>,
    false_branch: &'a Vec<Vec<i32>>,
}

impl<'a> TreeNode<'a> {
    pub fn new(
        question: Question,
        true_branch: &'a Vec<Vec<i32>>,
        false_branch: &'a Vec<Vec<i32>>,
    ) -> Self {
        Self {
            question,
            true_branch,
            false_branch,
        }
    }
}
