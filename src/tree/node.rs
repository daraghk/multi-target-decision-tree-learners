use crate::question::Question;

pub struct Node<'a>{
    question: Question,
    true_branch: &'a Vec<Vec<i32>>,
    false_branch: &'a Vec<Vec<i32>>,
}