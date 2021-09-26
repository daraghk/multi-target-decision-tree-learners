#[derive(Debug)]
pub struct Question{
    pub column: u32,
    column_is_categorical: bool,
    pub value: i32
}

impl Question{
    pub fn new(column: u32, column_is_categorical: bool, value: i32) -> Self{
        Self{
            column,
            column_is_categorical,
            value
        }
    }

    pub fn solve(&self, row: &Vec<i32>) -> bool{
        let val_to_check = row[self.column as usize];
        if self.column_is_categorical{
            return val_to_check == self.value;
        }
        val_to_check >= self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_question_solve_numerical_true(){
        let question = Question::new(0, false, 1);
        let data_row = vec![1, 2, 4];
        assert_eq!(question.solve(&data_row), true);
    }

    #[test]
    fn test_question_solve_numerical_false(){
        let question = Question::new(1, false, 5);
        let data_row = vec![1, 2, 4];
        assert_eq!(question.solve(&data_row), false);
    }

    #[test]
    fn test_question_solve_categorical_true(){
        let question = Question::new(0, true, 1);
        let data_row = vec![1, 2, 4];
        assert_eq!(question.solve(&data_row), true);
    }

    #[test]
    fn test_question_solve_categorical_false(){
        let question = Question::new(1, true, 5);
        let data_row = vec![1, 2, 4];
        assert_eq!(question.solve(&data_row), false);
    }
}