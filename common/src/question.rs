// Question struct purpose:
// Given a data row, is this rows value at question.column >= question.value?
#[derive(Debug)]
pub struct Question {
    pub column: u32,
    pub value: f64,
}

impl Question {
    pub fn new(column: u32, value: f64) -> Self {
        Self { column, value }
    }

    pub fn solve(&self, row: &[f64]) -> bool {
        let val_to_check = row[self.column as usize];
        val_to_check >= self.value
    }

    pub fn to_string(&self, feature_name: &str) -> String {
        format!("Is {} >= {}", feature_name, self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_question_solve_numerical_true() {
        let question = Question::new(0, 1.0);
        let data_row = vec![1.0, 2.0, 4.0];
        assert_eq!(question.solve(&data_row), true);
    }

    #[test]
    fn test_question_solve_numerical_false() {
        let question = Question::new(1, 5.0);
        let data_row = vec![1.0, 2.0, 4.0];
        assert_eq!(question.solve(&data_row), false);
    }

    #[test]
    fn test_question_solve_categorical_true() {
        let question = Question::new(0, 1.0);
        let data_row = vec![1.0, 2.0, 4.0];
        assert_eq!(question.solve(&data_row), true);
    }

    #[test]
    fn test_question_solve_categorical_false() {
        let question = Question::new(1, 5.);
        let data_row = vec![1.0, 2.0, 4.0];
        assert_eq!(question.solve(&data_row), false);
    }
}
