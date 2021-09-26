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