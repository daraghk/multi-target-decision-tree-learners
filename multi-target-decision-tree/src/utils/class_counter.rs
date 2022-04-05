#[derive(Debug, Clone)]
pub struct ClassCounter {
    pub counts: Vec<u32>,
}

impl ClassCounter {
    pub fn new(number_of_classes: usize) -> Self {
        Self {
            counts: vec![0; number_of_classes],
        }
    }
}
