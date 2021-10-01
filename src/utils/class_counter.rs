use std::collections::HashMap;

#[derive(Debug)]
pub struct ClassCounter {
    pub counts: Vec<u32>,
}

impl ClassCounter {
    pub fn new(number_of_classes: u32) -> Self {
        Self {
            counts: vec![0; number_of_classes as usize],
        }
    }
}
