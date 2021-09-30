use std::collections::HashMap;

#[derive(Debug)]
pub struct ClassCounter<K, V> {
    pub map: HashMap<K, V>,
}

impl<K, V> ClassCounter<K, V> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }
}