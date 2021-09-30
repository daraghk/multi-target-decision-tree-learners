use crate::class_counter::ClassCounter;

#[derive(Debug)]
pub struct Leaf{
    pub predictions : ClassCounter<i32, i32>
}