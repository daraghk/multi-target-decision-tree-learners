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

pub fn get_class_counts_multi_target(classes: &Vec<Vec<f32>>, number_of_unique_classes: u32) -> ClassCounter {
    let mut class_counter = ClassCounter::new(number_of_unique_classes);
    classes.iter().for_each(|label_vector| {
        label_vector.iter().for_each(|class|{
            if *class == 1. {
                class_counter.counts[*class as usize] += 1;
            }
        })
    });
    class_counter
}

#[cfg(test)]
mod tests {
   
}