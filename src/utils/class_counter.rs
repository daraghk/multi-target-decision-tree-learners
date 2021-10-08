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

pub fn get_class_counts(classes: &Vec<i32>, number_of_unique_classes: u32) -> ClassCounter {
    let mut class_counter = ClassCounter::new(number_of_unique_classes);
    classes.iter().for_each(|class| {
        class_counter.counts[*class as usize] += 1;
    });
    class_counter
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_class_counts_one_row_and_class() {
        //set up data
        let number_classes = 1;
        let data = vec![0];
        let class_counts = get_class_counts(&data, number_classes);
        assert_eq!(class_counts.counts.get(0).unwrap(), &1);
    }

    #[test]
    fn test_get_class_counts_multiple_rows_and_classes() {
        //set up data
        let number_classes = 2;
        let data = vec![0, 0, 1];
        let class_counts = get_class_counts(&data, number_classes);
        assert_eq!(class_counts.counts.get(0).unwrap(), &2);
        assert_eq!(class_counts.counts.get(1).unwrap(), &1);
    }
}
