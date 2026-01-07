use std::iter::Sum;
use std::ops::{Add, BitAnd, BitXor, Shr};

pub fn average_int<T>(x: T, y: T) -> T
where
    T: Copy + BitAnd<Output = T> + BitXor<Output = T> + Shr<u32, Output = T> + Add<Output = T>,
{
    (x & y) + ((x ^ y) >> 1)
}

pub fn average_f64<T>(numbers: &[T]) -> f64
where
    T: Copy + Into<f64> + Sum<T>,
{
    let sum: f64 = numbers.iter().copied().map(|x| x.into()).sum();
    sum / (numbers.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_average_int() {
        assert_eq!(average_int(4, 3), 3);
    }

    proptest! {
        #[test]
        fn prop_average_between(x in 0u8..=255, y in 0u8..=255) {
            let avg = average_int(x, y);
            prop_assert!(avg >= x.min(y) && avg <= x.max(y));
        }
    }

    proptest! {
        #[test]
        fn prop_average_matches_manual(x in 0u8..=255, y in 0u8..=255) {
            let avg = average_int(x, y);
            let manual_avg = ((x as u16 + y as u16) / 2) as u8;
            prop_assert_eq!(avg, manual_avg);
        }
    }

    proptest! {
        #[test]
        fn prop_average_symmetric(x in 0u8..=255, y in 0u8..=255) {
            prop_assert_eq!(average_int(x, y), average_int(y, x));
        }
    }
}