use std::ops::{Add, Sub, Mul};

// ---------------------------
// Fixed-size generic 2D vector
// ---------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vec2<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y
    }
}

// ---------------------------
// Fixed-size generic 3D vector
// ---------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vec3<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
}

// ---------------------------
// Dynamic-size generic vector
// ---------------------------
#[derive(Debug, Clone, PartialEq)]
pub struct DVec<T> {
    data: Vec<T>,
}

impl<T> DVec<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn new(data: Vec<T>) -> Self {
        assert!(!data.is_empty(), "DVec cannot be empty");
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn dot(&self, other: &Self) -> T {
        assert_eq!(
            self.len(),
            other.len(),
            "Vectors must have the same length"
        );
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a * *b)
            .fold(self.data[0] - self.data[0], |acc, val| acc + val)
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Add;
    use crate::linalg::vectors::Vec2;

    #[test]
    fn test_2d_vec_dot() {
        let x = Vec2::new(1.0, 2.0);
        let y = Vec2::new(3.0, 4.0);

        assert_eq!(x.dot(y), 11.0);
    }
    
    fn test_2d_vec_arithmetic() {
        todo!("unimplemented");
    }

}
