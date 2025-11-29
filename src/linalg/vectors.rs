use std::ops::{Add, Mul, Sub};

// ---------------------------
// Fixed-size generic 2D vector
// ---------------------------
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

#[derive(Debug, PartialEq)]
pub enum VectorError {
    DimensionMismatch,
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

impl<T> Add for Vec2<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Vec2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T> Sub for Vec2<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

// Vec2 * scalar (T)
impl<T> Mul<T> for Vec2<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        Vec2 {
            x: self.x * scalar,
            y: self.y * scalar,
        }
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

impl<T> Add for Vec3<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T> Sub for Vec3<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T> Mul<T> for Vec3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, scalar: T) -> Self::Output {
        Vec3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
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

    pub fn dot(&self, other: &Self) -> Result<T, VectorError> {
        if self.len() != other.len() {
            return Err(VectorError::DimensionMismatch);
        }

        Ok(self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| *a * *b)
            .fold(self.data[0] - self.data[0], |acc, val| acc + val))
    }
}

impl<T> Add for DVec<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    type Output = Result<DVec<T>, VectorError>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<T> Add for &DVec<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    type Output = Result<DVec<T>, VectorError>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            return Err(VectorError::DimensionMismatch);
        }
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| *a + *b)
            .collect();

        Ok(DVec { data: result_data })
    }
}

impl<T> Sub for DVec<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    type Output = Result<DVec<T>, VectorError>;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<T> Sub for &DVec<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    type Output = Result<DVec<T>, VectorError>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.len() != rhs.len() {
            return Err(VectorError::DimensionMismatch);
        }
        let result_data: Vec<T> = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(a, b)| *a - *b)
            .collect();

        Ok(DVec { data: result_data })
    }
}

impl<T> Mul<T> for DVec<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, scalar: T) -> Self::Output {
        DVec {
            data: self.data.iter().map(|x| *x * scalar).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::vectors::{DVec, Vec2, Vec3, VectorError};

    #[test]
    fn test_2d_vec_dot() {
        let x = Vec2::new(1.0, 2.0);
        let y = Vec2::new(3.0, 4.0);

        assert_eq!(x.dot(y), 11.0);
    }

    #[test]
    fn test_2d_vec_arithmetic() {
        let x = Vec2::new(1.0, 2.0);
        let y = Vec2::new(3.0, 4.0);

        assert_eq!(x + y, Vec2::new(4.0, 6.0));
        assert_eq!(x - y, Vec2::new(-2.0, -2.0));
        assert_eq!(x * 2f32, Vec2::new(2.0, 4.0));
    }

    #[test]
    fn test_3d_vec_arithmetic() {
        let x = Vec3::new(1.0, 2.0, 3.0);
        let y = Vec3::new(3.0, 4.0, 5.0);

        assert_eq!(x + y, Vec3::new(4.0, 6.0, 8.0));
        assert_eq!(x - y, Vec3::new(-2.0, -2.0, -2.0));
        assert_eq!(x * 2f32, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_nd_vec_arithmetic() {
        let x = DVec::new(vec![1.0, 2.0, 3.0, 4.0]);
        let y = DVec::new(vec![3.0, 4.0, 5.0, 6.0]);

        assert_eq!((&x + &y).unwrap(), DVec::new(vec![4.0, 6.0, 8.0, 10.0]));
        assert_eq!(x.dot(&y).unwrap(), 50.0);
        assert_eq!((&x - &y).unwrap(), DVec::new(vec![-2.0, -2.0, -2.0, -2.0]));
        assert_eq!(x * 2f32, DVec::new(vec![2.0, 4.0, 6.0, 8.0]));
    }

    #[test]
    fn test_dvec_dimension_mismatch() {
        let x = DVec::new(vec![1.0, 2.0, 3.0]);
        let y = DVec::new(vec![1.0, 2.0]);

        assert_eq!((&x + &y).unwrap_err(), VectorError::DimensionMismatch);
    }
}
