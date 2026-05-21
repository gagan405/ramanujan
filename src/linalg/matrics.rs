#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    RankMismatch {
        expect: usize,
        actual: usize,
    },
    ShapeMismatch {
        expect: usize,
        actual: usize,
    },
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
}

impl std::error::Error for TensorError {}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::RankMismatch { expect, actual } => {
                write!(f, "rank mismatch: expected {expect}, got {actual}")
            }
            TensorError::ShapeMismatch { expect, actual } => write!(
                f,
                "shape mismatch: expected {expect} elements, got {actual}"
            ),
            TensorError::IndexOutOfBounds { index, shape } => {
                write!(f, "index {index:?} out of bounds for shape {shape:?}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T> Tensor<T>
where
    T: Copy,
{
    pub fn new(data: Vec<T>, shape: Vec<usize>, strides: Vec<usize>) -> Result<Self, TensorError> {
        let expect: usize = shape.iter().product();
        if data.len() != expect {
            return Err(TensorError::ShapeMismatch {
                expect,
                actual: data.len(),
            });
        }
        if shape.len() != strides.len() {
            return Err(TensorError::RankMismatch {
                expect: shape.len(),
                actual: strides.len(),
            });
        }
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    pub fn index(&self, idx: &[usize]) -> Result<&T, TensorError> {
        if idx.len() != self.shape.len() {
            return Err(TensorError::RankMismatch {
                expect: self.shape.len(),
                actual: idx.len(),
            });
        }

        for (_dim, (&i, &dim_size)) in idx.iter().zip(self.shape.iter()).enumerate() {
            if i >= dim_size {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx.to_vec(),
                    shape: self.shape.clone(),
                });
            }
        }

        let offset: usize = idx
            .iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum();

        Ok(&self.data[offset])
    }
}

#[cfg(test)]
mod tests {
    use crate::linalg::matrics::{Tensor, TensorError};

    #[test]
    fn test_new_1d() {
        let t = Tensor::new(vec![1, 2, 3, 4], vec![4], vec![1]).unwrap();
        assert_eq!(*t.index(&[0]).unwrap(), 1);
        assert_eq!(*t.index(&[3]).unwrap(), 4);
    }

    #[test]
    fn test_new_2d_row_major() {
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3], vec![3, 1]).unwrap();
        assert_eq!(*t.index(&[0, 0]).unwrap(), 1);
        assert_eq!(*t.index(&[0, 2]).unwrap(), 3);
        assert_eq!(*t.index(&[1, 0]).unwrap(), 4);
        assert_eq!(*t.index(&[1, 2]).unwrap(), 6);
    }

    #[test]
    fn test_new_3d() {
        let data: Vec<i32> = (0..8).collect();
        let t = Tensor::new(data, vec![2, 2, 2], vec![4, 2, 1]).unwrap();
        assert_eq!(*t.index(&[0, 0, 0]).unwrap(), 0);
        assert_eq!(*t.index(&[0, 1, 1]).unwrap(), 3);
        assert_eq!(*t.index(&[1, 1, 1]).unwrap(), 7);
    }

    #[test]
    fn test_new_shape_mismatch() {
        let err = Tensor::new(vec![1, 2, 3], vec![2, 2], vec![2, 1]).unwrap_err();
        assert_eq!(
            err,
            TensorError::ShapeMismatch {
                expect: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn test_new_rank_mismatch() {
        let err = Tensor::new(vec![1, 2, 3, 4], vec![2, 2], vec![2]).unwrap_err();
        assert_eq!(
            err,
            TensorError::RankMismatch {
                expect: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn test_index_rank_mismatch() {
        let t = Tensor::new(vec![1, 2, 3, 4], vec![2, 2], vec![2, 1]).unwrap();
        let err = t.index(&[0]).unwrap_err();
        assert_eq!(
            err,
            TensorError::RankMismatch {
                expect: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn test_index_out_of_bounds() {
        let t = Tensor::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3], vec![3, 1]).unwrap();
        let err = t.index(&[2, 0]).unwrap_err();
        assert_eq!(
            err,
            TensorError::IndexOutOfBounds {
                index: vec![2, 0],
                shape: vec![2, 3],
            }
        );

        let err = t.index(&[0, 3]).unwrap_err();
        assert_eq!(
            err,
            TensorError::IndexOutOfBounds {
                index: vec![0, 3],
                shape: vec![2, 3],
            }
        );
    }

    #[test]
    fn test_custom_strides() {
        // column-major layout: shape [2, 3], strides [1, 2]
        let t = Tensor::new(vec![1, 4, 2, 5, 3, 6], vec![2, 3], vec![1, 2]).unwrap();
        assert_eq!(*t.index(&[0, 0]).unwrap(), 1);
        assert_eq!(*t.index(&[0, 1]).unwrap(), 2);
        assert_eq!(*t.index(&[1, 2]).unwrap(), 6);
    }
}
